"""OSNet-AIN: OSNet with adaptively-placed instance normalization (X-ain tiers).

Architecture from Torchreid (MIT License, Copyright (c) 2018 Kaiyang Zhou),
"Learning Generalisable Omni-Scale Representations for Person
Re-Identification", TPAMI 2021. Vendored with module names kept identical to
Torchreid so the published ImageNet weights load key-for-key (verified
against the osnet_ain_x1_0 checkpoint: IN stem, block arrangement
[[INin, INin], [OSBlock, INin], [INin, OSBlock]], standalone pool2/pool3
transitions, and BN-free conv3 inside OSBlockINin). Exposes the TransReID
backbone interface: forward(x, cam_label, view_label) -> [B, feature_dim],
`in_planes`, and `load_param(path, hw_ratio)`.
"""

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, rep=False):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,
                               bias=False, groups=out_channels)
        # train-time structural reparameterization (RepVGG/MobileOne style):
        # two extra linear branches parallel to the depthwise 3x3 — a
        # depthwise 1x1 and a per-channel identity scale — summed BEFORE the
        # shared BN. Both are linear in the same input, so at export they
        # fold EXACTLY into conv2's center tap (tools/fold_rep.py) and the
        # deployment graph stays the plain osnet_ain_x1_0 graph at zero
        # cost. Zero init keeps warm starts function-preserving from step
        # one (new keys only); the value is the changed optimization
        # dynamics, not added inference capacity.
        if rep:
            self.rep_conv = nn.Conv2d(out_channels, out_channels, 1, stride=1,
                                      padding=0, bias=False, groups=out_channels)
            self.rep_id = nn.Parameter(torch.zeros(out_channels))
        else:
            self.rep_conv = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        if self.rep_conv is not None:
            y = y + self.rep_conv(x) + self.rep_id.view(1, -1, 1, 1) * x
        return self.relu(self.bn(y))


class LightConvStream(nn.Module):
    """Stacked LightConv3x3 stream of a given depth."""

    def __init__(self, in_channels, out_channels, depth, rep=False):
        super(LightConvStream, self).__init__()
        assert depth >= 1
        layers = [LightConv3x3(in_channels, out_channels, rep=rep)]
        for _ in range(depth - 1):
            layers.append(LightConv3x3(out_channels, out_channels, rep=rep))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelGate, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True,
                             padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return input * x


class OSBlock(nn.Module):
    """Omni-scale residual block (BatchNorm variant)."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, rep=False):
        super(OSBlock, self).__init__()
        assert T >= 1
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t, rep=rep))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return torch.relu(x3 + identity)


class OSBlockINin(nn.Module):
    """Omni-scale residual block with instance normalization on the branch."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, rep=False):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t, rep=rep))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return torch.relu(x3 + identity)


class LiteSelfAttention(nn.Module):
    """Bottlenecked residual self-attention over the final feature map.

    OSNet builds local omni-scale features; this single block adds the
    global pairwise-relation modeling the conv stack lacks (the structural
    gap behind the CNN students' inability to match the ViT teacher's
    similarity geometry). The 512 -> dim -> 512 bottleneck keeps it light
    (~0.2M parameters at dim=128; 16x8 = 128 tokens, so the attention
    matrix is tiny), and the zero-initialized scalar gate makes the block
    an exact identity at initialization — trained checkpoints warm-start
    unchanged, and the learned gate magnitude doubles as a readout of how
    much attention the model recruits.
    """

    def __init__(self, channels, dim=128, num_heads=4):
        super(LiteSelfAttention, self).__init__()
        # attention is written out with plain matmul/softmax instead of
        # nn.MultiheadAttention: the latter takes the fused
        # aten::_native_multi_head_attention fast path in eval mode, which
        # has no ONNX opset-17 export
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # dim == channels removes the bottleneck entirely: attention operates
        # at native width and the residual update can span every direction
        self.reduce = (nn.Conv2d(channels, dim, 1, bias=False)
                       if dim != channels else None)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.expand = (nn.Conv2d(dim, channels, 1, bias=False)
                       if dim != channels else None)
        # small non-zero init: an exactly-zero gate blocks every gradient
        # into the block (bootstrap deadlock; with weight decay the unused
        # weights then decay to zero — observed in the attn_nokd arm). 0.01
        # perturbs the warm-started output by ~0.006 while letting the
        # internals train from step one. Weight decay is additionally
        # disabled for '.attn.' parameters in solver/make_optimizer.py.
        self.gate = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x):
        identity = x
        if self.reduce is not None:
            x = self.reduce(x)
        batch, dim, height, width = x.shape
        tokens = height * width
        x = x.flatten(2).transpose(1, 2)  # B, T, dim
        x = self.norm(x)
        # rank-4 head split only (the OSNet export contract forbids the
        # rank-5 qkv transpose that identifies ViT graphs)
        query, key, value = self.qkv(x).chunk(3, dim=-1)  # each B, T, dim
        query = query.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / (self.head_dim ** 0.5)
        x = scores.softmax(dim=-1) @ value  # B, heads, T, head_dim
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(batch, dim, height, width)
        if self.expand is not None:
            x = self.expand(x)
        return identity + self.gate * x


class SeparableSelfAttention(nn.Module):
    """O(N) separable self-attention (MobileViTv2-style) for early feature maps.

    Entrance placement puts attention where the token count is 16x the tail
    (64x32 = 2048 tokens after the stem vs 16x8 = 128 after conv5), so the
    quadratic LiteSelfAttention is unaffordable there (+55% of the whole P
    tier). This variant replaces the T x T attention matrix with a single
    softmax-weighted context vector: scores = softmax over tokens of a 1-ch
    projection, context = score-weighted sum of keys (one d-vector), output
    = proj(relu(values) * context). Cost is O(T*d) — ~25M MACs (+2.6% of P)
    at the stem — and the graph is NCHW-native: no attention matrix, no head
    split, no transposes; only two flatten-to-rank-3 reshapes for the token
    softmax/sum.
    """

    def __init__(self, channels):
        super(SeparableSelfAttention, self).__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # same bootstrap contract as LiteSelfAttention: small non-zero init
        # (exactly-zero blocks every gradient into the block) and weight
        # decay disabled in solver/make_optimizer.py
        self.gate = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x):
        batch, channels, height, width = x.shape
        scores = self.score(x).flatten(2).softmax(dim=-1)       # B, 1, T
        context = (self.key(x).flatten(2) * scores).sum(dim=-1)  # B, C
        context = context.reshape(batch, channels, 1, 1)
        out = self.proj(torch.relu(self.value(x)) * context)
        return x + self.gate * out


class OSNetAIN(nn.Module):
    """OSNet-AIN feature extractor with the TransReID backbone interface."""

    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512,
                 extra_blocks=(0, 0, 0), attn_dim=None, attn_heads=4,
                 stem_in_only=False, sep_stem_attn=False, rep=False):
        super(OSNetAIN, self).__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # optional entrance attention right after the stem (post-IN/maxpool):
        # every later conv then processes globally-contextualized features.
        # Separable (O(N)) because 2048 tokens make quadratic attention
        # unaffordable here. New keys only -> checkpoints warm-start.
        self.stem_attn = (SeparableSelfAttention(channels[0])
                          if sep_stem_attn else None)
        # searched arrangement of osnet_ain_x1_0; extra_blocks appends plain
        # OSBlocks at the end of each stage (existing parameter names stay
        # unchanged, and the searched IN placement is not disturbed) —
        # identity-initialize the appended blocks via tools/init_depth_expand.py.
        # stem_in_only=True keeps ONLY the conv1 stem IN and replaces every
        # OSBlockINin with a plain (BN) OSBlock — the IN-information-loss
        # hypothesis probe: spatial IN discards instance statistics 5x, which
        # may be what makes the L_cam teacher geometry unrepresentable.
        inin = OSBlock if stem_in_only else OSBlockINin
        self.conv2 = nn.Sequential(
            inin(channels[0], channels[1], rep=rep),
            inin(channels[1], channels[1], rep=rep),
            *[OSBlock(channels[1], channels[1], rep=rep) for _ in range(extra_blocks[0])],
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            OSBlock(channels[1], channels[2], rep=rep),
            inin(channels[2], channels[2], rep=rep),
            *[OSBlock(channels[2], channels[2], rep=rep) for _ in range(extra_blocks[1])],
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            inin(channels[2], channels[3], rep=rep),
            OSBlock(channels[3], channels[3], rep=rep),
            *[OSBlock(channels[3], channels[3], rep=rep) for _ in range(extra_blocks[2])],
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        # optional global-relation block between conv5 and GAP (identity at
        # init via its zero gate; new keys only, so checkpoints warm-start)
        self.attn = (LiteSelfAttention(channels[3], attn_dim, attn_heads)
                     if attn_dim else None)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.in_planes = feature_dim
        self._init_params()
        # _init_params kaiming-inits every Conv2d; re-zero the rep branches
        # afterwards so a rep model is function-identical to its plain
        # warm-start checkpoint at step one
        for m in self.modules():
            if isinstance(m, LightConv3x3) and m.rep_conv is not None:
                nn.init.zeros_(m.rep_conv.weight)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, cam_label=None, view_label=None):
        x = self.conv1(x)
        x = self.maxpool(x)
        if self.stem_attn is not None:
            x = self.stem_attn(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def load_param(self, model_path, hw_ratio=None):
        param_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        count = 0
        for k, v in param_dict.items():
            k = k.replace('module.', '')
            if 'classifier' in k:
                continue
            if k not in self.state_dict():
                print('warning. skip {} params'.format(k))
                continue
            if self.state_dict()[k].shape != v.shape:
                print('warning. shape mismatch, skip {}'.format(k))
                continue
            self.state_dict()[k].copy_(v)
            count += 1
        print('Load {} / {} layers.'.format(count, len(self.state_dict())))


def osnet_ain_x1_0(**kwargs):
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512)


def osnet_ain_x1_25(**kwargs):
    # custom multiplier (no ImageNet zoo weights); init via tools/init_width_expand.py
    return OSNetAIN(channels=(80, 320, 480, 640), feature_dim=512)


def osnet_ain_x1_5(**kwargs):
    # custom multiplier (no ImageNet zoo weights); init via tools/init_width_expand.py
    return OSNetAIN(channels=(96, 384, 576, 768), feature_dim=512)


def osnet_ain_x0_75(**kwargs):
    return OSNetAIN(channels=(48, 192, 288, 384), feature_dim=512)


def osnet_ain_x1_0_attn(**kwargs):
    # + one LiteSelfAttention block after conv5; warm-starts from plain
    # osnet_ain_x1_0 checkpoints (identity at init via the zero gate)
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512, attn_dim=128)


def osnet_ain_x1_25_attn(**kwargs):
    return OSNetAIN(channels=(80, 320, 480, 640), feature_dim=512, attn_dim=128)


def osnet_ain_x1_5_attn(**kwargs):
    return OSNetAIN(channels=(96, 384, 576, 768), feature_dim=512, attn_dim=128)


def osnet_ain_stem_x1_0_attn(**kwargs):
    # IN-information-loss probe: stem IN only (1 InstanceNorm instead of 5),
    # plain BN OSBlocks elsewhere, plus the bottlenecked attention block
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512,
                    attn_dim=128, stem_in_only=True)


def osnet_ain_x1_0_rep(**kwargs):
    # train-time structural reparameterization: every depthwise 3x3 in the
    # LightConv streams gains two zero-init linear branches (dw 1x1 +
    # per-channel identity scale). Fold with tools/fold_rep.py before
    # export/eval — the deployment graph and cost are EXACTLY osnet_ain_x1_0.
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512, rep=True)


def osnet_ain_x1_0_sepattn(**kwargs):
    # entrance-attention probe on the standard (5-IN) architecture: one O(N)
    # separable attention block right after the stem, plus the gated tail
    # LiteSelfAttention (kept by decision: both gates read out independently)
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512,
                    sep_stem_attn=True, attn_dim=128)


def osnet_ain_stem_x1_0_sepattn(**kwargs):
    # entrance-attention probe on the stem-IN-only lineage: 1 InstanceNorm
    # (conv1 stem), plain BN OSBlocks elsewhere, one O(N) separable attention
    # block right after the stem, plus the gated tail LiteSelfAttention.
    # The tail block self-suppressed in all three prior environments but is
    # kept by decision — entrance context may change what the tail sees, and
    # its gate doubles as a free readout of that interaction.
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512,
                    stem_in_only=True, sep_stem_attn=True, attn_dim=128)


def osnet_ain_x1_0_attn_full(**kwargs):
    # bottleneck-free attention probe: native 512-dim, 8 heads (head_dim 64)
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512,
                    attn_dim=512, attn_heads=8)


def osnet_ain_x1_0_deep(**kwargs):
    # depth-expanded tier (+1 plain OSBlock per stage); init via tools/init_depth_expand.py
    return OSNetAIN(channels=(64, 256, 384, 512), feature_dim=512,
                    extra_blocks=(1, 1, 1))


def osnet_ain_x1_25_deep(**kwargs):
    # depth-expanded tier (+1 plain OSBlock per stage); init via tools/init_depth_expand.py
    return OSNetAIN(channels=(80, 320, 480, 640), feature_dim=512,
                    extra_blocks=(1, 1, 1))


def osnet_ain_x1_5_deep(**kwargs):
    # depth-expanded tier (+1 plain OSBlock per stage); init via tools/init_depth_expand.py
    return OSNetAIN(channels=(96, 384, 576, 768), feature_dim=512,
                    extra_blocks=(1, 1, 1))


def osnet_ain_x0_5(**kwargs):
    return OSNetAIN(channels=(32, 128, 192, 256), feature_dim=512)
