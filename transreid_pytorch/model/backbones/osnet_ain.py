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
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,
                               bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    """Stacked LightConv3x3 stream of a given depth."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1
        layers = [LightConv3x3(in_channels, out_channels)]
        for _ in range(depth - 1):
            layers.append(LightConv3x3(out_channels, out_channels))
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

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super(OSBlock, self).__init__()
        assert T >= 1
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
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

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
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


class OSNetAIN(nn.Module):
    """OSNet-AIN feature extractor with the TransReID backbone interface."""

    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512,
                 extra_blocks=(0, 0, 0)):
        super(OSNetAIN, self).__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # searched arrangement of osnet_ain_x1_0; extra_blocks appends plain
        # OSBlocks at the end of each stage (existing parameter names stay
        # unchanged, and the searched IN placement is not disturbed) —
        # identity-initialize the appended blocks via tools/init_depth_expand.py
        self.conv2 = nn.Sequential(
            OSBlockINin(channels[0], channels[1]),
            OSBlockINin(channels[1], channels[1]),
            *[OSBlock(channels[1], channels[1]) for _ in range(extra_blocks[0])],
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            OSBlock(channels[1], channels[2]),
            OSBlockINin(channels[2], channels[2]),
            *[OSBlock(channels[2], channels[2]) for _ in range(extra_blocks[1])],
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            OSBlockINin(channels[2], channels[3]),
            OSBlock(channels[3], channels[3]),
            *[OSBlock(channels[3], channels[3]) for _ in range(extra_blocks[2])],
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.in_planes = feature_dim
        self._init_params()

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
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
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
