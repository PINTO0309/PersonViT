"""OSNet: omni-scale CNN backbone for person ReID (tiers P/F/A).

Architecture from Torchreid (MIT License, Copyright (c) 2018 Kaiyang Zhou),
"Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019.
Vendored as a single file with module names kept identical to Torchreid so
the published ImageNet weights load key-for-key. The class exposes the same
interface the TransReID pipeline expects from a backbone:
forward(x, cam_label, view_label) -> [B, feature_dim], `in_planes`, and
`load_param(path, hw_ratio)`.
"""

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
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
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 conv followed by a depthwise 3x3 conv."""

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,
                               bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Unified aggregation gate (channel attention)."""

    def __init__(self, in_channels, reduction=16):
        super(ChannelGate, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return input * x


class OSBlock(nn.Module):
    """Omni-scale residual block with four multi-scale streams."""

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + \
            self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x3 + identity)


class OSNet(nn.Module):
    """OSNet feature extractor with the TransReID backbone interface."""

    def __init__(self, layers=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=512):
        super(OSNet, self).__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.in_planes = feature_dim
        self._init_params()

    @staticmethod
    def _make_layer(num_blocks, in_channels, out_channels, reduce_spatial_size):
        layers = [OSBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(nn.Sequential(
                Conv1x1(out_channels, out_channels),
                nn.AvgPool2d(2, stride=2),
            ))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        x = self.conv3(x)
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


def osnet_x1_0(**kwargs):
    return OSNet(layers=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=512)


def osnet_x0_75(**kwargs):
    return OSNet(layers=(2, 2, 2), channels=(48, 192, 288, 384), feature_dim=512)


def osnet_x0_5(**kwargs):
    return OSNet(layers=(2, 2, 2), channels=(32, 128, 192, 256), feature_dim=512)
