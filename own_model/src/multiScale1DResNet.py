# https://github.com/geekfeiw/Multi-Scale-1D-ResNet/blob/master/figs/network.png


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


def conv9x9(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=1, bias=False)


def conv13x13(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=113, stride=stride,
                     padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample:
        residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def _make_layer(in_channels, out_channels, kernel_size, stride, dilation):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels),
        )

    layers = [ResidualBlock(in_channels, out_channels, kernel_size, stride, dilation, downsample)]
    # Add more blocks as needed
    # for i in range(1, blocks):
    #     layers.append(ResidualBlock(out_channels, out_channels, kernel_size, stride, dilation, downsample))
    return nn.Sequential(*layers)


class MSResNet(nn.Module):
    def __init__(self, input_channel, num_classes=1):
        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=25, stride=2, padding=11,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer7x7_1 = _make_layer(64, 128, 7, stride=1, dilation=1)
        self.layer7x7_2 = _make_layer(128, 256, 7, stride=1, dilation=1)
        # self.layer7x7_3 = _make_layer(256, 512, 7, stride=1, dilation=1)

        self.layer9x9_1 = _make_layer(64, 128, 9, stride=1, dilation=1)
        self.layer9x9_2 = _make_layer(128, 256, 9, stride=1, dilation=1)
        # self.layer9x9_3 = _make_layer(256, 512, 9, stride=1, dilation=1)

        self.layer13x13_1 = _make_layer(64, 128, 13, stride=1, dilation=1)
        self.layer13x13_2 = _make_layer(128, 256, 13, stride=1, dilation=1)
        # self.layer13x13_3 = _make_layer(256, 512, 13, stride=1, dilation=1)

        # self.drop = nn.Dropout(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * 3, num_classes)

    def forward(self, src):
        x0 = self.conv1(src)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer7x7_1(x0)
        x = self.layer7x7_2(x)
        # x = self.layer7x7_3(x)
        # x = self.layer3x3_4(x)
        x = self.avgpool(x)

        y = self.layer9x9_1(x0)
        y = self.layer9x9_2(y)
        # y = self.layer9x9_3(y)
        # y = self.layer5x5_4(y)
        y = self.avgpool(y)

        z = self.layer13x13_1(x0)
        z = self.layer13x13_2(z)
        # z = self.layer13x13_3(z)
        # z = self.layer7x7_4(z)
        z = self.avgpool(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.squeeze()
        # out = self.drop(out)
        out1 = self.fc(out)

        return out1


class SimpleResNet(nn.Module):
    def __init__(self, input_channels: int, seq_len: int):
        super(SimpleResNet, self).__init__()

        self.initial_conv = nn.Conv1d(input_channels, 16, kernel_size=13, stride=1, padding=6)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        # self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(128)

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(32 * (seq_len // 4), 512)  # think this // 4 because two times pooling size 2
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DumbNet(nn.Module):
    def __init__(self, input_channels: int, seq_len: int, num_classes: int):
        super(DumbNet, self).__init__()

        self.flatten = nn.Flatten(1)

        self.input = nn.Linear(input_channels * seq_len, 128)

        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)

        self.out = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.out(x)
        return x
