"""RAF-DB RGB MiniResNet 模型。"""

from __future__ import annotations

import torch
from torch import nn


class RGBResidualBlock(nn.Module):
    """RGB MiniResNet 使用的基础残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class RGBMiniResNet(nn.Module):
    """适配 3 通道 96×96 RAF-DB 彩色图像的轻量残差网络。"""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        # 第一层卷积显式使用 3 通道输入，用于处理 RGB 彩色图像。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            RGBResidualBlock(32, 32),
            RGBResidualBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            RGBResidualBlock(32, 64, stride=2),
            RGBResidualBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            RGBResidualBlock(64, 128, stride=2),
            RGBResidualBlock(128, 128),
        )
        self.layer4 = nn.Sequential(
            RGBResidualBlock(128, 192, stride=2),
            RGBResidualBlock(192, 192),
        )
        # 全局平均池化让模型可以聚合空间特征，同时控制参数量。
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
