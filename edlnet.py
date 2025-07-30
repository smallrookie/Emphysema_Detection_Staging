import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from scrb import SCRB


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, alpha=0.5, scrb=False):
        super().__init__()
        if scrb:
            self.block = nn.Sequential(
                nn.MaxPool2d(2),
                SCRConv(op_channel=in_channels, alpha=alpha),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_gate=False):
        super().__init__()

        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels // 2, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.gate = None
        self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        if self.gate:
            gate_weight = self.gate(x2)
            x2_weighted = gate_weight * x2
            x1_weighted = (1 - gate_weight) * x1.detach()
            x = torch.cat([x2_weighted, x1_weighted], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EDLNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        alpha=0.5,
        use_gate=False,
        use_scrb=False,
    ):
        super(EDLNet, self).__init__()
        self.use_gate = use_gate

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024, alpha=alpha, scrb=use_scrb)

        self.up1 = Up(1024, 512, use_gate)
        self.up2 = Up(512, 256, use_gate)
        self.up3 = Up(256, 128, use_gate)
        self.up4 = Up(128, 64, use_gate)

        self.outc = OutConv(64, n_classes)

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x0):
        x1 = self.inc(x0)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        recon_img = self.outc(x)
        return recon_img
