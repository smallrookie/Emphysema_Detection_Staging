import torch
import torch.nn as nn
import torch.nn.functional as F


class SCM(nn.Module):
    def __init__(self, in_ch, ms_kernel=1, dilation=1):
        super(SCM, self).__init__()
        self.pad = ms_kernel + dilation
        self.border_input = ms_kernel + 2 * dilation + 1

        self.dwconv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ms_kernel,
            groups=in_ch,
        )
        self.dwconv2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ms_kernel,
            groups=in_ch,
        )
        self.dwconv3 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ms_kernel,
            groups=in_ch,
        )
        self.dwconv4 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ms_kernel,
            groups=in_ch,
        )

        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()

    def forward(self, x_in):
        x = F.pad(x_in, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        x1 = self.dwconv1(x[:, :, : -self.border_input, : -self.border_input])
        x2 = self.dwconv2(x[:, :, self.border_input :, : -self.border_input])
        x3 = self.dwconv3(x[:, :, : -self.border_input, self.border_input :])
        x4 = self.dwconv4(x[:, :, self.border_input :, self.border_input :])
        x = self.act(self.bn(x1 + x2 + x3 + x4))
        return x


class CRM(nn.Module):
    def __init__(
        self,
        op_channel: int,
        alpha: float = 1 / 2,
        mlp_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super(CRM, self).__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel

        self.squeeze1 = nn.Conv2d(
            up_channel,
            up_channel * mlp_ratio,
            kernel_size=1,
            bias=False,
        )
        self.squeeze2 = nn.Conv2d(
            low_channel,
            low_channel * mlp_ratio,
            kernel_size=1,
            bias=False,
        )

        self.GWC = nn.Conv2d(
            up_channel * mlp_ratio,
            op_channel * mlp_ratio,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
        )
        self.PWC1 = nn.Conv2d(
            up_channel * mlp_ratio,
            op_channel * mlp_ratio,
            kernel_size=1,
            bias=False,
        )
        self.PWC2 = nn.Conv2d(
            low_channel * mlp_ratio,
            (op_channel - low_channel) * mlp_ratio,
            kernel_size=1,
            bias=False,
        )
        self.act = nn.GELU()
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up = self.squeeze1(up)
        low = self.squeeze2(low)

        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        out = self.act(Y1) * Y2
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCRB(nn.Module):
    def __init__(
        self,
        op_channel: int,
        ms_kernel: int = 1,
        dilation: int = 1,
        alpha: float = 1 / 2,
        mlp_ratio: int = 2,
    ):
        super(SCRB, self).__init__()
        self.scm = SCM(
            in_ch=op_channel,
            ms_kernel=ms_kernel,
            dilation=dilation,
        )
        self.crm = CRM(
            op_channel,
            alpha=alpha,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        x = self.scm(x)
        x = self.crm(x)
        return x
