"""
SCRB: Scale-wise Convolutional Residual Block
Implementation of a scale-wise convolutional residual block for feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCM(nn.Module):
    """Scale-wise Convolution Module
    
    A module that performs multi-scale convolution operations using dilated convolutions
    to capture features at different scales.
    """

    def __init__(self, in_ch, ms_kernel=1, dilation=1):
        """Initialize the Scale-wise Convolution Module.
        
        Args:
            in_ch (int): Number of input channels
            ms_kernel (int): Kernel size for multi-scale convolutions
            dilation (int): Dilation factor for dilated convolutions
        """
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
        """Forward pass through the scale-wise convolution module.
        
        Args:
            x_in (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after multi-scale convolution
        """
        x = F.pad(x_in, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        x1 = self.dwconv1(x[:, :, : -self.border_input, : -self.border_input])
        x2 = self.dwconv2(x[:, :, self.border_input :, : -self.border_input])
        x3 = self.dwconv3(x[:, :, : -self.border_input, self.border_input :])
        x4 = self.dwconv4(x[:, :, self.border_input :, self.border_input :])
        x = self.act(self.bn(x1 + x2 + x3 + x4))
        return x


class CRM(nn.Module):
    """Channel-wise Residual Module
    
    A module that performs channel-wise residual operations to enhance feature representation
    by combining global and local features with attention mechanisms.
    """

    def __init__(
        self,
        op_channel: int,
        alpha: float = 1 / 2,
        mlp_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        """Initialize the Channel-wise Residual Module.
        
        Args:
            op_channel (int): Number of output channels
            alpha (float): Ratio for splitting channels
            mlp_ratio (int): Ratio for expanding channels in MLP
            group_size (int): Group size for grouped convolutions
            group_kernel_size (int): Kernel size for grouped convolutions
        """
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
        """Forward pass through the channel-wise residual module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after channel-wise residual operations
        """
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
    """Scale-wise Convolutional Residual Block
    
    Combines Scale-wise Convolution Module (SCM) and Channel-wise Residual Module (CRM)
    to extract enhanced features for medical image processing.
    """

    def __init__(
        self,
        op_channel: int,
        ms_kernel: int = 1,
        dilation: int = 1,
        alpha: float = 1 / 2,
        mlp_ratio: int = 2,
    ):
        """Initialize the Scale-wise Convolutional Residual Block.
        
        Args:
            op_channel (int): Number of output channels
            ms_kernel (int): Kernel size for multi-scale convolutions
            dilation (int): Dilation factor for dilated convolutions
            alpha (float): Ratio for splitting channels in CRM
            mlp_ratio (int): Ratio for expanding channels in CRM
        """
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
        """Forward pass through the scale-wise convolutional residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after SCRB processing
        """
        x = self.scm(x)
        x = self.crm(x)
        return x