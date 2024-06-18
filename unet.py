# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:15:30 2024

@author: A0067501
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




def padsize(kernel_size=3, mode="same", dilation=1):
    """
    translates mode to size of padding
    """
    if not isinstance(kernel_size, list):
        k = [kernel_size, kernel_size]
    else:
        k = kernel_size
    if not isinstance(dilation, list):
        d = [dilation, dilation]
    else:
        d = dilation
    assert len(d) == len(k)

    p = [0 for _ in range(len(k))]
    if mode == "same":
        for i in range(len(p)):
            p[i] = (d[i] * (k[i] - 1)) // 2

    if np.unique(p).shape[0] == 1:
        p = p[0]
    return p


def get_activation(activation="prelu", **kwargs):
    if activation == "prelu":
        num_parameters = kwargs.get("activation_num_parameters", 1)
        init = kwargs.get("activation_init", 0.25)
        return torch.nn.PReLU(num_parameters, init=init)
    elif activation == "identity":
        return torch.nn.Identity()
    elif activation == "softmax":
        return torch.nn.Softmax(dim=1)
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "leakyrelu":
        return torch.nn.LeakyReLU(0.01, inplace=True)
    else:
        return torch.nn.ReLU(inplace=kwargs.get("activation_inplace", False))

class Interpolate(nn.Module):
    """
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest"):
        super().__init__()
        self.interp = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, size=self.size, mode=self.mode)
        return x

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class UpBlock2D(nn.Module):
    """
    Upsampling followed by a convolution block
    """

    def __init__(self, f_in, f_out, kernel, pad, batchnorm, dropout, activation_config, up_mode):
        super().__init__()
        if "conv" in up_mode:
            self.up = nn.ConvTranspose2d(f_in, f_out, 2, stride=2)
        if up_mode == "bilinear":
            self.up = nn.Sequential(Interpolate(scale_factor=2, mode="bilinear"), nn.Conv2d(f_in, f_out, 1))
        else:
            self.up = nn.Sequential(Interpolate(scale_factor=2, mode="nearest"), nn.Conv2d(f_in, f_out, 1))

        self.convs = ConvBlock2D(f_in, f_out, kernel, pad, batchnorm, dropout, activation_config)

    def forward(self, x1, x0):
        # x0...output from connected layer before
        up = self.up(x1)
        out = torch.cat([up, x0], dim=1)
        return self.convs(out)


class ConvBlock2D(nn.Module):
    def __init__(self, f_in, f_out, kernel, pad, batchnorm, dropout, activation):
        super().__init__()
        block = [nn.Conv2d(f_in, f_out, kernel, padding=pad)]
        if dropout:
            block.append(nn.Dropout2d(dropout))
        block.append(get_activation(activation))

        block.append(nn.Conv2d(f_out, f_out, kernel, padding=pad, bias=(not batchnorm)))
        if dropout:
            block.append(nn.Dropout2d(dropout))
        if batchnorm:
            block.append(nn.InstanceNorm2d(f_out, affine=True))
        block.append(get_activation(activation))

        self.f = nn.Sequential(*block)

    def forward(self, x):
        return self.f(x)



class UNet2D(torch.nn.Module):
    """
    U-net model introduced in :cite:`unet`.
    On the left side of the network the number of filter channels doubles after each downsampling step
    and on the right side it is the other way.

    Parameters
    ----------
    L: int
        is number of downsampling steps
    start_filters: int
        number of filter channels used for the first convolution
    in_channels: int
        number of channels of the inputs
    kernel_size: int
        convolution kernel size
    out_channels: int
        number of channels of the output
    residual: bool
        if true use residual U-Net
    dropout: float
        dropout for all convolution layers
    batchnorm: bool
        if true calculate batchnorm after each convolution
    activation: nn.Module
        activation function
    up_mode: str
        can be 'nearest', 'bilinear' or 'conv' (for transposed convolution based upsampling)
    return_bottom: bool
        if true also returns the bottom of the network i.e. features before the first upsampling step
    """

    def __init__(
            self,
            L=4,
            start_filters=32,
            in_channels=1,
            kernel_size=3,
            out_channels=1,
            residual=True,
            dropout=0,
            batchnorm=False,
            activation="relu",
            up_mode="nearest",
            return_bottom=False,
            deep_supervision=True,
            **kwargs
    ):
        super().__init__()
        self.residual = residual
        self.L = L
        self.deep_supervision=deep_supervision
        self.return_bottom = return_bottom
        pad = padsize(kernel_size)
        self.act = activation
        f_in, f = in_channels, start_filters
        # left side of U-net (without pooling ops)
        self.left = nn.ModuleList()
        for i in range(L + 1):
            self.left.append(ConvBlock2D(f_in, f * (2 ** i), kernel_size, pad, batchnorm, dropout, self.act))
            f_in = f * (2 ** i)
        # right side of U-net
        self.right = nn.ModuleList()
        for i in reversed(range(L)):
            self.right.append(UpBlock2D(f_in, f * (2 ** i), kernel_size, pad, batchnorm, dropout, self.act, up_mode))
            f_in = f * (2 ** i)
        #deep supervisiom
        self.output = nn.ModuleList()
        for i in reversed(range(L)):
            self.output.append(nn.Conv2d(f * (2 ** i), out_channels, 1))
        
        self.upscale_ops = nn.ModuleList()
        for i in reversed(range(L)):
            self.upscale_ops.append(Upsample(scale_factor=(2**i,2**i),mode='bilinear'))
        self.last = nn.Conv2d(f_in, out_channels, 1)
        self.softmax = get_activation(kwargs.get("last_activation", "sigmoid"))

    def forward(self, inputs):
        left_block = []
        x = inputs
        for i, down in enumerate(self.left):
            x = down(x)
            if i != len(self.left) - 1:
                left_block.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        bottom = x

        seg_outputs = []
        for i in range(len(self.right)):
            x = self.right[i](x,  left_block[-i - 1])
            seg_outputs.append(self.softmax(self.output[i](x)))

        out = self.softmax(self.last(x))
        if self.residual:
            out = self.last(x)+ inputs
            out = self.softmax(out) 
        if self.return_bottom:
            return out, bottom
        if self.deep_supervision:
            return tuple([seg_outputs[-1]]+[self.upscale_ops[i](seg_outputs[i]) for i in reversed(range(1,len(seg_outputs)))])
        else:
            return out
