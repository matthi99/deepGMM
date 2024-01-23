# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:26:55 2022

@author: A0067501
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




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




class ConvBlock(nn.Module):
    def __init__(self, f_in, f_out, kernel, stride, padding, batchnorm, dropout, activation):
        super().__init__()
        block = [nn.Conv3d(f_in, f_out, kernel_size=kernel, stride=stride, padding=padding)]
        if dropout:
            block.append(nn.Dropout3d(dropout))
        if batchnorm:
            block.append(nn.InstanceNorm3d(f_out, affine=True))
        block.append(get_activation(activation))

        block.append(nn.Conv3d(f_out, f_out, kernel_size=kernel, stride=(1,1,1), padding=padding, bias=(not batchnorm)))
        if dropout:
            block.append(nn.Dropout3d(dropout))
        if batchnorm:
            block.append(nn.InstanceNorm3d(f_out, affine=True))
        block.append(get_activation(activation))

        self.f = nn.Sequential(*block)

    def forward(self, x):
        return self.f(x)


class UpBlock(nn.Module):
    """
    Upsampling followed by a convolution block
    """

    def __init__(self, f_in, f_out, kernel, stride, padding, batchnorm, dropout, activation_config, up_mode):
        super().__init__()
        if "conv" in up_mode:
            self.up = nn.ConvTranspose3d(f_in, f_out, kernel_size=(1,2,2),  stride=(1,2,2))
        elif up_mode == "bilinear":
            self.up = nn.Sequential(Interpolate(scale_factor=(1,2,2), mode="trilinear"), nn.Conv3d(f_in, f_out, 1))
        else:
            self.up = nn.Sequential(Interpolate(scale_factor=(1,2,2), mode="nearest"), nn.Conv3d(f_in, f_out, 1))

        self.convs = ConvBlock(f_in, f_out, kernel, stride, padding, batchnorm, dropout, activation_config)

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