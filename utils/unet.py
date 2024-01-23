# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:26:31 2022

@author: A0067501
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_net import padsize, get_activation, UpBlock, ConvBlock, UpBlock2D, ConvBlock2D, Upsample



class UNet(torch.nn.Module):
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
            padding=1,
            stride=1,
            out_channels=1,
            residual=True,
            dropout=0.1,
            batchnorm=False,
            activation="relu",
            up_mode="conv",
            return_bottom=False,
            deep_supervision=True,
            **kwargs
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size=[kernel_size for i in range(L+1)]
        if isinstance(padding, int):
            padding=[padding for i in range(L+1)]
        if isinstance(stride, int):
            stride=[stride for i in range(L+1)]
        self.residual = residual
        self.deep_supervision=deep_supervision
        self.L = L
        self.return_bottom = return_bottom
        self.act = activation
        f_in, f = in_channels, start_filters
        # left side of U-net (without pooling ops)
        self.left = nn.ModuleList()
        for i in range(L + 1):
            self.left.append(ConvBlock(f_in, f * (2 ** i), kernel_size[i], stride[i], padding[i], batchnorm, dropout, self.act))
            f_in = f * (2 ** i)
        # right side of U-net
        self.right = nn.ModuleList()
        for i in reversed(range(L)):
            self.right.append(UpBlock(f_in, f * (2 ** i), 3, 1, 1, batchnorm, dropout, self.act, up_mode))
            f_in = f * (2 ** i)
        #deep supervision    
        self.output = nn.ModuleList()
        for i in reversed(range(L)):
            self.output.append(nn.Conv3d(f * (2 ** i), out_channels, 1))
            
        self.upscale_ops = nn.ModuleList()
        for i in reversed(range(L)):
            self.upscale_ops.append(Upsample(scale_factor=(1,2**i,2**i),mode='trilinear'))
    
        self.last = nn.Conv3d(f_in, out_channels, 1)
        self.softmax = get_activation(kwargs.get("last_activation", "softmax"))

    def forward(self, inputs):
        left_block = []
        x = inputs
        for i, down in enumerate(self.left):
            x = down(x)
            #print(x.shape)
            if i != len(self.left) - 1:
                left_block.append(x)
                # x = F.max_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
                # print(x.shape)
        bottom = x

        seg_outputs = []
        for i in range(len(self.right)):
            x = self.right[i](x,  left_block[-i - 1])
            #print(x.shape)
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


