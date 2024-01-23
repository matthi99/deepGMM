# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:40:54 2022

@author: A0067501
"""

import os
import torch
import yaml

from utils.unet import UNet, UNet2D

def get_network(architecture="unet", device="cuda:0", **kwargs):
    architecture = architecture.lower()
    if architecture == "unet":
        net = UNet(**kwargs)
    elif architecture == "unet2d":
        net = UNet2D(**kwargs)

    else:
        net = UNet(**kwargs)
    return net.to(device=device)


class InputPreprocessing:
    def __init__(self, **kwargs):
        self.cls = kwargs.get("cls", "heart")
        self.device = kwargs.get("device", "cuda:0")
        self.networks = self.get_networks(kwargs.get("network_path", "weights/"))
        self.tau = kwargs.get("tau", 0.5)

    def get_networks(self, path):
        nets = {}
        temp = [os.path.splitext(file)[0] for file in os.listdir(path) if file.endswith('.pth')]
        for cls in temp:
            params = yaml.load(open(f"{path}/params_{cls}.json", 'r'), Loader=yaml.SafeLoader)
            weights = torch.load(f"{path}/{cls}.pth")
            net = get_network(device=self.device, **params)

            net.load_state_dict(weights)
            net.eval()
            nets[cls] = net
        return nets

    @torch.no_grad()
    def __call__(self, inputs, *args, **kwargs):
        if self.cls == "heart":
            return inputs
        elif self.cls == "bloodpool":
            heart = self.networks['heart'](inputs)
            return torch.cat([heart * inputs, heart], dim=1)
        elif self.cls == "scartotal":
            heart = self.networks['heart'](inputs)
            temp = torch.cat([heart * inputs, heart], dim=1)
            bp = self.networks['bloodpool'](temp)
            ring = heart * (1. - bp)
            return torch.cat([inputs, ring * inputs, ring, temp], dim=1)
        elif self.cls == "muscle":
            heart = self.networks['heart'](inputs)
            temp = torch.cat([heart * inputs, heart], dim=1)
            bp = self.networks['bloodpool'](temp)
            ring = heart * (1. - bp)
            return torch.cat([inputs, ring * inputs, ring, temp], dim=1)
        elif self.cls == "mvo":
            heart = self.networks['heart'](inputs)
            temp = torch.cat([heart * inputs, heart], dim=1)
            bp = self.networks['blood'](temp)
            ring = heart * (1. - bp)
            return torch.cat([inputs, ring * inputs, ring, temp], dim=1)
        else:
            return inputs
