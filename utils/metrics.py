# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:05:38 2022

@author: A0067501
"""

import torch


class HausdorffDistance(torch.nn.Module):
    """
    Implements the pixel Hausdorff-distance using the following logic:
    1) If exactly one input is empty then the distance is defined to be the diameter of the other input.
    2) If both inputs are empty then the distance is defined to be 0.
    3) If both inputs are non-empty then the usual Hausdorff distance is calculated.

    Both inputs are first thresholded to decide which points belong to the mask.
    """

    def __init__(self, **kwargs):
        super(HausdorffDistance, self).__init__()
        self.tau = kwargs.get("tau", 0.5)
        self.reduction = kwargs.get("collate", "mean")

    @staticmethod
    def _single_element(set1, set2):
        if set1.numel() == 0 and set2.numel() != 0:
            dist = torch.cdist(set2, set2)
            ret = dist.max()
        elif set2.numel() == 0 and set1.numel() != 0:
            dist = torch.cdist(set1, set1)
            ret = dist.max()
        elif set1.numel() == 0 and set2.numel() == 0:
            ret = 0
        else:
            dist = torch.cdist(set1, set2).squeeze()
            ret = max([dist.min(1)[0].max(), dist.min(0)[0].max()])
        return ret

    def forward(self, prediction, target):
        assert prediction.shape == target.shape
        b = prediction.size(0)
        pred = prediction >= self.tau
        label = target >= self.tau
        hd = []
        for i in range(b):
            pred_set = torch.nonzero(pred[i, ...]).float()
            label_set = torch.nonzero(label[i, ...]).float()
            hd.append(self._single_element(pred_set, label_set))
        if self.reduction == "mean":
            return torch.tensor(hd).mean()
        return torch.tensor(hd)


class DiceMetric(torch.nn.Module):
    """
    Dice metric function for training a multi class segmentation network.
    The prediction of the network should be of type softmax(...) and the target should be one-hot encoded.
    """

    def __init__(self, **kwargs):
        super(DiceMetric, self).__init__()
        self.num_classes = kwargs.get("num_classes", 1)
        self.weights = kwargs.get("weights", self.num_classes * [1])
        self.smooth = kwargs.get("smooth", 0)
        

    def forward(self, prediction, target):
        bs = prediction.size(0)
        cl = prediction.size(1)
        p = prediction.view(bs,cl, -1)
        t = target.view(bs,cl, -1)

        intersection = (p * t).sum(2)
        total = (p+t).sum(2)
        
        dice_coeff=torch.zeros(bs, cl)
        for i in range(bs):
            for j in range(cl):
                if total[i,j] == 0:
                    dice_coeff[i,j]=1
                else:
                   dice_coeff[i,j]=((2 * intersection[i,j]) / (total[i,j] )) 
        dice_coeff= torch.mean(dice_coeff,0)
        return dice_coeff[1:]
       


def get_metric(metric="dice", **kwargs):
    if metric == "dice":
        return DiceMetric( **kwargs)
    elif metric == "hausdorff":
        return HausdorffDistance(**kwargs)
    else:
        return DiceMetric(**{**{'smooth': 0., 'p': 1}, **kwargs})
