# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:05:37 2022

@author: A0067501
"""

import torch
#import kornia
from torch import nn


class Likelyloss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Likelyloss, self).__init__()
    def forward(self, predictions, inputs, heart):
        (B,K,X,Y)=predictions.shape
        mu=torch.zeros((B,K))
        var=torch.zeros((B,K))
        alpha=torch.zeros((B,K))
        N=torch.sum(heart[:,0,...],axis=[1,2])
        eps=torch.finfo(torch.float32).eps
        for b in range(B):
            for k in range(1,K):
                alpha[b,k]=(torch.sum(predictions[b,k,...]))/N[b]
                mu[b,k]=torch.sum(predictions[b,k,...]*inputs[b,0,...])/(torch.sum(predictions[b,k,...])+eps)
                var[b,k]=(torch.sum(predictions[b,k,...]*(inputs[b,0,...]-mu[b,k])**2)/(torch.sum(predictions[b,k,...])+eps))+eps
                
        temp=torch.zeros_like(predictions)
        for b in range(B):
            for k in range(1,K):
                temp[b,k,...]=alpha[b,k]*(1/(torch.sqrt(2*torch.pi*var[b,k])))*torch.exp(-((inputs[b,0,...]-mu[b,k])**2/(2*var[b,k])))
        likelylosses=torch.zeros(B)
        for b in range(B):
            likelylosses[b]=-torch.mean(torch.log(torch.sum(temp[b,...],axis=0)+eps))
        likelyloss=torch.mean(likelylosses)
        return likelyloss

class Likelyloss_heart(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Likelyloss_heart, self).__init__()
    def forward(self, predictions, inputs, mask_heart):
        likelyloss=0
        B=inputs.shape[0]
        
        for b in range(B):
            l_blood=predictions[b,1,...][mask_heart[b,0,...]==1]
            l_muscle=predictions[b,2,...][mask_heart[b,0,...]==1]
            l_scar=predictions[b,3,...][mask_heart[b,0,...]==1]
            l_mvo=predictions[b,4,...][mask_heart[b,0,...]==1]
            l_in=inputs[b,...][mask_heart[b,...]==1]
            
            l_all=torch.stack((l_blood,l_scar,l_muscle,l_mvo))
            l_classes=torch.stack((l_blood+l_scar,l_muscle+l_mvo))
            device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            l_components= torch.zeros(2,4,len(l_blood)).to(device)
            
            l_component1=torch.stack((l_blood,l_scar))
            l_component1=nn.Softmax(dim=0)(l_component1)
            l_components[0,0,...]=l_component1[0,...]
            l_components[0,2,...]=l_component1[1,...]
            
            l_component2=torch.stack((l_muscle,l_mvo))
            l_component2=nn.Softmax(dim=0)(l_component2)
            l_components[1,1,...]=l_component2[0,...]
            l_components[1,3,...]=l_component2[1,...]
            
            
            mu=torch.zeros(4)
            var=torch.zeros(4)
            
            eps=torch.finfo(torch.float32).eps
            for k in range(4):
                  mu[k]=torch.sum(l_all[k]*l_in)/(torch.sum(l_all[k])+eps)
                  var[k]=(torch.sum(l_all[k]*(l_in-mu[k])**2)/(torch.sum(l_all[k])+eps))+eps
                
            temp=torch.zeros_like(l_components)
            for l in range(2):
                for c in range(4):
                    temp[l,c]=l_components[l,c]*(1/(torch.sqrt(2*torch.pi*var[c])))*torch.exp(-((l_in-mu[c])**2/(2*var[c])))
            
            percentage_scar
            temp=l_classes*torch.sum(temp, axis=1)
            
            
        
            likelyloss-=torch.mean(torch.log(torch.sum(temp,axis=0)+eps))
            
            
        return likelyloss/B

class DiceLoss(torch.nn.Module):
    """
    Dice loss function for training a multi class segmentation network.
    The prediction of the network should be of type softmax(...) and the target should be one-hot encoded.
    """

    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.num_classes = kwargs.get("num_classes", 1)
        self.weights = kwargs.get("weights", self.num_classes * [1])
        self.smooth = kwargs.get("smooth", 1.)
        self.p = kwargs.get("p", 2)

    def _single_class(self, prediction, target):
        bs = prediction.size(0)
        p = prediction.reshape(bs, -1)
        t = target.reshape(bs, -1)

        intersection = (p * t).sum(1)
        total = (p.pow(self.p) + t.pow(self.p)).sum(1)

        loss = 1 - (2 * intersection + self.smooth) / (total + self.smooth)
        return loss.mean()

    def forward(self, prediction, target):
        assert prediction.shape == target.shape
        loss = 0
        for c in range(self.num_classes):
            loss += self._single_class(prediction[:, c, ...], target[:, c, ...]) * self.weights[c]
        return loss / sum(self.weights)


def get_loss(crit="likely", **kwargs):
    if crit == "likely":
        return Likelyloss()
    elif crit == "likely_heart":
        return Likelyloss_heart()
    elif crit == "dice":
        return DiceLoss()
    else:
        return print("wrong crit!")

