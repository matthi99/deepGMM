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
        K=4
        M=inputs.shape[1]
        eps=1e-10
        likelylosses=torch.zeros(B)
        mu_mean=torch.zeros((K,M))
        sigma_mean=torch.zeros((K,M))
        for b in range(B):
            l_blood=predictions[b,1,...][heart[b,0,...]==1]
            l_muscle=predictions[b,2,...][heart[b,0,...]==1]
            l_edema=predictions[b,3,...][heart[b,0,...]==1]
            l_scar=predictions[b,4,...][heart[b,0,...]==1]
            
            in_LGE =inputs[b,0,...][heart[b,0,...]==1]
            in_T2 = inputs[b,1,...][heart[b,0,...]==1]
            in_C0 = inputs[b,2,...][heart[b,0,...]==1]
            
            inp= torch.stack((in_LGE, in_T2, in_C0))
            pred_lge=torch.stack((l_blood, l_muscle + l_edema, l_muscle + l_edema, l_scar))
            pred_t2=torch.stack((l_blood, l_muscle, l_edema+l_scar, l_edema+l_scar))
            pred_c0=torch.stack((l_blood, l_muscle + l_edema + l_scar, 
                                  l_muscle + l_edema + l_scar, l_muscle + l_edema + l_scar))
            
            
            
            pred=torch.stack((l_blood, l_muscle, l_edema, l_scar))
            #pred_mu=torch.stack((pred_lge, pred_t2, pred_c0),1)
            pred_mu=torch.stack((pred, pred, pred),1)
            
            # pred_gt= torch.stack((gt[b,1,...][heart[b,0,...]==1], gt[b,2,...][heart[b,0,...]==1], 
            #                       gt[b,3,...][heart[b,0,...]==1], gt[b,4,...][heart[b,0,...]==1]))
            
            
            # mu_gt=torch.zeros((K,M))
            # var_gt=torch.zeros((K,M))
            
            # for k in range(K):
            #     for m in range(M):
            #         mu_gt[k,m]=torch.sum(pred_gt[k]*inp[m])/(torch.sum(pred_gt[k])+eps)
            #         var_gt[k,m]=(torch.sum(pred_gt[k]*(inp[m,...]-mu_gt[k,m])**2)/(torch.sum(pred_gt[k])+eps))+eps
            
            mu=torch.zeros((K,M))
            var=torch.zeros((K,M))
            
            for k in range(K):
                for m in range(M):
                    mu[k,m]=torch.sum(pred_mu[k,m,...]*inp[m,...])/(torch.sum(pred_mu[k,m,...])+eps)
                    var[k,m]=(torch.sum(pred_mu[k,m,...]*(inp[m,...]-mu[k,m])**2)/(torch.sum(pred_mu[k,m,...])+eps))+eps
             
            mu_mean+=mu    
            sigma_mean+=torch.sqrt(var)
            #print(mu)        
            temp=torch.zeros((K,M,len(in_LGE)))
            
            for k in range(K):
                for m in range(M):
                    temp[k,m,...]=pred[k]*(1/(torch.sqrt(2*torch.pi*var[k,m])))*torch.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
            
            
            likelylosses[b]=-torch.mean(torch.log(torch.sum(torch.prod(temp,1),axis=0)+eps))
            #print(mu_gt)
            
            
        likelyloss=torch.mean(likelylosses)
        
        
        
        # N=B*X*Y
        # probs = (torch.sum(predictions,axis=[0,2,3])/N)
        # a=probs[0]
        # probs=probs[1:]*(1/(1-a))
        # #print(probs.sum())
        # probs_gt= (1/4)*torch.ones_like(probs)
        # #print(probs_gt)
        # likelyloss = likelyloss + torch.sum((probs-probs_gt)**2)
        
        return likelyloss


class Mu_sigma(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Mu_sigma, self).__init__()
    def forward(self, predictions, inputs, heart):
        (B,K,X,Y)=predictions.shape
        K=4
        M=inputs.shape[1]
        eps=1e-10
        mu_mean=torch.zeros((K,M))
        sigma_mean=torch.zeros((K,M))
        mu_sigma=0
        for b in range(B):
            l_blood=predictions[b,1,...][heart[b,0,...]==1]
            l_muscle=predictions[b,2,...][heart[b,0,...]==1]
            l_edema=predictions[b,3,...][heart[b,0,...]==1]
            l_scar=predictions[b,4,...][heart[b,0,...]==1]
            
            in_LGE =inputs[b,0,...][heart[b,0,...]==1]
            in_T2 = inputs[b,1,...][heart[b,0,...]==1]
            in_C0 = inputs[b,2,...][heart[b,0,...]==1]
            
            inp= torch.stack((in_LGE, in_T2, in_C0))
            pred=torch.stack((l_blood, l_muscle, l_edema, l_scar))
            pred_mu=torch.stack((pred, pred, pred),1)
            mu=torch.zeros((K,M))
            var=torch.zeros((K,M))
            for k in range(K):
                for m in range(M):
                    mu[k,m]=torch.sum(pred_mu[k,m,...]*inp[m,...])/(torch.sum(pred_mu[k,m,...])+eps)
                    var[k,m]=(torch.sum(pred_mu[k,m,...]*(inp[m,...]-mu[k,m])**2)/(torch.sum(pred_mu[k,m,...])+eps))+eps
            mu_mean+=mu    
            sigma_mean+=torch.sqrt(var)
        
        mu_mean =mu_mean/B
        sigma_mean = sigma_mean/B
        mu_gt= torch.tensor([[ 1.13263178, -0.13630785,  1.6058956 ],
                             [-0.41842756,  0.51262672, -0.00925156],
                             [ 0.29902016,  1.25442789,  0.52296464],
                             [ 1.15254939,  1.04698159,  0.3860968 ]])
        
        sigma_gt= torch.tensor([[0.67609396, 0.61812245, 0.59229357],
                                [0.57415889, 0.41136607, 0.46106531],
                                [0.71600954, 0.45154963, 0.58179293],
                                [0.80249576, 0.5147093,  0.50932555]])
        
        mu_sigma+=torch.mean((mu_gt-mu_mean)**2)
        
        #mu_sigma+=torch.mean((sigma_gt-sigma_mean)**2)
        return mu_sigma


class Probs(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Probs, self).__init__()
    def forward(self, predictions, heart):    
        (B,K,X,Y)=predictions.shape
        l_blood=predictions[:,1,...][heart[:,0,...]==1]
        l_muscle=predictions[:,2,...][heart[:,0,...]==1]
        l_edema=predictions[:,3,...][heart[:,0,...]==1]
        l_scar=predictions[:,4,...][heart[:,0,...]==1]
        pred=torch.stack((l_blood, l_muscle, l_edema, l_scar))
        
        N=len(l_blood)
        probs = (torch.sum(pred,axis=1)/N)
        
        # a=probs[0]
        # probs=probs[1:]*(1/(1-a))
        #print(probs.sum())
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #probs_gt=torch.tensor([0.3839563844238861, 0.4023091477825316,  0.098236304180275, 0.11549816361330738]).to(device)
        probs_gt = (torch.ones(4)/4).to(device)
        #print(probs_gt)
        probs_loss = torch.mean((probs-probs_gt)**2)
        return probs_loss
    


# class Likelyloss(torch.nn.Module):
#     def __init__(self, **kwargs):
#         super(Likelyloss, self).__init__()
#     def forward(self, predictions, inputs, heart):
#         (B,K,X,Y)=predictions.shape
#         K=5
#         M=inputs.shape[1]
#         eps=1e-10
#         likelylosses=torch.zeros(B)
#         for b in range(B):
#             l_blood=predictions[b,1,...][heart[b,0,...]==1]
#             l_muscle=predictions[b,2,...][heart[b,0,...]==1]
#             l_edema=predictions[b,3,...][heart[b,0,...]==1]
#             l_scar=predictions[b,4,...][heart[b,0,...]==1]
#             l_mvo=predictions[b,5,...][heart[b,0,...]==1]
#             in_LGE =inputs[b,0,...][heart[b,0,...]==1]
#             in_T2 = inputs[b,1,...][heart[b,0,...]==1]
#             in_C0 = inputs[b,2,...][heart[b,0,...]==1]
            
#             inp= torch.stack((in_LGE, in_T2, in_C0))
#             pred=torch.stack((l_blood, l_muscle, l_edema, l_scar, l_mvo))
            
            
#             mu=torch.zeros((K,M))
#             var=torch.zeros((K,M))
#             for k in range(K):
#                 for m in range(M):
#                     mu[k,m]=torch.sum(pred[k,...]*inp[m,...])/(torch.sum(pred[k,...])+eps)
#                     var[k,m]=(torch.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(torch.sum(pred[k,...])+eps))+eps
                    
#             temp=torch.zeros((K,M,len(in_LGE)))
            
#             for k in range(1,K):
#                 for m in range(M):
#                     temp[k,m,...]=pred[k]*(1/(torch.sqrt(2*torch.pi*var[k,m])))*torch.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
            
            
#             likelylosses[b]=-torch.mean(torch.log(torch.sum(torch.prod(temp,1),axis=0)+eps))
        
#         likelyloss=torch.mean(likelylosses)
#         #print(likelyloss)
#         N=B*X*Y
#         probs = (torch.sum(predictions,axis=[0,2,3])/N)
#         a=probs[0]
#         probs=probs[1:]*(1/(1-a))
#         #print(probs)
#         probs_gt= torch.zeros_like(probs)
#         probs_gt[0]=0.5
#         probs_gt[1]=0.3
#         probs_gt[2]=0.1
#         probs_gt[3]=0.08
#         probs_gt[4]=0.02
#         #print(probs_gt)
#         likelyloss = likelyloss + torch.mean((probs-probs_gt)**2)
        
#         return likelyloss


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
    elif crit == "mu_sigma":
        return Mu_sigma()
    elif crit == "probs":
        return Probs()
    else:
        return print("wrong crit!")

