# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:18:54 2024

@author: A0067501
"""


import numpy as np
import torch
from utils.architectures import get_network
import yaml

def normalize(img):
    return (img - np.mean(img)) / np.std(img)


def prepare_data(path_to_data):
    data= np.load(path_to_data, allow_pickle=True).item()
    gt= np.argmax(data['masks'][data["center"][0]-80:data["center"][0]+80,data["center"][1]-80:data["center"][1]+80,:],axis=2)
    mask_heart= (1-data['masks'][:,:,0])[data["center"][0]-80:data["center"][0]+80,data["center"][1]-80:data["center"][1]+80]
    LGE= data["LGE"][data["center"][0]-80:data["center"][0]+80,data["center"][1]-80:data["center"][1]+80]
    T2 = data["T2"][data["center"][0]-80:data["center"][0]+80,data["center"][1]-80:data["center"][1]+80]
    C0 = data["C0"][data["center"][0]-80:data["center"][0]+80,data["center"][1]-80:data["center"][1]+80]
    LGE = normalize(LGE)
    T2 = normalize(T2)
    C0 = normalize(C0)
    X=np.stack((LGE, T2, C0), axis=0)
    return X, gt, mask_heart


def variant_log_likelyhood(predictions, inputs, heart):
    (K,X,Y)=predictions.shape
    M=inputs.shape[0]
    eps=1e-14
    
    pred = []
    for cl in range(K):
        pred.append(predictions[cl,...][heart==1])
    pred = np.stack(pred,0)
        
    inp=[]
    for ch in range(M):
        inp.append(inputs[ch,...][heart==1])
    inp = np.stack(inp,0)
        
    mu=np.zeros((K,M))
    var=np.zeros((K,M))
    for k in range(K):
        for m in range(M):
            mu[k,m]=np.sum(pred[k,...]*inp[m,...])/(np.sum(pred[k,...])+eps)
            var[k,m]=(np.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(np.sum(pred[k,...])+eps))+eps
        temp=np.zeros((K,M,inp.shape[1]))
        for k in range(K):
            for m in range(M):
                temp[k,m,...]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
        temp =  np.prod(temp,1)
        temp= pred * temp
        likelyloss = -np.mean(np.log(np.sum(temp,axis=0)+eps))
    return likelyloss



def normal_log_likelyhood(predictions, inputs, heart):
    (K,X,Y)=predictions.shape
    M=inputs.shape[0]
    eps=1e-14
    
    pred = []
    for cl in range(K):
        pred.append(predictions[cl,...][heart==1])
    pred = np.stack(pred,0)
        
    inp=[]
    for ch in range(M):
        inp.append(inputs[ch,...][heart==1])
    inp = np.stack(inp,0)
        
    mu=np.zeros((K,M))
    var=np.zeros((K,M))
    alpha = np.mean(pred,1)
    for k in range(K):
        for m in range(M):
            mu[k,m]=np.sum(pred[k,...]*inp[m,...])/(np.sum(pred[k,...])+eps)
            var[k,m]=(np.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(np.sum(pred[k,...])+eps))+eps
        temp=np.zeros((K,M,inp.shape[1]))
        for k in range(K):
            for m in range(M):
                temp[k,m,...]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
        temp =  np.prod(temp,1)
        temp = alpha[:,np.newaxis] *temp
        likelyloss = -np.mean(np.log(np.sum(temp,axis=0)+eps))
    return likelyloss

def load_2dnet(path, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + f"/weights.pth",  map_location=torch.device(device))
    net2d = get_network(architecture='unet2d', device=device, **params)
    net2d.load_state_dict(weights)
    net2d.eval()
    return net2d


def load_2dnet_single(path, device):
    path_params = path[0:-13]+"config.json"
    params= yaml.load(open(path_params, 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path,  map_location=torch.device(device))
    net2d = get_network(architecture='unet2d', device=device, **params)
    net2d.load_state_dict(weights)
    net2d.eval()
    return net2d

def dicecoeff(prediction, target):
    intersection = np.sum(prediction * target)
    total =  np.sum(prediction + target)
    if total == 0:
        return 1
    else:
        dice =(2 * intersection) / total 
        return dice


def order_dice(pred, gt):
    ordered= np.zeros_like(pred)
    #bloodpool
    pred_gt = (gt==1)*1
    dice = np.zeros(5)
    for j in np.unique(pred)[1:].astype(int):
        dice[j]= dicecoeff(pred_gt, (pred==j)*1)
    value = np.argmax(dice)
    ordered[pred==value]=1
    pred[pred==value]=0
    if len(np.unique(gt)[1:])==2:
        ordered[pred!=0]=2
    elif len(np.unique(gt)[1:])==3 and 3 in np.unique(gt)[1:]:
        #best fit for edema
        pred_gt = (gt==3)*1
        dice = np.zeros(5)
        for j in np.unique(pred)[1:].astype(int):
            dice[j]= dicecoeff(pred_gt, (pred==j)*1)
        value = np.argmax(dice)
        ordered[pred==value]=3
        pred[pred==value]=0
        ordered[pred!=0]=2
    elif len(np.unique(gt)[1:])==3 and 4 in np.unique(gt)[1:]:
        #best fit for scar
        pred_gt = (gt==4)*1
        dice = np.zeros(5)
        for j in np.unique(pred)[1:].astype(int):
            dice[j]= dicecoeff(pred_gt, (pred==j)*1)
        value = np.argmax(dice)
        ordered[pred==value]=4
        pred[pred==value]=0
        ordered[pred!=0]=2
    else:
        #best fit for muscle
        pred_gt = (gt==2)*1
        dice = np.zeros(5)
        for j in np.unique(pred)[1:].astype(int):
            dice[j]= dicecoeff(pred_gt, (pred==j)*1)
        value = np.argmax(dice)
        ordered[pred==value]=2
        pred[pred==value]=0
        #best fit for scar
        pred_gt = (gt==4)*1
        dice = np.zeros(5)
        for j in np.unique(pred)[1:].astype(int):
            dice[j]= dicecoeff(pred_gt, (pred==j)*1)
        value = np.argmax(dice)
        ordered[pred==value]=4
        pred[pred==value]=0
        #rest edema
        ordered[(pred!=0)]=3
    return ordered
        