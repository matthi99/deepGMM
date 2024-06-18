# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:18:54 2024

@author: A0067501
"""


import numpy as np
#import torch
from sklearn.mixture import GaussianMixture as GMM
#from utils.architectures import get_network
#import yaml
import os
import torch


class SVGMM():
    def __init__(self, 
                 n_components = 4, 
                 tol=0.001, 
                 means_init = None, 
                 init_params = "random", 
                 weights_init = None, 
                 covariances_init=None, 
                 max_iter = 100,
                 covariance_type = "diag",
                 random_state = None):
        self.tol = tol
        self.means = means_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.n_components = n_components
        self.weights_ = weights_init
        self.means_ = means_init
        self.covariances_ = covariances_init
        self.reg_covar = 1e-6
        self.max_iter = max_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        
    
    def initialize_parameters(self, X):
        gmm = GMM(n_components=self.n_components, covariance_type=self.covariance_type,
                  means_init = self.means_init, random_state=self.random_state)
        gmm._initialize_parameters(X,self.random_state)
        #gmm.fit(X)
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_ = gmm.predict_proba(X)
        
    def M_step(self, X):
        mu=np.zeros_like(self.means_)
        var=np.zeros_like(self.covariances_)
        n_samples, n_mods = X.shape
        eps= 1e-10
        for k in range(self.n_components):
            for m in range(n_mods):
                mu[k,m]=np.sum(self.weights_[:,k]*X[:,m])/(np.sum(self.weights_[:,k])+eps)
                var[k,m]=(np.sum(self.weights_[:,k]*(X[:,m]-mu[k,m])**2)/(np.sum(self.weights_[:,k])+eps))+eps
        
        self.means_ = mu
        self.covariances_ = var
        
    def E_step(self, X):
        n_samples, n_mods = X.shape
        
        temp=np.zeros((n_samples, self.n_components, n_mods))
        for k in range(self.n_components):
            for m in range(n_mods):
                temp[:,k,m]=(1/(np.sqrt(2*np.pi*self.covariances_[k,m])))*np.exp(-((X[:,m]-self.means_[k,m])**2/(2*self.covariances_[k,m])))
        temp =  np.prod(temp,2)
        temp= self.weights_ * temp
        self.weights_ = temp/temp.sum(axis=1)[:, np.newaxis]
        
    def compute_neg_log_likely(self,X):
        n_samples, n_mods = X.shape
        
        temp=np.zeros((n_samples, self.n_components, n_mods))
        for k in range(self.n_components):
            for m in range(n_mods):
                temp[:,k,m]=(1/(np.sqrt(2*np.pi*self.covariances_[k,m])))*np.exp(-((X[:,m]-self.means_[k,m])**2/(2*self.covariances_[k,m])))
        temp =  np.prod(temp,2)
        temp= self.weights_ * temp
        neg_log_likely = -np.mean(np.log(np.sum(temp,axis=1)))
        return neg_log_likely
        
    
    def fit(self, X):
        self.initialize_parameters(X)
        for n_iter in range(1, self.max_iter + 1):
            prev_neg_log_likely = self.compute_neg_log_likely(X)
            self.E_step(X)
            self.M_step(X)
            neg_log_likely = self.compute_neg_log_likely(X)
            change = neg_log_likely - prev_neg_log_likely
            if abs(change) < self.tol:
                break
        self.n_iter = n_iter
        self.neg_log_likely = neg_log_likely
        

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



def NLL_V(predictions, inputs, heart):
    (K,X,Y)=predictions.shape
    M=inputs.shape[0]
    
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
            if np.sum(pred[k,...]) != 0:
                mu[k,m]=np.sum(pred[k,...]*inp[m,...])/np.sum(pred[k,...])
                var[k,m]=(np.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(np.sum(pred[k,...])))
    temp=np.zeros((K,M,inp.shape[1]))
    for k in range(K):
        for m in range(M):
            if np.sum(pred[k,...]) != 0:
                temp[k,m,...]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
    temp =  np.prod(temp,1)
    temp= pred * temp
    likelyloss = -np.mean(np.log(np.sum(temp,axis=0)))
    return likelyloss



def NLL(predictions, inputs, heart):
    (K,X,Y)=predictions.shape
    M=inputs.shape[0]
    
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
            if np.sum(pred[k,...]) != 0:
                mu[k,m]=np.sum(pred[k,...]*inp[m,...])/(np.sum(pred[k,...]))
                var[k,m]=(np.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(np.sum(pred[k,...])))
    temp=np.zeros((K,M,inp.shape[1]))
    for k in range(K):
        for m in range(M):
            temp[k,m,...]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
    temp =  np.prod(temp,1)
    temp = alpha[:,np.newaxis] *temp
    likelyloss = -np.mean(np.log(np.sum(temp,axis=0)))
    return likelyloss

# def load_2dnet(path, device):
#     params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
#     weights = torch.load(path + f"/weights.pth",  map_location=torch.device(device))
#     net2d = get_network(architecture='unet2d', device=device, **params)
#     net2d.load_state_dict(weights)
#     net2d.eval()
#     return net2d


# def load_2dnet_single(path, device):
#     path_params = path[0:-13]+"config.json"
#     params= yaml.load(open(path_params, 'r'), Loader=yaml.SafeLoader)['network']
#     weights = torch.load(path,  map_location=torch.device(device))
#     net2d = get_network(architecture='unet2d', device=device, **params)
#     net2d.load_state_dict(weights)
#     net2d.eval()
#     return net2d

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
        

def save_checkpoint(net, checkpoint_dir, fold, name="weights", savepath=False, z_dim=False):
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(net.state_dict(), checkpoint_path)
    if savepath:
        if not os.path.exists('paths/'):
            os.makedirs('paths/')
        if z_dim==True:
            with open(f"paths/best_weights3d_{fold}.txt", "w") as text_file:
                text_file.write(checkpoint_dir+"/")
        else:
            with open(f"paths/best_weights2d_{fold}.txt", "w") as text_file:
                text_file.write(checkpoint_dir+"/")
                
                
