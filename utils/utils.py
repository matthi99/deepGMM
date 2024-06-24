# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:18:54 2024

@author: A0067501
"""


import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import os
import torch
import logging
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from utils.unet import UNet2D


class SVGMM():
    """
    Spatially variant Gaussian mixture model (SVGMM). 
    This class allows to estimate the parameters of a spatially variant Gaussian mixture
    distribution with the help of EM-algorithm.
    Initialization is build upon GaussianMixture class of the sklearn package
    
    """
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
        


#helper fuctions 

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

def NLL(X, mu, var, pi):
    """Calculate NLL of GMM """
    K,M = mu.shape
    temp=np.zeros((K,M,X.shape[0]))
    for k in range(K):
        for m in range(M):
            temp[k,m,:]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((X[:,m]-mu[k,m])**2/(2*var[k,m])))
    temp =  np.prod(temp,1)
    temp = pi[:,np.newaxis] *temp
    nll = -np.mean(np.log(np.sum(temp,axis=0)))
    return nll
    
def NLL_V(X, mu, var, weights):
    """Calculate NLL of SVGMM (NLL_V)"""
    K,M = mu.shape
    temp=np.zeros((K,M,X.shape[0]))
    for k in range(K):
        for m in range(M):
            temp[k,m,:]=(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((X[:,m]-mu[k,m])**2/(2*var[k,m])))
    temp =  np.prod(temp,1)
    temp = weights.T *temp
    nll = -np.mean(np.log(np.sum(temp,axis=0)))
    return nll

def load_2dnet(path, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/weights.pth",  map_location=torch.device(device))
    net2d = UNet2D(**params).to(device)         
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
    """
    Rearrange the channels of the prediction such that the Dice coefficient with the ground truth 
    segmentation gets maximized.
    """
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
        
def save_checkpoint(net, checkpoint_dir, name="weights"):
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(net.state_dict(), checkpoint_path)
    
def get_logger(name, level=logging.INFO, formatter = '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.handler_set = True
    return logger

def plot_result(X, pred, gt, savepath, file):
    cmap = cm.get_cmap("jet").copy()
    cmap.set_bad(color='black')
    classes= ["blood", "muscle", "edema", "scar"]
    values = [1,2,3,4]
    plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(X[2,...], cmap = "gray")
    plt.axis("off")
    plt.title("bSSFP", fontsize = 11)
    plt.subplot(1,5,2)
    plt.imshow(X[1,...], cmap = "gray")
    plt.axis("off")
    plt.title("T2")
    plt.subplot(1,5,3)
    plt.imshow(X[0,...], cmap = "gray")
    plt.axis("off")
    plt.title("LGE",fontsize = 11)
    plt.subplot(1,5,4)
    pred_masked = np.ma.masked_where(pred ==0, pred)
    plt.imshow(pred_masked, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("Prediction",fontsize = 11)
    plt.subplot(1,5,5)
    masked_gt = np.ma.masked_where(gt ==0, gt)
    im=plt.imshow(masked_gt, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.7, 1.03), loc=2,fontsize=5 )
    plt.axis("off")
    plt.title("Ground truth",fontsize = 11)
    plt.savefig(os.path.join(savepath,f"{file}.png"), bbox_inches='tight', dpi=300)
    plt.close()