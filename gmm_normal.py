# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:53:47 2024

@author: A0067501
"""

#Packages and functions

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils.architectures import get_network
import yaml

def normalize(img):
    return (img - np.mean(img)) / np.std(img)


def neg_log_likely(predictions, inputs, heart):
    K=4
    M=3
    mu=np.zeros((K,M))
    var=np.zeros((K,M))
    eps=1e-10
    for k in range(K):
        for m in range(M):
            mu[k,m]=np.sum(predictions[k,...]*inputs[m,...])/(np.sum(predictions[k,...]))
            var[k,m]=(np.sum(predictions[k,...]*(inputs[m,...]-mu[k,m])**2)/(np.sum(predictions[k,...])))
    
    temp=np.zeros((5,3,160,160))
    for k in range(K):
        for m in range(M):
            temp[k,m,...]=predictions[k]*(1/(np.sqrt(2*np.pi*var[k,m])))*np.exp(-((inputs[m,...]-mu[k,m])**2/(2*var[k,m])))
    for m in range(M):        
        temp[4,m,...]=1-heart
    neg_log_likely=-np.sum(np.log(np.sum(np.prod(temp,1),axis=0)))
    return neg_log_likely, temp

def load_2dnet(path, epoch, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + f"/weights_{epoch}.pth",  map_location=torch.device(device))
    net2d = get_network(architecture='unet2d', device=device, **params)
    net2d.load_state_dict(weights)
    net2d.eval()
    return net2d

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


def mean(X, pred):
    mu = np.zeros((5, X.shape[0]))
    classes = np.unique(pred).astype(int)
    for i in classes:
        for j in range(mu.shape[1]):
            mu[i,j]= np.mean(X[j,...][pred==i])
    return mu


def order (pred, gt, means, gt_means):
    ordered= np.zeros_like(pred)
    classes = np.unique(gt)[1:]
    print(classes)
    for i in range(1,5):
        diff= 100*np.ones(5)
        for j in classes:
            diff[j]=np.sum(abs(gt_means[j]-means[i]))
        cl= np.argmin(diff)
        ordered[pred == i]=cl
    return ordered

def dicecoeff(pred, gt):
    eps=1 
    intersection = (pred * gt).sum()
    total = (pred + gt).sum()
    coeff = (2 * intersection + eps) / (total + eps)
    return coeff


def order_dice(pred, gt):
    ordered= np.zeros_like(pred)
    for i in np.unique(pred)[1:]:
        pred_class = (pred==i)*1
        dice = np.zeros(4)
        for j in range(1,5):
            dice[j-1]= dicecoeff(pred_class, (gt==j)*1)
        gt_class = np.argmax(dice)+1
        ordered[pred_class==1]=gt_class
    return ordered
    
            
    

#%%
#Prepare files

FOLDER= "DATA/preprocessed/traindata2d/"
VAL_PATIENTS = [109,114,117,120,122]

patients = [f"Case_{format(num, '03d')}" for num in VAL_PATIENTS]

files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])



#%%
"""
Normal Gaussian mixture model
"""



for test in files:
    X, gt, mask_heart = prepare_data(test)
    
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    
    means_init= np.array([[ 1.13263178, -0.13630785,  1.6058956 ],
                         [-0.41842756,  0.51262672, -0.00925156],
                         [ 0.29902016,  1.25442789,  0.52296464],
                         [ 1.15254939,  1.04698159,  0.3860968 ]])
    
    
    gmm = GMM(n_components=4, covariance_type="diag")
    gmm.fit(in_gmm)
    labels = gmm.predict(in_gmm)
    pred = np.zeros_like(mask_heart)
    pred[mask_heart==1]=labels+1
    
                
    gmm_means = mean(X, pred)
    gt_means = mean(X, gt)
    means = np.zeros((5,3))
    means[1:,:]= means_init
    pred = order_dice(pred, gt)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(X[0,...])
    plt.axis("off")
    plt.title("Normal")
    plt.subplot(1,3,2)
    plt.imshow(X[0,...])
    plt.imshow(pred, alpha=0.9, interpolation="none")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(X[0,...])
    plt.imshow(gt, alpha=0.9, interpolation="none")
    plt.axis("off")
    plt.show()
    
    predictions=np.zeros((4,160,160))
    predictions[0,...][pred==1]=1
    predictions[1,...][pred==2]=1
    predictions[2,...][pred==3]=1
    predictions[3,...][pred==4]=1
    
    
    print(neg_log_likely(predictions, X, mask_heart)[0])
    
    #%%
    """
    Network predictions
    """
    
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_net = "RESULTS_FOLDER/2d-net_test_with_mu0"
    net = load_2dnet(path_net, 4, device)
    
    
    
    X, gt, mask_heart = prepare_data(test)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    pred= np.argmax(pred, axis=0)
    pred[mask_heart==0]=0
    
    
    # gmm_means = mean(X, pred)
    # gt_means = mean(X, gt)
    #pred = order(pred, gt, gmm_means, means)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(X[0])
    plt.axis("off")
    plt.title("Network")
    plt.subplot(1,3,2)
    plt.imshow(X[0])
    plt.imshow(pred, alpha=0.9, interpolation="none")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(X[0])
    plt.imshow(gt, alpha=0.9, interpolation="none")
    plt.axis("off")
    plt.show()
    
    
    predictions=np.zeros((4,160,160))
    predictions[0,...][pred==1]=1
    predictions[1,...][pred==2]=1
    predictions[2,...][pred==3]=1
    predictions[3,...][pred==4]=1
    
    
    print(neg_log_likely(predictions,X, mask_heart)[0])

     
    
 