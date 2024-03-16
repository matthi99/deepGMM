# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:55:08 2024

@author: A0067501
"""

#Packages and functions

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils.architectures import get_network
from utils.utils_test import prepare_data, normalize, variant_log_likelyhood, normal_log_likelyhood, load_2dnet, order_dice, dicecoeff
from utils.spatially_variant_gmm import VariantGMM
import yaml
import json
import pandas as pd




#%%
#Prepare files

lam = 1.0

if lam == 0:
    savefolder = "RESULTS_FOLDER/RESULTS/spatially_variant_GMM/multiple/"
else:
    savefolder = f"RESULTS_FOLDER/RESULTS/spatially_variant_GMM/multiple_reg_{lam}/"

if not os.path.exists(savefolder):
    os.makedirs(savefolder)

    
FOLDER= "DATA/preprocessed/traindata2d/"
VAL_PATIENTS = [109,114,117,120,122]


dice_coeffs= {}
dice_coeffs["Network"]={}
dice_coeffs["EM"]={}
likelyhoods = {}
likelyhoods["Network"]={}
likelyhoods["EM"]={}


patients = [f"Case_{format(num, '03d')}" for num in VAL_PATIENTS]
for patient in patients: 
    dice_coeffs["EM"][patient]=[]
    likelyhoods["EM"][patient]=[]
    dice_coeffs["Network"][patient]=[]
    likelyhoods["Network"][patient]=[]

files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])


#%%
"""
Normal Gaussian mixture model
"""

for file in files:
    X, gt, mask_heart = prepare_data(file)
    patientnr= file[-9:-6]
    
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    
    
    if lam == 0:
        gmm = VariantGMM(n_components=4, tol=0.005)
    else:
        means_init= np.array([[ 1.0907561,  -0.15736851,  1.70052552],
                             [-0.38956344,  0.35907112,  0.00187305],
                             [ 0.12236157,  0.99027221,  0.42901193],
                             [ 1.25641782,  0.86135793,  0.53020001]])
        gmm = VariantGMM(n_components=4, means_init=means_init,  tol=0.005)
    
    
    gmm.fit(in_gmm)
    
    probs= gmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["EM"][f"Case_{patientnr}"].append(my_score)
    
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    if lam == 0:
        pred = order_dice(pred, gt)
    pred_gmm = pred.copy()
    
    dice=[]
    for i in range(1,5):
        dice.append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    dice_coeffs["EM"][f"Case_{patientnr}"].append(dice)
    
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(X[0,...])
    plt.axis("off")
    plt.title("LGE")
    plt.subplot(2,3,2)
    plt.imshow(X[1,...])
    plt.axis("off")
    plt.title("T2")
    plt.subplot(2,3,3)
    plt.imshow(X[2,...])
    plt.axis("off")
    plt.title("C0")
    
    
    plt.subplot(2,3,4)
    plt.imshow(X[0,...])
    plt.imshow(pred, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("EM")
    plt.text(0, 180, f'Mean Dice: {np.round(np.mean(dice),3)}', fontsize = 9)
    
    
    predictions=np.zeros((4,160,160))
    predictions[0,...][pred==1]=1
    predictions[1,...][pred==2]=1
    predictions[2,...][pred==3]=1
    predictions[3,...][pred==4]=1
    plt.text(0, 200, f'neg-ll: {np.round(my_score,3)}', fontsize = 9)
    #plt.text(0, 220, f'neg-ll2: {np.round(score,3)}', fontsize = 9)
    

    #%%
    """
    Network predictions (multiple)
    """
    
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if lam == 0:
        path_net = "RESULTS_FOLDER/spatially_variant_GMM/multiple/"
    else:
        path_net = f"RESULTS_FOLDER/spatially_variant_GMM/multiple_reg_{lam}/"
    net = load_2dnet(path_net, device)
    
    
    X, gt, mask_heart = prepare_data(file)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["Network"][f"Case_{patientnr}"].append(my_score)
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    if lam == 0:
        pred = order_dice(pred, gt)
    
    
    dice=[]
    for i in range(1,5):
        dice.append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    dice_coeffs["Network"][f"Case_{patientnr}"].append(dice)
    
    plt.subplot(2,3,5)
    plt.imshow(pred, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("Network")
    plt.text(0, 180, f'Mean Dice: {np.round(np.mean(dice),3)}', fontsize = 9)
    plt.text(0, 200, f'neg-ll: {np.round(my_score,3)}', fontsize = 9)
   
    plt.subplot(2,3,6)
    plt.imshow(gt, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("Ground truth")
    predictions_gt=np.zeros((4,160,160))
    predictions_gt[0,...][gt==1]=1
    predictions_gt[1,...][gt==2]=1
    predictions_gt[2,...][gt==3]=1
    predictions_gt[3,...][gt==4]=1
    plt.text(0, 200, f'neg-ll: {np.round(variant_log_likelyhood(predictions_gt, X, mask_heart),3)}', fontsize = 9)
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}.png"), bbox_inches='tight', dpi=500)
    plt.show()
    
    
with open(savefolder+'results_dice.txt', 'w') as f:
        json.dump(dice_coeffs, f, indent=4, sort_keys=False)
        
with open(savefolder+'results_likely.txt', 'w') as f:
        json.dump(likelyhoods, f, indent=4, sort_keys=False)
    
df=pd.DataFrame(data=dice_coeffs)
df = df.transpose()
df.to_excel(savefolder+"results_dice.xlsx")
    
df=pd.DataFrame(data=likelyhoods)
df = df.transpose()
df.to_excel(savefolder+"results_likely.xlsx")
    
    

     
    
 