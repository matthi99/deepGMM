# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:17:44 2024

@author: A0067501
"""

#Packages and functions

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils.architectures import get_network
from utils.utils_test import prepare_data, normalize, variant_log_likelyhood, normal_log_likelyhood, variant_log_likelyhood, load_2dnet, order_dice, dicecoeff, load_2dnet_single
from utils.spatially_variant_gmm import VariantGMM
import yaml
import json
import pandas as pd



#prepare files

FOLDER= "DATA/preprocessed/traindata2d/"
VAL_PATIENTS = [109,114,117,120,122]


dice_coeffs= {}
dice_coeffs["GMM"]={}
dice_coeffs["SVGMM"]={}
dice_coeffs["GMM_CNN"]={}
dice_coeffs["SVGMM_CNN"]={}
likelyhoods = {}
likelyhoods["GMM"]=[]
likelyhoods["SVGMM"]=[]
likelyhoods["GMM_CNN"]=[]
likelyhoods["SVGMM_CNN"]=[]

patients = [f"Case_{format(num, '03d')}" for num in VAL_PATIENTS]
classes= ["blood", "muscle", "edema", "scar"]
for cl in classes: 
    dice_coeffs["GMM"][cl]=[]
    dice_coeffs["SVGMM"][cl]=[]
    dice_coeffs["GMM_CNN"][cl]=[]
    dice_coeffs["SVGMM_CNN"][cl]=[]
    
    
    

files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savefolder = "RESULTS_FOLDER/RESULTS/plots_for_paper/single/"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)


#%%
"""
Calculate results and plot them
"""

for file in files:
    #GMM_CNN
    slicenr =file.split(".")[0][-1]
    patientnr= file[-9:-6]
    path_net = f"RESULTS_FOLDER/normal_GMM/single/Patient_{patientnr}/weights_{slicenr}.pth"
    net = load_2dnet_single(path_net, device)
    
    
    X, gt, mask_heart = prepare_data(file)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    my_score = normal_log_likelyhood(pred, X, mask_heart)
    likelyhoods["GMM_CNN"].append(my_score)
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["GMM_CNN"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_gmm_cnn = pred.copy()
    
    #SVGMM_CNN
    path_net = f"RESULTS_FOLDER/spatially_variant_GMM/single/Patient_{patientnr}/weights_{slicenr}.pth"
    net = load_2dnet_single(path_net, device)
    
    X, gt, mask_heart = prepare_data(file)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["SVGMM_CNN"].append(score)
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["SVGMM_CNN"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_svgmm_cnn = pred.copy()
    
    #GMM
    X, gt, mask_heart = prepare_data(file)
    patientnr= file[-9:-6]
    
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    
    gmm = GMM(n_components=4, covariance_type="diag", tol=0.001, init_params= "kmeans")
    gmm.fit(in_gmm)
    
    #score = -gmm.score(in_gmm)
    probs=gmm.predict_proba(in_gmm)
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    score = normal_log_likelyhood(pred, X, mask_heart)
    likelyhoods["GMM"].append(score)
    
    pred = np.zeros_like(mask_heart)
    labels = gmm.predict(in_gmm)
    pred[mask_heart==1]=labels+1
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["GMM"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
    
    pred_gmm = pred.copy()
    
    
    #SVGMM
    vgmm = VariantGMM(n_components=4, tol=0.001, init_params= "random")
    vgmm.fit(in_gmm)
    
    probs= vgmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["SVGMM"].append(score)
    
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["SVGMM"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
    
    pred_svgmm = pred.copy()
    
    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(X[2,...], cmap = "gray")
    plt.axis("off")
    plt.title("C0")
    plt.subplot(2,4,2)
    plt.imshow(X[1,...], cmap = "gray")
    plt.axis("off")
    plt.title("T2")
    plt.subplot(2,4,3)
    plt.imshow(X[0,...], cmap = "gray")
    plt.axis("off")
    plt.title("LGE")
    plt.subplot(2,4,4)
    plt.imshow(gt, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("Ground truth")
    predictions_gt=np.zeros((4,160,160))
    predictions_gt[0,...][gt==1]=1
    predictions_gt[1,...][gt==2]=1
    predictions_gt[2,...][gt==3]=1
    predictions_gt[3,...][gt==4]=1
    plt.text(0, 200, f'NLL: {np.round(normal_log_likelyhood(predictions_gt, X, mask_heart),3)}', fontsize = 9)
    
    plt.subplot(2,4,5)
    plt.imshow(pred_gmm, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("GMM")
    dice=0
    #dice = np.mean(dice_coeffs["GMM"][f"Case_{patientnr}"][-1])
    score = likelyhoods["GMM"][-1]
    plt.text(0, 180, f'Mean Dice: {np.round(dice,3)}', fontsize = 9)
    plt.text(0, 200, f'neg-ll: {np.round(score,3)}', fontsize = 9)
    
    plt.subplot(2,4,6)
    plt.imshow(pred_svgmm, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("SVGMM")
    #dice = np.mean(dice_coeffs["SVGMM"][f"Case_{patientnr}"][-1])
    score = likelyhoods["SVGMM"][-1]
    plt.text(0, 180, f'Mean Dice: {np.round(dice,3)}', fontsize = 9)
    plt.text(0, 200, f'neg-ll: {np.round(score,3)}', fontsize = 9)
    
    plt.subplot(2,4,7)
    plt.imshow(pred_gmm_cnn, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("GMM_CNN")
    #dice = np.mean(dice_coeffs["GMM_CNN"][f"Case_{patientnr}"][-1])
    score = likelyhoods["GMM_CNN"][-1]
    plt.text(0, 180, f'Mean Dice: {np.round(dice,3)}', fontsize = 9)
    plt.text(0, 200, f'neg-ll: {np.round(score,3)}', fontsize = 9)
    
    plt.subplot(2,4,8)
    plt.imshow(pred_svgmm_cnn, interpolation="none", vmin=0, vmax=4)
    plt.axis("off")
    plt.title("SVGMM_CNN")
    #dice = np.mean(dice_coeffs["SVGMM_CNN"][f"Case_{patientnr}"][-1])
    score = likelyhoods["SVGMM_CNN"][-1]
    plt.text(0, 180, f'Mean Dice: {np.round(dice,3)}', fontsize = 9)
    plt.text(0, 200, f'neg-ll: {np.round(score,3)}', fontsize = 9)
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}.png"), bbox_inches='tight', dpi=500)
    plt.show()
    
results={}
methods = dice_coeffs.keys()
for method in methods:
    results[method]={}
    for cl in classes:
        results[method][cl]=np.mean(dice_coeffs[method][cl])
    mean = np.mean(list(results[method].values()))
    results[method]["mean"]=mean

df=pd.DataFrame(data=results)
df = df.transpose()
print(df)

results={}
methods = likelyhoods.keys()
for method in methods:
    results[method]=np.mean(likelyhoods[method])


df=pd.DataFrame(data=results)
df = df.transpose()
