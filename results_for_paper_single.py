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
import matplotlib
import matplotlib.patches as mpatches


#prepare files

FOLDER= "DATA/preprocessed/traindata2d/"
VAL_PATIENTS = [i for i in range(101,126)]
#VAL_PATIENTS=[109,114,117,120,122]


dice_coeffs= {}
dice_coeffs["GMM"]={}
dice_coeffs["GMM_CNN"]={}
dice_coeffs["SVGMM"]={}
dice_coeffs["SVGMM_CNN"]={}
likelyhoods = {}
likelyhoods["GMM"]=[]
likelyhoods["GMM_CNN"]=[]
likelyhoods["SVGMM"]=[]
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


#set cmap and classes
cmap = matplotlib.cm.get_cmap("jet").copy()
cmap.set_bad(color='black')
classes= ["blood", "muscle", "edema", "scar"]
values = [1,2,3,4]

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
    
    gmm = GMM(n_components=4, covariance_type="diag", tol=0.0002, init_params= "random", random_state=33)
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
    vgmm = VariantGMM(n_components=4, tol=0.001, init_params= "random", random_state =33)
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
    plt.subplots_adjust(left  = 0.125,  
                        right = 0.9,   
                        bottom = 0,  
                        top = 0.7,      
                        wspace = 0.1,   
                        hspace = 0 )
    plt.subplot(2,4,1)
    plt.imshow(X[2,...], cmap = "gray")
    plt.axis("off")
    plt.title("bSSFP", fontsize = 11)
    plt.subplot(2,4,2)
    plt.imshow(X[1,...], cmap = "gray")
    plt.axis("off")
    plt.title("T2")
    plt.subplot(2,4,3)
    plt.imshow(X[0,...], cmap = "gray")
    plt.axis("off")
    plt.title("LGE",fontsize = 11)
    plt.subplot(2,4,4)
    masked_gt = np.ma.masked_where(gt ==0, gt)
    im=plt.imshow(masked_gt, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]
    #plt.legend(handles=patches, bbox_to_anchor=(0.6, 1.03), loc=2,fontsize=5 )
    plt.axis("off")
    plt.title("Ground truth",fontsize = 11)
    predictions_gt=np.zeros((4,160,160))
    predictions_gt[0,...][gt==1]=1
    predictions_gt[1,...][gt==2]=1
    predictions_gt[2,...][gt==3]=1
    predictions_gt[3,...][gt==4]=1
    #plt.text(0, 200, f'NLL: {np.round(normal_log_likelyhood(predictions_gt, X, mask_heart),3)}', fontsize = 9)
    
    plt.subplot(2,4,5)
    pred_gmm = np.ma.masked_where(pred_gmm ==0, pred_gmm)
    plt.imshow(pred_gmm, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("GMM",fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["GMM"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["GMM"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,7)
    pred_svgmm = np.ma.masked_where(pred_svgmm ==0, pred_svgmm)
    plt.imshow(pred_svgmm, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("SVGMM",fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["SVGMM"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["SVGMM"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL_V: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,6)
    pred_gmm_cnn = np.ma.masked_where(pred_gmm_cnn ==0, pred_gmm_cnn)
    plt.imshow(pred_gmm_cnn, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("GMM$_{CNN}$", fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["GMM_CNN"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["GMM_CNN"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,8)
    pred_svgmm_cnn = np.ma.masked_where(pred_svgmm_cnn ==0, pred_svgmm_cnn)
    plt.imshow(pred_svgmm_cnn, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("SVGMM$_{CNN}$", fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["SVGMM_CNN"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["SVGMM_CNN"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL_V: {np.round(score,3)}$', fontsize = 9)
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}.png"), bbox_inches='tight', dpi=500)
    plt.show()
    
results={}
methods = dice_coeffs.keys()
for method in methods:
    results[method]={}
    for cl in classes:
        results[method][cl]=round(np.mean(dice_coeffs[method][cl]),3)
    mean = round(np.mean(list(results[method].values())),3)
    results[method]["mean"]=mean
    results[method]["NLL"]=round(np.mean(likelyhoods[method]),3)

df_means=pd.DataFrame(data=results)
df_means = df_means.transpose()
print(df_means)

results={}
methods = dice_coeffs.keys()
for method in methods:
    results[method]={}
    for cl in classes:
        results[method][cl]=round(np.std(dice_coeffs[method][cl]),3)
    std = round(np.std(list(results[method].values())),3)
    results[method]["mean"]=std
    results[method]["NLL"]=round(np.std(likelyhoods[method]),3)

df_stds=pd.DataFrame(data=results)
df_stds = df_stds.transpose()
print(df_stds)