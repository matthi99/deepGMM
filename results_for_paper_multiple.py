# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:33:43 2024

@author: matth
"""

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
VAL_PATIENTS=[109,114,117,120,122]


dice_coeffs= {}
dice_coeffs["SVGMM"]={}
dice_coeffs["CNN"]={}
dice_coeffs["SVGMM_mu"]={}
dice_coeffs["CNN_mu"]={}

likelyhoods = {}
likelyhoods["SVGMM"]=[]
likelyhoods["CNN"]=[]
likelyhoods["SVGMM_mu"]=[]
likelyhoods["CNN_mu"]=[]

patients = [f"Case_{format(num, '03d')}" for num in VAL_PATIENTS]
classes= ["blood", "muscle", "edema", "scar"]
for cl in classes: 
    dice_coeffs["SVGMM"][cl]=[]
    dice_coeffs["CNN"][cl]=[]
    dice_coeffs["SVGMM_mu"][cl]=[]
    dice_coeffs["CNN_mu"][cl]=[]
    
    
    

files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savefolder = "RESULTS_FOLDER/RESULTS/plots_for_paper/multiple/"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
    

#set cmap and classes
cmap = matplotlib.cm.get_cmap("jet").copy()
cmap.set_bad(color='black')
classes= ["blood", "muscle", "edema", "scar"]
values = [1,2,3,4]
    
lam =1.0

for file in files:
    #CNN
    slicenr =file.split(".")[0][-1]
    patientnr= file[-9:-6]
    
    path_net = "RESULTS_FOLDER/spatially_variant_GMM/multiple/"
    
    net = load_2dnet(path_net, device)
    
    X, gt, mask_heart = prepare_data(file)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["CNN"].append(my_score)
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["CNN"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_cnn = pred.copy()
    
    #CNN_mu
    path_net = f"RESULTS_FOLDER/spatially_variant_GMM/multiple_reg_{lam}/"
    net = load_2dnet(path_net, device)
    
    X, gt, mask_heart = prepare_data(file)
    in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
    pred= net(in_nn)[0][0,...].cpu().detach().numpy()
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["CNN_mu"].append(my_score)
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["CNN_mu"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_cnn_mu = pred.copy()
    
    #SVGMM
    X, gt, mask_heart = prepare_data(file)
    
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    
    gmm = VariantGMM(n_components=4, tol=0.001, init_params= "random", random_state=33)
    gmm.fit(in_gmm)
    
    probs= gmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["SVGMM"].append(my_score)
    
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["SVGMM"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_svgmm = pred.copy()
    
    
    #SVGMM_mu
    X, gt, mask_heart = prepare_data(file)
    
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    
    
    means_init= np.array([[ 1.0907561,  -0.15736851,  1.70052552],
                             [-0.38956344,  0.35907112,  0.00187305],
                             [ 0.12236157,  0.99027221,  0.42901193],
                             [ 1.25641782,  0.86135793,  0.53020001]])
    gmm = VariantGMM(n_components=4, means_init=means_init,  tol=0.001, random_state=33)
    gmm.fit(in_gmm)
    
    probs= gmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    my_score = variant_log_likelyhood(pred, X, mask_heart)
    likelyhoods["SVGMM_mu"].append(my_score)
    
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    
    for cl,i in zip(classes,range(1,5)):
        dice_coeffs["SVGMM_mu"][cl].append(dicecoeff((pred==i)*1, (gt==i)*1))
        
    pred_svgmm_mu = pred.copy()
    
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
    
    
    plt.subplot(2,4,5)
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
    plt.text(0, 200, f'$NLL: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,7)
    pred_svgmm_mu = np.ma.masked_where(pred_svgmm_mu ==0, pred_svgmm_mu)
    plt.imshow(pred_svgmm_mu, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("SVGMM$_{\mu}$",fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["SVGMM_mu"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["SVGMM_mu"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL_V: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,6)
    pred_cnn = np.ma.masked_where(pred_cnn ==0, pred_cnn)
    plt.imshow(pred_cnn, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("CNN", fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["CNN"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["CNN"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL: {np.round(score,3)}$', fontsize = 9)
    
    plt.subplot(2,4,8)
    pred_cnn_mu = np.ma.masked_where(pred_cnn_mu ==0, pred_cnn_mu)
    plt.imshow(pred_cnn_mu, interpolation="none", vmin=0, vmax=4, cmap = cmap)
    plt.axis("off")
    plt.title("CNN$_{\mu}$", fontsize = 11)
    dice=0
    for cl in classes:
        dice+=dice_coeffs["CNN_mu"][cl][-1]
    dice /= len(classes) 
    score = likelyhoods["CNN_mu"][-1]
    plt.text(0, 180, f'Mean Dice: ${np.round(dice,2)}$', fontsize = 9)
    plt.text(0, 200, f'$NLL_V: {np.round(score,3)}$', fontsize = 9)
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}.png"), bbox_inches='tight', dpi=500)
    plt.show()
    
results={}
methods = dice_coeffs.keys()
for method in methods:
    results[method]={}
    for cl in classes:
        results[method][cl]=round(np.mean(dice_coeffs[method][cl]),2)
    mean = round(np.mean(list(results[method].values())),2)
    results[method]["mean"]=mean
    results[method]["NLL"]=round(np.mean(likelyhoods[method]),2)

df_means=pd.DataFrame(data=results)
df_means = df_means.transpose()
print(df_means)

results={}
methods = dice_coeffs.keys()
for method in methods:
    results[method]={}
    for cl in classes:
        results[method][cl]=round(np.std(dice_coeffs[method][cl]),2)
    std = round(np.std(list(results[method].values())),2)
    results[method]["mean"]=std
    results[method]["NLL"]=round(np.std(likelyhoods[method]),3)

df_stds=pd.DataFrame(data=results)
df_stds = df_stds.transpose()
print(df_stds)

    
    
    
    
    
    