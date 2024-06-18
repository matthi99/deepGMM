# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:40:40 2024

@author: A0067501
"""


import numpy as np
import os
from utils import prepare_data, NLL_V, order_dice, dicecoeff, SVGMM
import json
import argparse


#Define parameters
parser = argparse.ArgumentParser(description='Define hyperparameters for predictions.')
parser.add_argument('--mu_data', type=bool, default=True)
parser.add_argument('--tol', type=float, default=0.001)
parser.add_argument('--patients', nargs='+', type=float, default= [i for i in range(101,126)])
parser.add_argument('--random_state', type = int, default = 33)
args = parser.parse_args()


#parameters and folders
FOLDER= "DATA/preprocessed/myops_2d/"
PATIENT_NRS = args.patients
patients = [f"Case_{format(num, '03d')}" for num in PATIENT_NRS]
mu_data = args.mu_data
tol= args.tol
random_state = args.random_state
if mu_data == True:
    means_init = np.load("DATA/mu_data.npy")
    savefolder = "RESULTS_FOLDER/SVGMM/SVGMM_mu"
else:
    init_params = "random"
    savefolder = "RESULTS_FOLDER/SVGMM/SVGMM"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

results= {}
classes= ["blood", "muscle", "edema", "scar"]
    
files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])

#Segmentation with GMM
for file in files:
    #GMM
    X, gt, mask_heart = prepare_data(file)
    patient = file.split("/")[-1]
    results[patient]={}
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    if mu_data == True:
        vgmm = SVGMM(n_components=4, covariance_type="diag", tol= tol, means_init = means_init, random_state=random_state)
    else:
        vgmm = SVGMM(n_components=4, covariance_type="diag", tol= tol, init_params= init_params, random_state=random_state)
    vgmm.fit(in_gmm)
    
    #labeling rule not neccessary
    probs= vgmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    score = NLL_V(pred, X, mask_heart)
    results[patient]["NLL"]=score
    
    probs_svgmm = pred.copy()
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        results[patient]["dice_"+cl] = dicecoeff((pred==i)*1, (gt==i)*1)
    
    np.save(os.path.join(savefolder,patient), pred)
    
with open(os.path.join(savefolder,'results.txt'), 'w') as f:
    json.dump(results, f, indent=4, sort_keys=False)