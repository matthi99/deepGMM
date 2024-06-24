# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:40:40 2024

@author: A0067501
"""

import numpy as np
import os
import json
import argparse

from utils.utils import prepare_data, NLL_V, order_dice, dicecoeff, SVGMM, plot_result

#Define parameters
parser = argparse.ArgumentParser(description='Define hyperparameters for predictions.')
parser.add_argument('--mu_data', type=bool, default=False)
parser.add_argument('--tol', type=float, default=0.001)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--patients', nargs='+', type=int, default= [i for i in range(101,126)])
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
    savefolder = "RESULTS_FOLDER/EM-SVGMM/SVGMM_mu"
else:
    init_params = "random"
    savefolder = "RESULTS_FOLDER/EM-SVGMM/SVGMM"
if not os.path.exists(savefolder):
    os.makedirs(os.path.join(savefolder,"plots"))
    os.makedirs(os.path.join(savefolder,"predictions"))


#create results dictionary and filelist
results= {}
classes= ["blood", "muscle", "edema", "scar"]
files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
files= sum(files, [])

#Segmentation with SVGMM
for file in files:
    #prepare data
    X, gt, mask_heart = prepare_data(file)
    patient = file.split("/")[-1][0:-4]
    results[patient]={}
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    #EM-algorithm
    if mu_data == True:
        vgmm = SVGMM(n_components=4, covariance_type="diag", tol= tol, max_iter = args.max_iter, means_init = means_init, random_state=random_state)
    else:
        vgmm = SVGMM(n_components=4, covariance_type="diag", tol= tol, max_iter = args.max_iter, init_params= init_params, random_state=random_state)
    vgmm.fit(in_gmm)
    nll = NLL_V(in_gmm, vgmm.means_, vgmm.covariances_, vgmm.weights_)
    results[patient]["NLL"]=nll
    
    #labeling rule not neccessary
    probs= vgmm.weights_
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    
    #get labels to image and calculate results
    probs_svgmm = pred.copy()
    pred= np.argmax(pred, axis=0)+1
    pred[mask_heart==0]=0
    pred = order_dice(pred, gt)
    for cl,i in zip(classes,range(1,5)):
        results[patient]["dice_"+cl] = dicecoeff((pred==i)*1, (gt==i)*1)
    
    #save predictions and plots
    np.save(os.path.join(savefolder,"predictions", patient), pred)
    plot_result(X, pred, gt, os.path.join(savefolder, "plots"), patient)

#save results    
with open(os.path.join(savefolder,'results.txt'), 'w') as f:
    json.dump(results, f, indent=4, sort_keys=False)
    
print(f"Results saved in {savefolder}")