# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:21:43 2024

@author: A0067501
"""

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import os
import json
import argparse

from utils.utils import prepare_data, NLL, order_dice, dicecoeff, plot_result


#Define parameters
parser = argparse.ArgumentParser(description='Define hyperparameters for predictions.')
parser.add_argument('--mu_data', type=bool, default=False)
parser.add_argument('--tol', type=float, default=0.0002)
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
    savefolder = "RESULTS_FOLDER/EM-GMM/GMM_mu"
else:
    init_params = "random"
    savefolder = "RESULTS_FOLDER/EM-GMM/GMM"
if not os.path.exists(savefolder):
    os.makedirs(os.path.join(savefolder,"plots"))
    os.makedirs(os.path.join(savefolder,"predictions"))

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
    patient = file.split("/")[-1][0:-4]
    results[patient]={}
    LGE= X[0,...][mask_heart==1]
    T2 = X[1,...][mask_heart==1]
    C0 = X[2,...][mask_heart==1]
    in_gmm=np.stack((LGE,T2,C0), axis=1)
    if mu_data == True:
        gmm = GMM(n_components=4, covariance_type="diag", tol= tol, means_init = means_init, random_state=random_state)
    else:
        gmm = GMM(n_components=4, covariance_type="diag", tol= tol, init_params= init_params, random_state=random_state)
    gmm.fit(in_gmm)
    nll = NLL(in_gmm, gmm.means_, gmm.covariances_, gmm.weights_)
    results[patient]["NLL"]=nll
    
    #labeling rule
    probs=gmm.predict_proba(in_gmm)
    pred= np.zeros((4,160,160))
    for i in range(4):
        pred[i][mask_heart==1]=probs[:,i]
    
    probs_gmm = pred.copy()
    pred = np.zeros_like(mask_heart)
    labels = gmm.predict(in_gmm)
    pred[mask_heart==1]=labels+1
    pred = order_dice(pred, gt)
    
    for cl,i in zip(classes,range(1,5)):
        results[patient]["dice_"+cl] = dicecoeff((pred==i)*1, (gt==i)*1)
    
    np.save(os.path.join(savefolder,"predictions", patient), pred)
    
    plot_result(X, pred, gt, os.path.join(savefolder, "plots"), patient)
    
with open(os.path.join(savefolder,'results.txt'), 'w') as f:
    json.dump(results, f, indent=4, sort_keys=False)
    
print(f"Results saved in {savefolder}")