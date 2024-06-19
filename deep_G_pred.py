# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:44:58 2024

@author: A0067501
"""

import argparse
from utils.utils import load_2dnet, prepare_data, order_dice, dicecoeff, plot_result
from utils.losses import get_loss
import os
import torch
import numpy as np
import json

parser = argparse.ArgumentParser(description='Define hyperparameters for training.')

parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--type',type = str, help = "Type of Gaussian mixture model (deepG, deepSVG) ",  default = "deepG")
parser.add_argument('--lam',type = float, help = "Regularization parameter",  default = 1)
parser.add_argument('--patients', nargs='+', type=float, default= [i for i in range(101,126)])
args = parser.parse_args()

folder = f"RESULTS_FOLDER/{args.type}/multiple_images/lam={args.lam}/"

patient_nrs = args.patients
patients = [f"Case_{format(num, '03d')}" for num in patient_nrs]
files=[]
for patient in patients:
    files.append([os.path.join("DATA/preprocessed/myops_2d/",f) for f in os.listdir("DATA/preprocessed/myops_2d/") if f.startswith(patient)])
files= sum(files, [])

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

results= {}
classes= ["blood", "muscle", "edema", "scar"]

if args.type == "deepG":
    NLL = get_loss(crit="NormalGMM") 
elif args.type == "deepSVG":
    NLL = get_loss(crit="VariantGMM") 

try:
    net = load_2dnet(folder, device)
    for file in files:
        patient = file.split("/")[-1][0:-4]
        results[patient]={}
        X, gt, mask_heart = prepare_data(file)
        in_nn = torch.from_numpy(X[None, ...].astype("float32")).to(device)
        mask_nn = torch.from_numpy(mask_heart[None,None, ...].astype("float32")).to(device)
        pred= net(in_nn)[0]
        nll = NLL(pred, in_nn, mask_nn).item()
        results[patient]["NLL"]=nll
            
        pred = np.argmax(pred[0,...].cpu().detach().numpy(),0)+1
        pred[mask_heart==0]=0
        if args.lam ==0:
            pred = order_dice(pred, gt)
            
        for cl,i in zip(classes,range(1,5)):
            results[patient]["dice_"+cl] = dicecoeff((pred==i)*1, (gt==i)*1)
            
        np.save(os.path.join(folder,"predictions", patient), pred)
        
        plot_result(X, pred, gt, os.path.join(folder, "plots"), patient)
            
    with open(os.path.join(folder,'results.txt'), 'w') as f:
        json.dump(results, f, indent=4, sort_keys=False)
            
    print(f"Results saved in {folder}")
    
    
except:
    print("You first have to train a network for this parameters!")

