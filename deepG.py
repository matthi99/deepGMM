# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:17:35 2024

@author: A0067501
"""

import torch
from unet import UNet2D
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, prepare_data
from losses import get_loss


parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--min_epochs', type=int, default=10)
parser.add_argument('--type',type = str, help = "Type of Gaussian mixture model (deepG, deepSVG) ",  default = "deepG")
parser.add_argument('--lam',type = float, help = "Regularization parameter",  default = 0)
parser.add_argument('--tol',type = float, help = "Tolerance for stopping criteria",  default = 0.005)
parser.add_argument('--patients', nargs='+', type=float, default= [i for i in range(101,126)])
args = parser.parse_args()


config = {
    "network":{
        "activation": "leakyrelu",
        "dropout": 0,
        "batchnorm": True,
        "start_filters": 32,
        "in_channels":3,
        "L":4,
        "out_channels": 4,
        "residual": False, 
        "last_activation":"softmax"},
    
    "classes": ["blood", "muscle", "edema", "scar"],
}






if args.type == "deepG":
    main_loss = get_loss(crit="NormalGMM") 
    if args.lam == 0:
        savefolder = "RESULTS_FOLDER/deepG/"
        
    else:
        savefolder = f"RESULTS_FOLDER/deepG_reg_{args.lam}/"
        reg_loss = get_loss(crit="mu_data") 
    
elif args.type == "deepSVG":
    main_loss = get_loss(crit="VariantGMM") 
    if args.lam == 0:
        savefolder = "RESULTS_FOLDER/deepSVG/"
    else:
        savefolder = f"RESULTS_FOLDER/deepSVG_reg_{args.lam}/"
        reg_loss = get_loss(crit="mu_data") 
else:
    print("Wrong type specified")

if not os.path.exists(os.path.join(savefolder,"predictions")):
    os.makedirs(os.path.join(savefolder,"predictions"))
    os.makedirs(os.path.join(savefolder,"nets"))


json.dump(config, open(os.path.join(savefolder,"nets","config.json"), "w"))


device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mu_data = torch.from_numpy(np.load("DATA/mu_data.npy").astype("float32")).to(device)

patients = args.patients
for patient in patients:
    files = [os.path.join("DATA/preprocessed/myops_2d",f) for f in os.listdir("DATA/preprocessed/myops_2d/") if int(f.split("_")[-2])==patient]
    for file in files:
        print(file.split("\\")[-1][0:-4])
        slicenr = file[-5]
        net = UNet2D(**config["network"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), weight_decay=1e-3,lr=1e-4)
        nll = []
        X, gt, mask_heart = prepare_data(file)
        X = torch.from_numpy(X.astype("float32")).to(device)
        X = X[None,...]
        gt = torch.from_numpy(gt.astype("float32")).to(device)
        mask_heart = torch.from_numpy(mask_heart.astype("float32")).to(device)
        mask_heart = mask_heart[None, None, ...]
        
        #train
        nll.append(float('inf'))
        for epoch in range(args.max_epochs):
            net.train()
            opt.zero_grad()
            loss=0
            out=net(X)[0]
            loss += main_loss(out, X, mask_heart.float())
            nll.append(loss.item())
            if args.lam != 0:
                loss+= args.lam * reg_loss(out, X, mask_heart.float(),mu_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            opt.step()
            out=out*mask_heart
            out =torch.cat((1-mask_heart,out),1)
            change = (nll[-1]-nll[-2])
            if (epoch > args.min_epochs  and  abs(change) < args.tol):    
                break
            
        
        save_checkpoint(net, os.path.join(savefolder, "nets"), 0,  name = f"weights_Patient_{patient}_{slicenr}")
        pred = out[0,...].cpu().detach().numpy()
        plt.figure()
        plt.imshow(np.argmax(pred,0))
        #plt.show()
        

    
