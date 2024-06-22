# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:15:31 2024

@author: A0067501
"""

import torch
import logging
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.config import Config
from utils.unet import UNet2D
from utils.dataloader import CardioDataset, CardioCollatorMulticlass
from utils.utils import save_checkpoint, get_logger
from utils.losses import get_loss, DiceMetric


#Define parameters for training 
parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--min_epochs', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--type',type = str, help = "Type of Gaussian mixture model (deepG, deepSVG) ",  default = "deepG")
parser.add_argument('--lam',type = float, help = "Regularization parameter",  default = 1)
parser.add_argument('--tol',type = float, help = "Tolerance for stopping criteria",  default = 0.005)
args = parser.parse_args()

#CNN configurations
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

#create RESULTS_FOLDER and specify losses
reg_loss = get_loss(crit="mu_data") 

if args.type == "deepG":
    main_loss = get_loss(crit="NormalGMM") 
    savefolder = f"RESULTS_FOLDER/deepG/multiple_images/lam={args.lam}/"
elif args.type == "deepSVG":
    main_loss = get_loss(crit="VariantGMM") 
    savefolder = f"RESULTS_FOLDER/deepSVG/multiple_images/lam={args.lam}/"
else:
    print("Wrong type specified")

if not os.path.exists(savefolder):
    os.makedirs(os.path.join(savefolder,"predictions"))
    os.makedirs(os.path.join(savefolder,"plots"))


#define logger and training setup
logger= get_logger(savefolder)
FileOutputHandler = logging.FileHandler(savefolder+"logs.log")
logger.addHandler(FileOutputHandler)
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on {device}")
mu_data = torch.from_numpy(np.load("DATA/mu_data.npy").astype("float32")).to(device)

setup_train=Config.train_data_setup
setup_val=Config.val_data_setup

setup_train['patients']=Config.train_patients
setup_val['patients']=Config.val_patients

#get dataloader  
cd_train = CardioDataset(folder="DATA/preprocessed/myops_2d/" ,z_dim=False,  **setup_train)
collator = CardioCollatorMulticlass(classes=("bg", "blood","muscle", "edema", "scar"))
dataloader_train = torch.utils.data.DataLoader(cd_train, batch_size=args.batchsize, collate_fn=collator,
                                                        shuffle=True)    

#get network, optimizer, loss, metric and histogramm    
net = UNet2D(**config["network"]).to(device)                                        
opt = torch.optim.AdamW(net.parameters(), weight_decay=1e-3,lr=1e-3)
lambda1 = lambda epoch: (1-epoch/args.max_epochs)**0.9
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    
histogramm ={}
histogramm["NLL"]=[]
histogramm["reg_loss"]=[]
histogramm["dice"]=[]
    
#train
for epoch in range(args.max_epochs):
    net.train()
    logger.info(f"Epoch {epoch}\{args.max_epochs}-------------------------------")
    steps = 0
    for key in histogramm.keys():
        histogramm[key].append(0)
    for im,mask  in dataloader_train:
        opt.zero_grad()
        loss=0
        gt= torch.cat([mask[key].float() for key in mask.keys()],1)
        out=net(im)[0]
        loss += main_loss(out, im, (1-mask["bg"]).float())
        histogramm["NLL"][-1]+=loss.item()
        if args.lam != 0:
            loss+= args.lam *reg_loss(out, im, (1-mask["bg"]).float(), mu_data)
            histogramm["reg_loss"][-1] += reg_loss(out, im, (1-mask["bg"]).float(), mu_data).item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        opt.step()
        out=out*(1-mask["bg"]).float()
        out =torch.cat((mask["bg"],out),1)
        histogramm["dice"][-1]+=DiceMetric()(out, gt).cpu().detach().numpy()
        steps += 1
    
    for key in histogramm.keys():
        histogramm[key][-1] /= steps
    #save if stopping criteria is satisfied
    if epoch > args.min_epochs:
        change = (histogramm["NLL"][-2]-histogramm["NLL"][-1])
        if abs(change) < args.tol:
            save_checkpoint(net, savefolder)
            json.dump(config, open(os.path.join(savefolder,"config.json"), "w"))
            break
    scheduler.step()
    
    
#save training progress
plt.figure()
plt.subplot(1,3,1)
plt.plot(histogramm["NLL"])
plt.title("NLL", fontsize = 11)
plt.subplot(1,3,2)
plt.plot(histogramm["reg_loss"])
plt.title("Reg. loss", fontsize = 11)
plt.subplot(1,3,3)
plt.plot(histogramm["dice"], label =["blood","muscle", "edema", "scar"])
plt.legend(loc='best', fontsize=8)
plt.title("Dice", fontsize = 11)
plt.savefig(os.path.join(savefolder, "train_progress.png"), bbox_inches='tight', dpi=500)
plt.show()

logger.info(f"Training finished after {epoch} epochs. Network is saved in {savefolder}.")

