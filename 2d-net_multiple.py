# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:36:36 2024

@author: A0067501
"""

import torch
from config import Config
from utils.architectures import get_network
from utils.utils_training import Histogram, plot_examples, save_checkpoint, get_logger
from dataloader import CardioDataset, CardioCollatorMulticlass
from utils.metrics import get_metric
from utils.losses import get_loss
import logging

import os
import argparse
import json
import numpy as np

#test

#Define parameters for training 
parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--batchnorm', type=bool, default=True)
parser.add_argument('--start_filters', type=int, default=32)
parser.add_argument('--out_channels', type=int, default=4)
parser.add_argument('--activation', type=str, default="leakyrelu")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--datafolder', help= "Path to 2d data folder", type=str, 
                    default="DATA/preprocessed/traindata2d/")
parser.add_argument('--savepath', help= "Path were resuts should get saved", type=str, 
                    default="RESULTS_FOLDER/")
parser.add_argument('--type',type = str, help = "Type of Gaussian mixture model (normal,variant) ",  default = "variant")
parser.add_argument('--lam',type = float, help = "Regularization parameter",  default = 0)
parser.add_argument('--tol',type = float, help = "Tolerance for stopping criteria",  default = 0.005)
args = parser.parse_args()

config = {
        "metrics":["dice"],
    "network":{
        "activation": args.activation,
        "dropout": args.dropout,
        "batchnorm": args.batchnorm,
        "start_filters": args.start_filters,
        "in_channels":3,
        "L":4,
        "out_channels": args.out_channels,
        "residual": False, 
        "last_activation":"softmax"},
    
    "classes": ["blood","muscle", "edema", "scar"],
    "best_metric":-float('inf'),
    "fold":args.fold
}

#create save folders if they dont exist already
path=args.savepath
reg_loss = get_loss(crit="mu_sigma") 
if args.type == "normal":
    main_loss = get_loss(crit="NormalGMM") 
    if args.lam == 0:
        savefolder = f"normal_GMM/multiple/"
    else:
        savefolder = f"normal_GMM/multiple_reg_{args.lam}/"
        
    
elif args.type == "variant":
    main_loss = get_loss(crit="VariantGMM") 
    if args.lam == 0:
        savefolder = f"spatially_variant_GMM/multiple/"
    else:
        savefolder = f"spatially_variant_GMM/multiple_reg_{args.lam}/"
else:
    print("Wrong type specified")
    

if not os.path.exists(os.path.join(path,savefolder, "plots")):
    os.makedirs(os.path.join(path,savefolder, "plots"))


#define logger and get network configs
logger= get_logger(savefolder)
FileOutputHandler = logging.FileHandler(path+savefolder+"logs.log")
logger.addHandler(FileOutputHandler)
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on {device}")

setup_train=Config.train_data_setup_no_aug
setup_val=Config.val_data_setup
if args.fold<5:
    logger.info(f"Training on fold {args.fold} of nnunet data split")
    setup_train['patients']=Config.cross_validation[f"fold_{args.fold}"]['train']
    setup_val['patients']=Config.cross_validation[f"fold_{args.fold}"]['val']
else:
    logger.info("using own data split")


#get dataloaders     
cd_train = CardioDataset(folder=args.datafolder ,z_dim=False,  **setup_train)
#for evaluation we read in the whole stack per patient so it has to be batch_size=1 
cd_val = CardioDataset(folder=args.datafolder ,z_dim=False, validation=True, **setup_val)    
collator = CardioCollatorMulticlass(classes=("bg", "blood","muscle", "edema", "scar"))
dataloader_train = torch.utils.data.DataLoader(cd_train, batch_size=args.batchsize, collate_fn=collator,
                                                        shuffle=True)    
dataloader_eval = torch.utils.data.DataLoader(cd_val, batch_size=args.batchsize, collate_fn=collator,
                                                        shuffle=False)


#get network, optimizer, loss, metric and histogramm    
net=get_network(architecture="unet2d", **config["network"]).to(device)
# opt = torch.optim.SGD(net.parameters(), 5*1e-3, weight_decay=1e-3,
#                                           momentum=0.99, nesterov=True)   
opt = torch.optim.AdamW(net.parameters(), weight_decay=1e-3,lr=1e-3)
lambda1 = lambda epoch: (1-epoch/args.epochs)**0.9
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    
# likely_loss = get_loss(crit="likely") 
# dice_loss = get_loss(crit= "dice")
# mu_sigma = get_loss(crit= "mu_sigma")
# probs= get_loss(crit= 'probs')
metrics={metric: get_metric(metric=metric) for metric in config['metrics']} 
classes=config["classes"]
losses= ["likely", "mu", "dice"]
histogram=Histogram(classes, metrics, losses)
val_loss=0
    

#train
best_metric=-float('inf')
train_loss = float('inf')
c=0
for epoch in range(args.epochs):
    net.train()
    logger.info(f"Epoch {epoch}\{args.epochs}-------------------------------")
    prev_train_loss = train_loss
    train_loss = 0
    steps = 0
    histogram.append_hist()
    for im,mask  in dataloader_train:
        opt.zero_grad()
        loss=0
        gt= torch.cat([mask[key].float() for key in mask.keys()],1)
        out=net(im)[0]
        loss += main_loss(out, im, (1-mask["bg"]).float())
        loss+= args.lam *reg_loss(out, im, (1-mask["bg"]).float())
        loss.backward()
        train_loss += main_loss(out, im, (1-mask["bg"]).float())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        opt.step()
        histogram.add_loss("likely", loss)
        # histogram.add_loss("mu", mu)
        # histogram.add_loss("dice", dice)
        out=out*(1-mask["bg"]).float()
        out =torch.cat((mask["bg"],out),1)
        histogram.add_train_metrics(out,gt)
        steps += 1
    
    train_loss = train_loss/steps
    change = (prev_train_loss-train_loss).item()
    histogram.scale_train(steps)
    if epoch%3==0:
        plot_examples(im,out,epoch,os.path.join(path,savefolder,"plots"), train=True)
    
    
    #evaluate (we are evaluating on a per patient level)
    net.eval()
    
    
    steps = 0
    for im,mask in dataloader_eval:
        with torch.no_grad():
            gt= torch.cat([mask[key].float() for key in mask.keys()],1)
            out=net(im)[0]
            val_loss += main_loss(out, im, (1-mask["bg"]).float())
            out=out*(1-mask["bg"]).float()
            out =torch.cat((mask["bg"],out),1)
            histogram.add_val_metrics(out,gt)
            steps += 1
    
    
    histogram.scale_val(steps)
    histogram.plot_hist(os.path.join(args.savepath,savefolder))
    if epoch%3==0:
            plot_examples(im,out,epoch,os.path.join(path,savefolder,"plots"), train=False)
    
    
    #ceck for improvement and save best model
    val_metric=0
    for cl in ["blood","muscle", "edema",  "scar"]:
        val_metric+=histogram.hist[config['metrics'][0]][f"val_{cl}"][-1]
    val_metric=val_metric/2
    if epoch%25==0:
        best_metric=val_metric
        config["best_metric"]=best_metric
        logger.info(f"New best Metric {best_metric}")
        histogram.print_hist(logger)
        save_checkpoint(net, os.path.join(path, savefolder), args.fold, f"weights_{epoch}",  savepath=True)
        json.dump(config, open(os.path.join(path,savefolder,"config.json"), "w"))
    
    print(change)
    if abs(change) < args.tol:
        c=c+1
    else:
        c=0
    if c >= 1:
        histogram.print_hist(logger)
        save_checkpoint(net, os.path.join(path, savefolder), args.fold, f"weights",  savepath=True)
        json.dump(config, open(os.path.join(path,savefolder,"config.json"), "w"))
        break
        

    logger.info(scheduler.get_last_lr())
    scheduler.step()
    
        
np.save(os.path.join(path, savefolder, "histogram.npy"),histogram.hist)
logger.info("Training Finished!") 
    
