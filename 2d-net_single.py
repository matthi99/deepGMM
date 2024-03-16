# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:10:28 2024

@author: A0067501
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:36:36 2024

@author: A0067501
"""

import torch
from config import Config
from utils.architectures import get_network
from utils.utils_training import Histogram, plot_examples, save_checkpoint, get_logger, normalize, prepare_data, plot_single
from dataloader import CardioDataset, CardioCollatorMulticlass
from utils.metrics import get_metric
from utils.losses import get_loss

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import logging


parser = argparse.ArgumentParser(description='Define hyperparameters for training.')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchnorm', type=bool, default=True)
parser.add_argument('--start_filters', type=int, default=32)
parser.add_argument('--out_channels', type=int, default=4)
parser.add_argument('--activation', type=str, default="leakyrelu")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--datafolder', help= "Path to 2d data folder", type=str, 
                    default="DATA/preprocessed/traindata2d/")
parser.add_argument('--savepath', help= "Path were resuts should get saved", type=str, 
                    default="RESULTS_FOLDER/")
parser.add_argument('--type',type = str, help = "Type of Gaussian mixture model (normal,variant) ",  default = "normal")
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
    "best_metric":-float('inf')
}


path= args.savepath
reg_loss = get_loss(crit="mu_sigma") 
if args.type == "normal":
    main_loss = get_loss(crit="NormalGMM") 
    if args.lam == 0:
        savefolder = f"normal_GMM/single/"
    else:
        savefolder = f"normal_GMM/single_reg_{args.lam}/"
        
    
elif args.type == "variant":
    main_loss = get_loss(crit="VariantGMM") 
    if args.lam == 0:
        savefolder = f"spatially_variant_GMM/single/"
    else:
        savefolder = f"spatially_variant_GMM/single_reg_{args.lam}/"
else:
    print("Wrong type specified")

if not os.path.exists(os.path.join(path,savefolder,"plots")):
    os.makedirs(os.path.join(path,savefolder,"plots"))

logger= get_logger(savefolder)
FileOutputHandler = logging.FileHandler(path+savefolder+"logs.log")
logger.addHandler(FileOutputHandler)


    
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on {device}")

# #get network, optimizer, loss, metric and histogramm    
# net=get_network(architecture="unet2d", **config["network"]).to(device)
# # opt = torch.optim.SGD(net.parameters(), 1e-3, weight_decay=1e-3,
# #                                             momentum=0.99, nesterov=True)   
# opt = torch.optim.AdamW(net.parameters(), weight_decay=1e-3,lr=1e-4)
# lambda1 = lambda epoch: (1-epoch/args.epochs)**0.9
# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)



metrics={metric: get_metric(metric=metric) for metric in config['metrics']} 
classes=config["classes"]



patients = Config.cross_validation[f"fold_0"]['val']
for patient in patients:
    files = [os.path.join(args.datafolder,f) for f in os.listdir(args.datafolder) if int(f.split("_")[-2])==patient]
    if not os.path.exists(os.path.join(path,savefolder,f"Patient_{patient}")):
        os.makedirs(os.path.join(path,savefolder,f"Patient_{patient}"))
    
    for file in files:
        slicenr = file[-5]
        net=get_network(architecture="unet2d", **config["network"]).to(device)
        # opt = torch.optim.SGD(net.parameters(), 1e-3, weight_decay=1e-3,
        #                                             momentum=0.99, nesterov=True)   
        opt = torch.optim.AdamW(net.parameters(), weight_decay=1e-3,lr=1e-4)
        histogram = []
        X, gt, mask_heart = prepare_data(file)
        X = torch.from_numpy(X.astype("float32")).to(device)
        X = X[None,...]
        gt = torch.from_numpy(gt.astype("float32")).to(device)
        mask_heart = torch.from_numpy(mask_heart.astype("float32")).to(device)
        mask_heart = mask_heart[None, None, ...]
        
        #train
        best_metric=-float('inf')
        train_loss = float('inf')
        for epoch in range(args.epochs):
            net.train()
            logger.info(f"Epoch {epoch}\{args.epochs}-------------------------------")
            prev_train_loss = train_loss
            opt.zero_grad()
            loss=0
            out=net(X)[0]
            loss += main_loss(out, X, mask_heart.float())
            train_loss =  main_loss(out, X, mask_heart.float())
            loss+= args.lam *reg_loss(out, X, mask_heart.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            opt.step()
            histogram.append(loss.item())
            out=out*mask_heart
            out =torch.cat((1-mask_heart,out),1)
            change = (prev_train_loss-train_loss).item()
            #scheduler.step()
            print(change)
            if (epoch > 10  and  abs(change) < args.tol):
                save_checkpoint(net, os.path.join(path, savefolder, f"Patient_{patient}"), 0,  f"weights_{slicenr}")
                json.dump(config, open(os.path.join(path,savefolder,f"Patient_{patient}","config.json"), "w"))
                break
        
        plot_single(X, out, gt, epoch, os.path.join(path,savefolder,f"Patient_{patient}",f"final_{slicenr}.png"))
        plt.figure()
        plt.plot(histogram)
        plt.savefig(os.path.join(path,savefolder,f"Patient_{patient}",f"histogram_{slicenr}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        #plt.show()
        plt.close()
        
            







    
# likely_loss = get_loss(crit="likely") 
# dice_loss = get_loss(crit= "dice")
# mu_sigma = get_loss(crit= "mu_sigma")
# probs= get_loss(crit= 'probs')
# metrics={metric: get_metric(metric=metric) for metric in config['metrics']} 
# classes=config["classes"]
# losses= ["likely", "mu", "dice"]
# histogram=Histogram(classes, metrics, losses)


# images = 
    


    
    
#     #evaluate (we are evaluating on a per patient level)
#     net.eval()
#     steps = 0
#     for im,mask in dataloader_eval:
#         with torch.no_grad():
#             gt= torch.cat([mask[key].float() for key in mask.keys()],1)
#             out=net(im)[0]
#             out[:,0:1,:,:][mask["bg"].float()==1]=1
#             for i in range(1,5):
#                 out[:,i:i+1,:,:][mask["bg"].float()==1]=0
#             histogram.add_val_metrics(out,gt)
#             steps += 1
    
#     histogram.scale_val(steps)
#     histogram.plot_hist(os.path.join(args.savepath,savefolder))
#     if epoch%3==0:
#             plot_examples(im,out,epoch,os.path.join(path,savefolder,"plots"), train=False)
    
    
#     #ceck for improvement and save best model
#     val_metric=0
#     for cl in ["blood","muscle", "edema",  "scar"]:
#         val_metric+=histogram.hist[config['metrics'][0]][f"val_{cl}"][-1]
#     val_metric=val_metric/2
#     if epoch%25==0:
#         best_metric=val_metric
#         config["best_metric"]=best_metric
#         logger.info(f"New best Metric {best_metric}")
#         histogram.print_hist(logger)
#         save_checkpoint(net, os.path.join(path, savefolder), args.fold, f"weights_{epoch}",  savepath=True)
#         json.dump(config, open(os.path.join(path,savefolder,"config.json"), "w"))
    
        
#     logger.info(scheduler.get_last_lr())
#     scheduler.step()
    
        
# np.save(os.path.join(path, savefolder, "histogram.npy"),histogram.hist)
# save_checkpoint(net, os.path.join(path, savefolder), "last_weights")
# json.dump(config, open(os.path.join(path, savefolder, "config-last_weights.json"), "w"))
# logger.info("Training Finished!") 
    
