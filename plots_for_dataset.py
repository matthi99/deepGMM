# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:56:33 2024

@author: matth
"""

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from utils.architectures import get_network
from utils.utils_test import prepare_data, normalize, variant_log_likelyhood, normal_log_likelyhood, variant_log_likelyhood, load_2dnet, order_dice, dicecoeff, load_2dnet_single
from utils.spatially_variant_gmm import VariantGMM
import yaml
import json
import pandas as pd
import matplotlib.patches as mpatches


cmap = matplotlib.cm.get_cmap("jet").copy()
cmap.set_bad(color='black')

FOLDER= "DATA/preprocessed/traindata2d/"
VAL_PATIENTS = [i for i in range(101,126)]

patients = [f"Case_{format(num, '03d')}" for num in VAL_PATIENTS]
files=[]
for patient in patients:
    files.append([os.path.join(FOLDER,f) for f in os.listdir(FOLDER) if f.startswith(patient)])
    
files= sum(files, [])

savefolder = "RESULTS_FOLDER/RESULTS/plots_for_paper/dataset/"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

classes= ["blood", "muscle", "edema", "scar"]
    
for file in files:
    slicenr =file.split(".")[0][-1]
    patientnr= file[-9:-6]
    X, gt, _ = prepare_data(file)
    gt=np.ma.masked_where(gt ==0, gt)
    bssfp= np.zeros_like(gt)
    bssfp[gt==1]=1
    bssfp[gt==2]=2
    bssfp[gt==3]=2
    bssfp[gt==4]=2
    bssfp=np.ma.masked_where(bssfp ==0, bssfp)
    T2 = np.zeros_like(gt)
    T2[gt==3]=3
    T2[gt==4]=3
    T2=np.ma.masked_where(T2 ==0, T2)
    LGE = np.zeros_like(gt)
    LGE[gt == 4]=4
    LGE=np.ma.masked_where(LGE ==0, LGE)
    plt.figure()
    plt.subplots_adjust(left  = 0.125,  
                        right = 0.9,   
                        bottom = 0,  
                        top = 0.75,      
                        wspace = 0.1,   
                        hspace = 0 )
    plt.subplot(2,3,1)
    plt.imshow(X[2,...], cmap = "gray")
    plt.axis("off")
    plt.title("bSSFP", fontsize = 11)
    plt.subplot(2,3,2)
    plt.imshow(X[1,...], cmap = "gray")
    plt.axis("off")
    plt.title("T2")
    plt.subplot(2,3,3)
    plt.imshow(X[0,...], cmap = "gray")
    plt.axis("off")
    plt.title("LGE",fontsize = 11)
    
    
    plt.subplot(2,3,4)
    plt.imshow(X[2,...], interpolation="none",  cmap = "gray")
    im=plt.imshow(bssfp, interpolation="none", vmin=0, vmax=4, alpha = 0.5, cmap ="jet")
    values = [1,2]
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.47, 1.03), loc=2,fontsize=7 )
    
    plt.axis("off")
    
    plt.subplot(2,3,5)
    plt.imshow(X[1,...], interpolation="none",  cmap = "gray")
    plt.imshow(T2, interpolation="none", vmin=0, vmax=4, alpha = 0.5, cmap ="jet")
    values = [3]
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="edema".format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.48, 1.03), loc=2,fontsize=7 )
    plt.axis("off")
    
    plt.subplot(2,3,6)
    plt.imshow(X[0,...], interpolation="none",  cmap = "gray")
    plt.imshow(LGE, interpolation="none", vmin=0, vmax=4, alpha = 0.5, cmap ="jet")
    values = [4]
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="scar".format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.57, 1.03), loc=2,fontsize=7 )
    plt.axis("off")
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}.png"), bbox_inches='tight', dpi=500)
    plt.show()
    
    
    plt.figure()
    im = plt.imshow(gt, interpolation="none", vmin=0, vmax=4, cmap =cmap)
    values = [1,2,3,4]
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.67, 1.015), loc= 2,fontsize=9)
    plt.axis("off")
    plt.savefig(os.path.join(savefolder+f"Case_{patientnr}_{file[-5:-4]}_gt.png"), bbox_inches='tight', dpi=500)
    plt.show()

