# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:08:40 2024

@author: A0067501
"""

import numpy as np
import os
import pandas as pd
import json

FOLDER = "RESULTS_FOLDER/RESULTS/normal_GMM/"

methods = os.listdir(FOLDER)
results ={}
results["net"]=[]
results["em"]=[]
for method in methods:
    with open(FOLDER + method + "/results_dice.txt") as f: 
        data = f.read() 
    dice = json.loads(data)
    net = dice["Network"]
    net_means=[]
    em = dice["EM"]
    em_means =[]
    for patient in net.keys():
        net_means.append(np.array(net[patient]).mean())
        em_means.append(np.array(em[patient]).mean())
    results["net"].append(np.mean(net_means))
    results["em"].append(np.mean(em_means))
    
print(results["net"])
print(results["em"])

FOLDER = "RESULTS_FOLDER/RESULTS/spatially_variant_GMM/"

methods = os.listdir(FOLDER)
results ={}
results["net"]=[]
results["em"]=[]
for method in methods:
    with open(FOLDER + method + "/results_dice.txt") as f: 
        data = f.read() 
    dice = json.loads(data)
    net = dice["Network"]
    net_means=[]
    em = dice["EM"]
    em_means =[]
    for patient in net.keys():
        net_means.append(np.array(net[patient]).mean())
        em_means.append(np.array(em[patient]).mean())
    results["net"].append(np.mean(net_means))
    results["em"].append(np.mean(em_means))
    
print(results["net"])
print(results["em"])