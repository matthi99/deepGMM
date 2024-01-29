# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:43:42 2022

@author: A0067501
"""

import yaml
import numpy as np


class Config:
    all_patients = [i for i in range(101,126)]
    train_patients = all_patients
    
    cross_validation={} #same cross validationa as nnunet
    for i in range(5):
        cross_validation['fold_'+str(i)]={}
    cross_validation['fold_0']['val']=[109,114,117,120,122]
    cross_validation['fold_1']['val']=[104,111,113,116,118]
    cross_validation['fold_2']['val']=[101,107,108,112,119]
    cross_validation['fold_3']['val']=[115,121,123,124,125]
    cross_validation['fold_4']['val']=[102,103,105,106,110]
    
    for i in range(5):
        cross_validation['fold_'+str(i)]['train']=[]
        for p in all_patients:
            if p not in cross_validation['fold_'+str(i)]['val']:
                cross_validation['fold_'+str(i)]['train'].append(p)
    
    
    train_data_setup = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,175.),
            },
        
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 192,
            "height": 192,
            "translation": 4
            },
        "gamma": {
            "retain_stats":True, 
            "gamma_range": (0.65, 1.55), 
            "p_per_sample": 0.15
            },
        "lowres": {
            "scale_factor": (0.45, 1),
            "p_per_sample": 0.15,
            },
        "contrast": {
            "contrast_range": (0.6, 1.55), 
            "p_per_sample": 0.15
            }, 
        "brightness":{},
        "gaussianblur":{
            "blur_sigma": (0.5, 1.5), 
            "p_per_sample": 0.1
            }, 
        "gaussian":{
            "sigma":0.1, 
            "p_gaussian":0.15},
        "flip":True, 
        }
    
    
    val_data_setup = {
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 192,
            "height": 192,
            "translation": 0
            },
    }  





