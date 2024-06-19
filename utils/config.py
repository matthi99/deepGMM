# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:43:42 2022

@author: A0067501
"""

import yaml
import numpy as np


class Config:
    all_patients = [i for i in range(101,126)]
    val_patients = [109,114,117,120,122]
    train_patients = []
    for p in all_patients:
        if p not in val_patients:
            train_patients.append(p)
    
    
    train_data_setup = {
        "spatialtransform":{
            "do_elastic_deform":True, 
            "alpha":(0.,175.),
            },
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 80,
            "height": 80,
            "translation": 0
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
            "width": 80,
            "height": 80,
            "translation": 0
            },
    }  
    
    val_data_setup_heart = {
        "normalize": {
            "mode": "mean"
            },
        "ROI": {
            "width": 160,
            "height": 160,
            "translation": 0
            },
    }  





