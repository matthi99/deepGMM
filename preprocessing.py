# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:27:15 2022

@author: Schwab Matthias
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data_folder = "DATA"

#3d data
train_folder= os.path.join(data_folder, 'MyoPS 2020 Dataset/')
preprocessed_folder= os.path.join(data_folder, 'preprocessed')
if not os.path.exists(preprocessed_folder):
    os.makedirs(preprocessed_folder)
    
save_folder= os.path.join(preprocessed_folder, 'traindata3d/')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
        
train_gt =os.path.join(train_folder, 'train25_myops_gd/')
train_img =os.path.join(train_folder, 'train25/')
    
files_gt=[train_gt+f for f in os.listdir(train_gt)]
files_LGE=[train_img+f for f in os.listdir(train_img) if f.endswith("DE.nii.gz")]
files_T2=[train_img+f for f in os.listdir(train_img) if f.endswith("T2.nii.gz")]
files_C0=[train_img+f for f in os.listdir(train_img) if f.endswith("C0.nii.gz")]
    
for gt, im_LGE, im_T2, im_C0 in zip(files_gt, files_LGE, files_T2, files_C0):
    data={}
    #load images
    #LGE
    LGE  = nib.load(im_LGE).get_fdata()
    data['center']=(LGE.shape[0]//2, LGE.shape[1]//2)
    LGE=np.transpose(LGE,(2,0,1))
    data['LGE']=LGE
    #T2
    T2  = nib.load(im_T2).get_fdata()
    T2=np.transpose(T2,(2,0,1))
    data['T2']=T2
    #C0
    C0  = nib.load(im_C0).get_fdata()
    C0=np.transpose(C0,(2,0,1))
    data['C0']=C0

    #create masks
    cont  = nib.load(gt).get_fdata()
    cont = np.transpose(cont,(2,0,1))
    patient_nr=im_LGE.split("_")[-2]
        
    temp=np.copy(cont)
    bg=np.zeros(temp.shape)
    bg[temp==0]=1
    bg[temp==600]=1
    blood=np.zeros(temp.shape)
    blood[temp==500]=1
    muscle=np.zeros(temp.shape)
    muscle[temp==200]=1
    edema = np.zeros(temp.shape)
    edema[temp==1220]=1
    scar=np.zeros(temp.shape)
    scar[temp==2221]=1
        
    masks=np.concatenate((bg[...,None], blood[...,None], muscle[...,None], edema[...,None], scar[...,None]), axis=3)
    data['masks']=masks
    
    #save
    np.save(os.path.join(save_folder, "Case_"+patient_nr+'.npy'), data)
    
    
#2d data
save_folder2d = os.path.join(preprocessed_folder, 'traindata2d/')
if not os.path.exists(save_folder2d):
    os.makedirs(save_folder2d)
        
train_patients= os.listdir(save_folder)
slices=[]
for patient in train_patients:
    data=np.load(save_folder+patient, allow_pickle=True).item()
    L=data['LGE'].shape[0]
    slices.append(L)
    for i in range(L):
        data2d={}
        data2d['center']=data['center']
        data2d['LGE']=data['LGE'][i,...]
        data2d['T2']=data['T2'][i,...]
        data2d['C0']=data['C0'][i,...]
        data2d['masks']=data['masks'][i,...]
        np.save(os.path.join(save_folder2d, patient[:-4]+'_'+str(i)+'.npy'), data2d)
            
    
print("Data prepared!")


