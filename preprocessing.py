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


#3d data
train_folder='DATA/emidec-dataset-1.0.1/'
save_folder='DATA/traindata/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


train_patients=[f for f in os.listdir(train_folder) if os.path.isdir(train_folder+f)]


shapes={}
shapes['masks']=[]
shapes['img']=[]


for patient in train_patients:
    data={}
    img  = nib.load(train_folder+patient+'/images/'+patient+'.nii.gz').get_fdata()
    data['center']=(img.shape[0]//2, img.shape[1]//2)
    img=np.transpose(img,(2,0,1))
    
    shapes['img'].append(img.shape)

    
    cont  = nib.load(train_folder+patient+'/contours/'+patient+'.nii.gz').get_fdata()
    cont = np.transpose(cont,(2,0,1))
    
    
    temp=np.copy(cont)
    heart=np.zeros(temp.shape)
    heart[temp==0]=1
    blood=np.zeros(temp.shape)
    blood[temp==1]=1
    muscle=np.zeros(temp.shape)
    muscle[temp==2]=1
    scar=np.zeros(temp.shape)
    scar[temp==3]=1
    mvo=np.zeros(temp.shape)
    mvo[temp==4]=1

    masks=np.concatenate((heart[...,None],blood[...,None], muscle[...,None], scar[...,None], mvo[...,None]), axis=3)
    
    # noise= np.random.normal(0,np.std(img)/10,img.shape)
    # img[heart==1]=noise[heart==1]
    
    data['img']=img
    data['masks']=masks
    shapes['masks'].append(masks.shape)
    
    np.save(save_folder+patient[0:5]+patient[-3:]+'.npy', data)
    

#2D data
save_folder2d='DATA/traindata2d/'
if not os.path.exists(save_folder2d):
    os.makedirs(save_folder2d)
    

for patient in train_patients:
    data=np.load(save_folder+patient[0:5]+patient[-3:]+'.npy', allow_pickle=True).item()
    L=data['img'].shape[0]
    for i in range(L):
        data2d={}
        data2d['center']=data['center']
        data2d['img']=data['img'][i,...]
        data2d['masks']=data['masks'][i,...]
        np.save(save_folder2d+patient[0:5]+patient[-3:]+'_'+str(i)+'.npy', data2d)
        
print("Data prepared!")