# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:27:15 2022

@author: Schwab Matthias
"""
import nibabel as nib
import numpy as np
import os
from scipy import ndimage



#folders and filelists
data_folder = "DATA"
train_folder= os.path.join(data_folder, 'MyoPS 2020 Dataset/')
preprocessed_folder= os.path.join(data_folder, 'preprocessed')
save_folder= os.path.join(preprocessed_folder, 'myops_2d/')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
        
train_gt =os.path.join(train_folder, 'train25_myops_gd/')
train_img =os.path.join(train_folder, 'train25/')
    
files_gt=[train_gt+f for f in os.listdir(train_gt)]
files_LGE=[train_img+f for f in os.listdir(train_img) if f.endswith("DE.nii.gz")]
files_T2=[train_img+f for f in os.listdir(train_img) if f.endswith("T2.nii.gz")]
files_C0=[train_img+f for f in os.listdir(train_img) if f.endswith("C0.nii.gz")]

#prepare data    
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
    
    middle= bg[bg.shape[0]//2,...]
    middle= 1-middle
    center=ndimage.center_of_mass(middle)
    data['center']=(int(center[0]), int(center[1]))

    masks=np.concatenate((bg[...,None], blood[...,None], muscle[...,None], edema[...,None], scar[...,None]), axis=3)
    data['masks']=masks
    
    #save images slice py slice
    L=data['LGE'].shape[0]
    for i in range(L):
        data2d={}
        data2d['center']=data['center']
        data2d['LGE']=data['LGE'][i,...]
        data2d['T2']=data['T2'][i,...]
        data2d['C0']=data['C0'][i,...]
        data2d['masks']=data['masks'][i,...]
        temp = np.argmax(data['masks'][i,...],-1)
        #only save images with were all classes occur
        if (len(np.unique(temp)))==5:
            np.save(os.path.join(save_folder, "Case_"+patient_nr+'_'+str(i)+'.npy'), data2d)
                      

# Get Mu_data for regularization
folder = "DATA/preprocessed/myops_2d/"
files= os.listdir(folder)

#choose 10 random images
np.random.seed(42)
np.random.shuffle(files)
files = files[0:10]

modalities = ["LGE", "T2", "C0"]
classes = ["blood", "muscle", "edema", "scar"]

means={}
probs={}
stds={}
#calculate means, covariances and wheights
for modality in modalities:
    means[modality]={}
    stds[modality]={}
    for cl in classes:
        means[modality][cl]=[]
        stds[modality][cl]=[]
        probs[cl]=[]
    for file, i in zip(files, range(len(files))):
        z= np.load(os.path.join(folder, file), allow_pickle=True).item()
        center= z["center"]
        img= z[modality][center[0]-80:center[0]+80, center[1]-80: center[1]+80]
        img=(img - np.mean(img)) / np.std(img)
        mask= np.moveaxis(z["masks"], -1, 0)
        mask= mask[:,center[0]-80:center[0]+80, center[1]-80: center[1]+80]
       
        blood= img[mask[1,...]==1]
        muscle= img[mask[2,...]==1]
        edema= img[mask[3,...]==1]
        scar= img[mask[4,...]==1]
        heart= img[mask[0,...]==0]
        N = len(heart)
        
        means[modality]["blood"].append(blood.mean())
        means[modality]["muscle"].append(muscle.mean())
        means[modality]["edema"].append(edema.mean())
        means[modality]["scar"].append(scar.mean())
        
        
        stds[modality]["blood"].append(blood.std())
        stds[modality]["muscle"].append(muscle.std())
        stds[modality]["edema"].append(edema.std())
        stds[modality]["scar"].append(scar.std())
        
        probs["blood"].append(len(blood)/N)
        probs["muscle"].append(len(muscle)/N)
        probs["edema"].append(len(edema)/N)
        probs["scar"].append(len(scar)/N)

mu=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        mu[i,j]=np.mean(means[modality][cl])    
np.save(os.path.join(data_folder, 'mu_data.npy'), mu)

sigma=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        sigma[i,j]=np.mean(stds[modality][cl])    
np.save(os.path.join(data_folder, 'sigma_data.npy'), sigma)

pi=np.zeros(4)
for cl,i in zip(classes, range(4)):
    pi[i]=np.mean(probs[cl])    
np.save(os.path.join(data_folder, 'pi_data.npy'), pi)

print("Data prepared!")


