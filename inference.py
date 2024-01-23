# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:03:27 2023

@author: matthias
"""

import torch
import os
from utils.architectures import get_network
import json
import numpy as np
import yaml
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 



classes=["bloodpool", "healthy muscle", "scar", "MVO"]
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_2dnet(path):
    params= yaml.load(open(path + "/config-best_weights.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
    net2d = get_network(architecture='unet2d', device=device, **params)
    net2d.load_state_dict(weights)
    net2d.eval()
    return net2d

def load_3dnet(path):
    params= yaml.load(open(path + "/config-best_weights.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
    net3d = get_network(architecture='unet', device=device, **params)
    net3d.load_state_dict(weights)
    net3d.eval()
    return net3d

def normalize(img):
    for i in range(img.shape[0]):
        img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
    return img


folder="DATA/emidec-segmentation-testset-1.0.0/"
patients= [f for f in os.listdir(folder) if os.path.isdir(folder+f)]

savepath="RESULTS_FOLDER/testset/"
if not os.path.exists(savepath):
    os.makedirs(savepath)
    

for patient in patients:
    if not os.path.exists(os.path.join(savepath,patient)):
        os.makedirs(os.path.join(savepath,patient))
    img  = nib.load(os.path.join(folder, patient, 'images', patient+'.nii.gz')).get_fdata()
    center=(img.shape[0]//2, img.shape[1]//2)
    img=np.transpose(img,(2,0,1))
    np.save(os.path.join(savepath,patient,"image.npy"),img)
    shape= img.shape
    img_roi=img[:,center[0]-48 : center[0]+48, center[1]-48: center[1]+48].copy()
    img_roi=normalize(img_roi)
    im = torch.from_numpy(img_roi[None,None, ...].astype("float32")).to(device)
    for i in range(5):
        path=f"RESULTS_FOLDER/2d-net_{i}"
        net2d=load_2dnet(path)
        with torch.no_grad():
            out2d=[]
            in2d=torch.moveaxis(im,2,0)[:,0,...]
            temp=net2d(in2d)[0]
            temp=torch.moveaxis(temp,0,1)
            temp=torch.argmax(temp,0).long()
            temp=torch.nn.functional.one_hot(temp,5)
            temp=torch.moveaxis(temp,-1,0)
            out2d.append(temp[3:,...])
            out2d=torch.stack(out2d,0)
            
        
        path=f"RESULTS_FOLDER/3d-cascade_{i}"
        net3d=load_3dnet(path)
        with torch.no_grad():
            in3d=torch.cat((im,out2d),1)
            out3d=net3d(in3d)[0]
            
                
            out3d=torch.argmax(out3d,1).long()
            out3d=torch.nn.functional.one_hot(out3d,5)
            out3d=torch.moveaxis(out3d,-1,1).float()
            if i==0:
                result=torch.zeros_like(out3d)
            result+=out3d
    result= torch.argmax(result,1)[0,...].cpu().detach().numpy()
    result=np.pad(result, ((0,0),(center[0]-48, shape[1]-(center[0]+48)), (center[1]-48, shape[2]-(center[1]+48))), 
                           constant_values=0)
    np.save(os.path.join(savepath,patient,"prediction.npy"),result)



    segmentation=np.ma.masked_where(result ==0, result)
    segmentation-=1
    for i in range(img.shape[0]):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i], cmap='gray')
        plt.gca().set_title(patient+"_"+str(i))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img[i], cmap='gray')
        mat=plt.imshow(segmentation[i], 'jet', alpha=0.5, interpolation="none", vmin = 0, vmax = 3)
        plt.axis('off')
        plt.gca().set_title('Prediction')
            
        values = np.array([0,1,2,3])
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
            
        plt.savefig(os.path.join(savepath,patient,f"Slice_{i}.png"), bbox_inches='tight', dpi=500)
        plt.show()
    print("Saved results for",patient)
        


