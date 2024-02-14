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
from scipy import ndimage


def normalize(img):
    for i in range(img.shape[0]):
        img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
    return img

def load_net(path, device):
    params= yaml.load(open(path + "/config.json", 'r'), Loader=yaml.SafeLoader)['network']
    weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
    net = get_network(architecture='unet2d', device=device, **params)
    net.load_state_dict(weights)
    net.eval()
    return net

def plot_prediction(img, result, patientnr, classes, savefolder):
    segmentation=np.ma.masked_where(result == 0, result)
    segmentation-=1
    for i in range(img.shape[0]):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i], cmap='gray')
        plt.gca().set_title(f"Case_{patientnr}_{i}")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img[i], cmap='gray')
        mat=plt.imshow(segmentation[i], 'jet', alpha=0.45, interpolation="none", vmin = 0, vmax = len(classes))
        plt.axis('off')
        plt.gca().set_title('Prediction')
            
        values = np.arange(len(classes))
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
        plt.savefig(os.path.join(savefolder,f"Case_{patientnr}", f"Slice_{i}.png"), bbox_inches='tight', dpi=500)
        plt.close()


device= torch.device("cpu")
path_to_testdata= os.path.join("DATA", "MyoPS 2020 Dataset", "test20")
savefolder = "RESULTS_FOLDER/testdataset"
files_LGE=[os.path.join(path_to_testdata,f) for f in os.listdir(path_to_testdata) if f.endswith("DE.nii.gz")]
files_T2=[os.path.join(path_to_testdata,f) for f in os.listdir(path_to_testdata) if f.endswith("T2.nii.gz")]
files_C0=[os.path.join(path_to_testdata,f) for f in os.listdir(path_to_testdata) if f.endswith("C0.nii.gz")]
WIDTH=160



for im_LGE, im_T2, im_C0 in zip(files_LGE, files_T2, files_C0):
    data={}
    #load images
    #LGE
    LGE  = nib.load(im_LGE).get_fdata()
    center=(LGE.shape[0]//2, LGE.shape[1]//2)
    LGE=np.transpose(LGE,(2,0,1))
    shape = LGE.shape
    LGE=normalize(LGE[:, center[0]-WIDTH : center[0]+WIDTH, center[1]-WIDTH: center[1]+WIDTH])
    #T2
    T2  = nib.load(im_T2).get_fdata()
    T2=np.transpose(T2,(2,0,1))
    T2=normalize(T2[:, center[0]-WIDTH : center[0]+WIDTH, center[1]-WIDTH: center[1]+WIDTH])
    #C0
    C0  = nib.load(im_C0).get_fdata()
    C0=np.transpose(C0,(2,0,1))
    C0=normalize(C0[:, center[0]-WIDTH : center[0]+WIDTH, center[1]-WIDTH: center[1]+WIDTH])
    
    img=np.stack((LGE, T2, C0),0)
    img= np.transpose(img, (1,0,2,3))
    im=img.copy()
    im = torch.from_numpy(img.astype("float32")).to(device)
    
    net_heart=load_net(os.path.join("RESULTS_FOLDER", f"2d-net_heart_0"), device)
    pred_heart = torch.argmax(net_heart(im)[0],1).cpu().detach().numpy()
    middle = pred_heart[pred_heart.shape[0]//2,...]
    center_roi = ndimage.measurements.center_of_mass(middle)
    center_roi = (int(center_roi[0]), int(center_roi[1])) 
    im_roi = im[:,:,center_roi[0]-80 : center_roi[0]+80, center_roi[1]-80: center_roi[1]+80]
    
    pred = torch.zeros(( im.shape[0] , 5, 160, 160)).to(device)
    for i in range(1,2,1):
        net_likely=load_net(os.path.join("RESULTS_FOLDER", f"2d-net_test_with_mu{i}"), device)
        pred += net_likely(im_roi)[0]
        pred += torch.flip(net_likely(torch.flip(im_roi, dims=[2]))[0], dims=[2])
        pred += torch.flip(net_likely(torch.flip(im_roi, dims=[3]))[0], dims=[3])
    
    
    pred = torch.argmax(pred,1).cpu().detach().numpy()
    pred = np.pad(pred, ((0,0),(center_roi[0]-80, 2*WIDTH-(center_roi[0]+80)), (center_roi[1]-80, 2*WIDTH-(center_roi[1]+80))), 
                            constant_values=0)

    pred= pred*pred_heart
    pred_orig = np.pad (pred, ((0,0), (center[0]-WIDTH, shape[1]-(center[0]+WIDTH)), (center[1]-WIDTH, shape[2]-(center[1]+WIDTH))), 
                            constant_values=0)
    
    
    example = nib.load(im_LGE)
    prediction = nib.Nifti1Image(pred_orig, example.affine, example.header)
    patientnr= im_LGE.split("_")[-2]
    if not os.path.exists(os.path.join(savefolder, f"Case_{patientnr}")):
        os.makedirs(os.path.join(savefolder, f"Case_{patientnr}"))
    nib.save(prediction, os.path.join(savefolder,f"Case_{patientnr}", f"Case_{patientnr}.nii.gz"))
    
    plot_prediction(img[0,...], pred, patientnr, ["blood", "muscle", "edema", "scar"], savefolder)
    



   



# classes=["bloodpool", "healthy muscle", "scar", "MVO"]
# device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def load_2dnet(path):
#     params= yaml.load(open(path + "/config-best_weights.json", 'r'), Loader=yaml.SafeLoader)['network']
#     weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
#     net2d = get_network(architecture='unet2d', device=device, **params)
#     net2d.load_state_dict(weights)
#     net2d.eval()
#     return net2d

# def load_3dnet(path):
#     params= yaml.load(open(path + "/config-best_weights.json", 'r'), Loader=yaml.SafeLoader)['network']
#     weights = torch.load(path + "/best_weights.pth",  map_location=torch.device(device))
#     net3d = get_network(architecture='unet', device=device, **params)
#     net3d.load_state_dict(weights)
#     net3d.eval()
#     return net3d

# def normalize(img):
#     for i in range(img.shape[0]):
#         img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
#     return img


# folder="DATA/emidec-segmentation-testset-1.0.0/"
# patients= [f for f in os.listdir(folder) if os.path.isdir(folder+f)]

# savepath="RESULTS_FOLDER/testset/"
# if not os.path.exists(savepath):
#     os.makedirs(savepath)
    

# for patient in patients:
#     if not os.path.exists(os.path.join(savepath,patient)):
#         os.makedirs(os.path.join(savepath,patient))
#     img  = nib.load(os.path.join(folder, patient, 'images', patient+'.nii.gz')).get_fdata()
#     center=(img.shape[0]//2, img.shape[1]//2)
#     img=np.transpose(img,(2,0,1))
#     np.save(os.path.join(savepath,patient,"image.npy"),img)
#     shape= img.shape
#     img_roi=img[:,center[0]-48 : center[0]+48, center[1]-48: center[1]+48].copy()
#     img_roi=normalize(img_roi)
#     im = torch.from_numpy(img_roi[None,None, ...].astype("float32")).to(device)
#     for i in range(5):
#         path=f"RESULTS_FOLDER/2d-net_{i}"
#         net2d=load_2dnet(path)
#         with torch.no_grad():
#             out2d=[]
#             in2d=torch.moveaxis(im,2,0)[:,0,...]
#             temp=net2d(in2d)[0]
#             temp=torch.moveaxis(temp,0,1)
#             temp=torch.argmax(temp,0).long()
#             temp=torch.nn.functional.one_hot(temp,5)
#             temp=torch.moveaxis(temp,-1,0)
#             out2d.append(temp[3:,...])
#             out2d=torch.stack(out2d,0)
            
        
#         path=f"RESULTS_FOLDER/3d-cascade_{i}"
#         net3d=load_3dnet(path)
#         with torch.no_grad():
#             in3d=torch.cat((im,out2d),1)
#             out3d=net3d(in3d)[0]
            
                
#             out3d=torch.argmax(out3d,1).long()
#             out3d=torch.nn.functional.one_hot(out3d,5)
#             out3d=torch.moveaxis(out3d,-1,1).float()
#             if i==0:
#                 result=torch.zeros_like(out3d)
#             result+=out3d
#     result= torch.argmax(result,1)[0,...].cpu().detach().numpy()
#     result=np.pad(result, ((0,0),(center[0]-48, shape[1]-(center[0]+48)), (center[1]-48, shape[2]-(center[1]+48))), 
#                            constant_values=0)
#     np.save(os.path.join(savepath,patient,"prediction.npy"),result)



#     segmentation=np.ma.masked_where(result ==0, result)
#     segmentation-=1
#     for i in range(img.shape[0]):
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.imshow(img[i], cmap='gray')
#         plt.gca().set_title(patient+"_"+str(i))
#         plt.axis('off')
#         plt.subplot(1,2,2)
#         plt.imshow(img[i], cmap='gray')
#         mat=plt.imshow(segmentation[i], 'jet', alpha=0.5, interpolation="none", vmin = 0, vmax = 3)
#         plt.axis('off')
#         plt.gca().set_title('Prediction')
            
#         values = np.array([0,1,2,3])
#         colors = [ mat.cmap(mat.norm(value)) for value in values]
#         patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
#         plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
            
#         plt.savefig(os.path.join(savepath,patient,f"Slice_{i}.png"), bbox_inches='tight', dpi=500)
#         plt.show()
#     print("Saved results for",patient)
        


