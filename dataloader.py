# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:20:16 2022

@author: A0067501
"""

import os

import numpy as np
import torch
import random
from PIL import Image
from config import Config

import torchio as tio
import scipy
#from scipy import stats
import matplotlib.patches as mpatches 
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from skimage.transform import resize

class CardioDataset(torch.utils.data.Dataset):
    """
    Dataset for the cardio data.
    This dataset loads an image and applies some transformations to it.
    """

    def __init__(self, folder="DATA/preprocessed/traindata2d/", validation=False, test=False, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder = folder
        self.validation = validation
        self.test = test
        self.patients = kwargs.get("patients")
        self.name = kwargs.get("name", "Case")
        self.modalities= kwargs.get("modalities", ["LGE", "T2", "C0"])
        
        if self.patients is None:
            if validation:
                self.patients = Config.val_patients
            elif test:
                self.patients = Config.test_patients
            else:
                self.patients = Config.train_patients

        self.patients = [f"{self.name}_{format(num, '03d')}" for num in self.patients]
        self.examples = self.get_examples()
        self.spatial_transforms, self.intensity_transforms = self.prepare_transforms(**kwargs)

    def __getitem__(self, idx):
        example=self.examples[idx]
        z = np.load(os.path.join(self.folder, self.examples[idx]), allow_pickle=True).item()
        
        mask = np.moveaxis(z["masks"], -1, 0)
        inp = {}
        for modality in self.modalities:
            inp[modality]= z[modality]
        center=z['center']
        
        for transform in self.spatial_transforms:
            inp, mask = transform(inp, mask, center)
            
        for transform in self.intensity_transforms:
            for modality in self.modalities:
                inp[modality], mask = transform(inp[modality], mask, center)
                
            
        inp= np.stack([inp[modality] for modality in self.modalities],0)        
        inp = torch.from_numpy(inp.astype("float32")).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)
        return inp, mask, example

    def __len__(self):
        return len(self.examples)

    def is_valid(self, example):
        flag = False
        for patient in self.patients:
            flag = flag or example.startswith(patient)
        return flag

    def get_examples(self):
        total = [f for f in os.listdir(self.folder) if f.endswith('.npy')]
        examples = [f for f in total if self.is_valid(f)]
        return examples

    
    def prepare_transforms(self,**kwargs):
        spacial_transforms = []
        intensity_transforms = []
        if kwargs.get("spatialtransform") is not None:
            spacial_transforms.append(SpatialTransform(self.modalities, **kwargs.get("spatialtransform")))
        if kwargs.get("ROI") is not None:
            spacial_transforms.append(getRoi(self.modalities,**kwargs.get("ROI")))
        if kwargs.get("flip"):
            spacial_transforms.append(RandomFlip(self.modalities))
        if kwargs.get("normalize") is not None:
            intensity_transforms.append(Normalize(**kwargs.get("normalize")))
        if kwargs.get("gaussian") is not None:
            intensity_transforms.append(Gaussian_noise(**kwargs.get("gaussian")))
        if kwargs.get("gaussianblur") is not None:
            intensity_transforms.append(GaussianBlur(**kwargs.get("gaussianblur")))    
        if kwargs.get("brighness") is not None:
            intensity_transforms.append(Brightness(**kwargs.get("brightness")))
        if kwargs.get("contrast") is not None:
            intensity_transforms.append(Contrast(**kwargs.get("contrast")))  
        if kwargs.get("lowres") is not None:
            intensity_transforms.append(LowResolution(**kwargs.get("lowres")))   
        if kwargs.get("ghosting"):
            intensity_transforms.append(RandomGhosting())       
        if kwargs.get("gamma") is not None:  
            intensity_transforms.append(GammaTransform())
        
        return spacial_transforms, intensity_transforms





class CardioCollatorMulticlass:
    """
    Data collator for multiple classes.
    This collator returns batches of (images, dictionaries) where the dictionary contains masks for all the classes.
    """

    def __init__(self, classes=("bg", "blood", "muscle", "edema", "scar"), classes_data =("bg", "blood", "muscle", "edema", "scar"), **kwargs):
        self.classes = classes
        self.kwargs = kwargs
        self.return_all = kwargs.get("return_all", False)
        self.classes_data = classes_data 

    
    def _get_index(self, cls):
        return [self.classes_data.index(cls)]

    def _prepare_masks(self, mask):
        temp = {}
        for cls in self.classes:
            idx = self._get_index(cls)
            temp[cls] = mask[idx, ...]
        return temp

    def _stack_masks(self, masks):
        return {cls: torch.stack([m[cls] for m in masks]) for cls in self.classes}

    def __call__(self, batch, *args, **kwargs):
        inputs = []
        masks = []
        examples=[]
        for inp, mask, example in batch:
            inputs.append(inp)
            masks.append(self._prepare_masks(mask))
            examples.append(example)
        if self.return_all:
             return torch.stack(inputs) ,self._stack_masks(masks), examples
        else:
            return torch.stack(inputs) ,self._stack_masks(masks)

        
        
class getRoi(object):
    def __init__(self, modalities, **kwargs):
        self.width=kwargs.get("width", 48)
        self.height=kwargs.get("height", 48)
        self.t=kwargs.get("translation", 0)
        self.modalities = modalities
    
    def __call__(self, img, mask, center):
        tx=np.random.randint(-self.t, self.t+1)
        ty=np.random.randint(-self.t, self.t+1)
        cx=center[0]+tx
        cy=center[1]+ty
        if len(img[self.modalities[0]].shape) ==3:
            for modality in self.modalities:
                img[modality]=img[modality][:,cx-self.width:cx+self.width,cy-self.height:cy+self.height]
            mask=mask[:,:,cx-self.width:cx+self.width,cy-self.height:cy+self.height]
        else:
            for modality in self.modalities:
                img[modality]=img[modality][cx-self.width:cx+self.width,cy-self.height:cy+self.height]
            mask=mask[:,cx-self.width:cx+self.width,cy-self.height:cy+self.height]
        return img, mask
        
    

class Normalize(object):
    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "mean")

    def __call__(self, img, mask, center):
        if self.mode == "mean":
            if len(img.shape) == 3:
                for i in range(img.shape[0]):
                    if np.std(img[i,...])!=0:
                        img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
            else:
                img=(img - np.mean(img)) / np.std(img)
            return img, mask
        if self.mode == "mask":
            temp=img*mask
            temp=temp[temp>0]
            temp1=img/(2*np.median(temp))
            temp2=temp1[temp1>0]
            return (temp1 - np.min(temp2)) / (np.max(temp2) - np.min(temp2)), mask
        else:
            return (img - np.min(img)) / (np.max(img) - np.min(img)), mask


class RandomFlip(object):
    def __init__(self, modalities, p=None, methods=None):
        self.p = p
        self.methods = methods
        self.modalities = modalities
        if self.methods is None:
            self.methods = [Image.FLIP_LEFT_RIGHT,
                            Image.FLIP_TOP_BOTTOM]
        if self.p is None:
            self.p = len(self.methods) / (len(self.methods) + 1)

    def __call__(self, img, mask, center):
        if np.random.uniform(0, 1) <= self.p:
            method = np.random.choice(self.methods)
            if len(img[self.modalities[0]].shape) == 3:
                for i in range(img.shape[0]):
                    img[i,...] = np.asarray(Image.fromarray(img[i,...]).transpose(method=method))
                    for j in range(mask.shape[0]):
                        mask[j,i, ...] = np.asarray(Image.fromarray(mask[j,i, ...]).transpose(method=method))
            else:
                for modality in self.modalities:
                    img[modality] = np.asarray(Image.fromarray(img[modality]).transpose(method=method))
                for j in range(mask.shape[0]):
                    mask[j,...] = np.asarray(Image.fromarray(mask[j,...]).transpose(method=method))
        
        return img, mask
    
    

class Gaussian_noise(object):
    def __init__(self, **kwargs):
        self.sigma=kwargs.get("sigma", 0.1)
        self.p= kwargs.get("p_gaussian", 0.15)
        
    def __call__(self, img, mask, center):
        if np.random.uniform(0, 1) <= self.p:
            sigma=np.random.uniform(0, self.sigma)
            noise=np.random.normal(0,sigma,img.shape)
            return img+noise, mask
        else:
            return img, mask


class GammaTransform(object):
    def __init__(self, **kwargs):
        self.retain_stats = kwargs.get("retain_stats",True)
        self.gamma_range = kwargs.get("gamma_range", (0.7, 1.5))
        self.invert_image = kwargs.get("invert_image", False)
        self.epsilon=1e-7
        self.p_per_sample = kwargs.get("p_per_sample", 0.15)
    
    def __call__(self, img, mask, center):
        if np.random.uniform() <= self.p_per_sample:
            if self.invert_image:
                img = - img
    
            retain_stats_here = self.retain_stats() if callable(self.retain_stats) else self.retain_stats
            if retain_stats_here:
                mn = img.mean()
                sd = img.std()
            if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                gamma = np.random.uniform(self.gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
            minm = img.min()
            rnge = img.max() - minm
            img = np.power(((img - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
            if retain_stats_here:
                img = img - img.mean()
                img = img / (img.std() + 1e-8) * sd
                img = img + mn
            if self.invert_image:
                img = - img
        return img, mask

class Brightness(object):
    def __init__(self, **kwargs):
        self.mult_range=kwargs.get("mult_range", (0.7, 1.3))
        self.p_per_sample = kwargs.get("p_per_sample", 0.15)
    
    def __call__(self, img, mask, center):
        if np.random.uniform() <= self.p_per_sample:
            multiplier = np.random.uniform(self.mult_range[0], self.mult_range[1])
            img = img * multiplier
        return img, mask
            
class Contrast(object):
    def __init__(self, **kwargs):
        self.contrast_range=kwargs.get("contrast_range", (0.65, 1.5))
        self.p_per_sample = kwargs.get("p_per_sample", 0.15)
        
    
    def __call__(self, img, mask, center):
        if np.random.uniform() <= self.p_per_sample:
            if len(img.shape) == 3:
                for i in range(img.shape[0]):
                    if np.random.random() < 0.5: 
                        factor = np.random.uniform(self.contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
                    mn = img[i,...].mean()
                    minm = img[i,...].min()
                    maxm = img[i,...].max()
    
                    img[i,...] = (img[i,...] - mn) * factor + mn
    
                    
                    img[i,...][img[i,...] < minm] = minm
                    img[i,...][img[i,...] > maxm] = maxm
            else:
                if np.random.random() < 0.5: 
                    factor = np.random.uniform(self.contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
                mn = img.mean()
                minm = img.min()
                maxm = img.max()

                img= (img- mn) * factor + mn
                
                img[img < minm] = minm
                img[img > maxm] = maxm
    
        return img, mask        
        
    
class GaussianBlur(object):
    def __init__(self, **kwargs):
        self.blur_sigma = kwargs.get("blur_sigma", (0.5, 1.5))
        self.p_per_sample = kwargs.get("p_per_sample", 0.1)
    def __call__(self, img, mask, center):
        if np.random.uniform() <= self.p_per_sample:
            sigma=np.random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            if len(img.shape) == 3:
                for i in range(img.shape[0]):
                    img[i,...] = gaussian_filter(img[i,...], sigma, order=0)
            else:
                img=gaussian_filter(img, sigma, order=0)
        return img, mask
                
class LowResolution(object):
    def __init__(self, **kwargs):
        self.scale_factor = kwargs.get("scale_factor", (0.5, 1))
        self.p_per_sample = kwargs.get("p_per_sample", 0.15)
    
    def __call__(self, img, mask, center):
        if np.random.uniform() < self.p_per_sample:
            scale=np.random.uniform(self.scale_factor[0], self.scale_factor[1])
            shp=img.shape
            target_shape =np.array(img.shape)
            target_shape[-1] = np.round(shp[-1] * scale).astype(int)
            target_shape[-2] = np.round(shp[-2] * scale).astype(int)
            downsampled = resize(img.astype(float), target_shape, order=0, mode='edge',
                         anti_aliasing=False)
            
                
            img = resize(downsampled, shp, order=3, mode='edge',
                                anti_aliasing=False)
        return img, mask
        
        
class RandomMotion(object):
    def __init__(self, degrees=5,translation=5, num_transorms=2, p_per_sample=0.1):
        self.degrees=degrees
        self.translation=translation
        self.num_transforms=num_transorms
        self.p_per_sample = p_per_sample 
    def __call__(self, img, mask, center):
        if np.random.uniform() < self.p_per_sample:
            subject = tio.Subject(image = tio.ScalarImage(tensor =np.expand_dims(img,0)),
                                  mask = tio.LabelMap(tensor = mask))
            transform=tio.RandomMotion(degrees=self.degrees,translation=self.translation,num_transforms=self.num_transforms)
            deformed=transform(subject)
            return np.array(deformed.image)[0,...], np.array(deformed.mask)
        else:
            return img, mask
            
class RandomGhosting(object):
    def __init__(self, num_ghosts=(4,10), axes=(1,2), intensity=(0.5, 1),restore = 0.02, p_per_sample=1):
        self.num_ghosts=num_ghosts
        self.axes=axes
        self.intensity=intensity
        self.restore=restore
        self.p_per_sample = p_per_sample 
    def __call__(self, img, mask, center):
        if np.random.uniform() < self.p_per_sample:
            transform=tio.RandomGhosting(num_ghosts=self.num_ghosts, axes=self.axes, intensity=self.intensity,restore=self.restore)
            img=transform(np.expand_dims(img,0))[0,...]
        return img, mask
        


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = 2
    offsets = []
    if len(coordinates.shape)==4:
        offsets.append(np.zeros(coordinates.shape[1:]))
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices



class SpatialTransform(object):
    """We have to create a tranform that does ellastic deformations rotating and scaling
            nnunet parametrs are:
                
            "do_elastic": True,
            "elastic_deform_alpha": (0., 900.) ---> in 2d alpha= (0.200.)
            "elastic_deform_sigma": (9., 13.),
            "p_eldef": 0.2,
        
        
            "do_scaling": True,
            "scale_range": (0.85, 1.25),
            "independent_scale_factor_for_each_axis": False,
            "p_independent_scale_per_axis": 1,
            "p_scale": 0.2,
            
            "do_rotation": True,
            "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_p_per_axis": 1,
            "p_rot": 0.2,


    """
    def __init__(self, modalities,  border_mode_data='nearest', border_cval_data=0, order_data=3,
                  border_mode_seg='constant', border_cval_seg=0, order_seg=0, **kwargs):
        self.do_elastic_deform= kwargs.get("do_elastic_deform", True)
        self.alpha = kwargs.get("alpha", (0., 900.))
        self.sigma = kwargs.get("sigma", (9., 13.))
        self.do_rotation = kwargs.get("do_rotation", True)
        self.angles = kwargs.get("angles", (0, 2 * np.pi))
        self.do_scale = kwargs.get("do_scale", True)
        self.scale = kwargs.get("scale", (0.7, 1.4))
        self.p_el_per_sample = kwargs.get("p_el_per_sample", 0.2)
        self.p_scale_per_sample = kwargs.get("p_scale_per_sample", 0.2)
        self.p_rot_per_sample = kwargs.get("p_rot_per_sample", 0.2)
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.modalities = modalities
        
        

    def __call__(self, img, mask, center):

        coords = create_zero_centered_coordinate_mesh(img[self.modalities[0]].shape)
        
        if len(img[self.modalities[0]].shape)==3:
            coords[1]+=(img[self.modalities[0]].shape[1]//2-center[0])
            coords[2]+=(img[self.modalities[0]].shape[2]//2-center[1])
        else:
            coords[0]+=(img[self.modalities[0]].shape[0]//2-center[0])
            coords[1]+=(img[self.modalities[0]].shape[1]//2-center[1])
        modified_coords = False
        
        if self.do_elastic_deform and np.random.uniform() < self.p_el_per_sample:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True
        
        if self.do_rotation and np.random.uniform() < self.p_rot_per_sample:
            
            angle = np.random.uniform(self.angles[0], self.angles[1])
            if len(img[self.modalities[0]].shape)==3:
                rot_matrix = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
            else: 
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
                
            coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
            modified_coords = True
    
       
        if self.do_scale and np.random.uniform() < self.p_scale_per_sample:
            if np.random.random() < 0.5 and self.scale[0] < 1.:
                sc = np.random.uniform(self.scale[0], 1)
            else:
                sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])
            if len(img[self.modalities[0]].shape)==3:
                coords[1:] *= sc
            else:
                coords*=sc
            modified_coords = True
        
        if modified_coords:
            #reverse previous translation
            if len(img[self.modalities[0]].shape)==3:
                coords[0]+=2
                coords[1]+=center[0]
                coords[2]+=center[1]
                
            else:
                coords[0]+=center[0]
                coords[1]+=center[1]

            
            # for i in range(len(img[self.modalities[0]].shape)):
            #     coords[i]+=img[self.modalities[0]].shape[i]//2
            
            for modality in self.modalities:
                img[modality] = map_coordinates(img[modality].astype(float), coords, order=self.order_data, mode= self.border_mode_data, cval=self.border_cval_data)
            temp=np.argmax(mask,0)
            temp=map_coordinates(temp.astype(float), coords, order = self.order_seg, mode=self.border_mode_seg, cval=self.border_cval_seg)
            mask=(np.stack([temp==i for i in range(5)],0)*1).astype(np.float64)
            
        return img, mask
    
    
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    # p = 'images/dataset/'
    classes=["bloodpool", "healthy muscle", "edema", "scar", "mvo"]

    # if not os.path.exists(p):
    #     os.mkdir(p)
    
    
    setup=Config.train_data_setup
    
    cd = CardioDataset( **setup)


    collator = CardioCollatorMulticlass(**{'return_all': True})
    dataset = DataLoader(cd,  batch_size=1, collate_fn=collator, shuffle=False)
    
    l=1
    k=0
    for img ,mask, example in dataset:
        
        im_LGE=img[0,0, ...].cpu().detach().numpy()
        im_T2=img[0,1, ...].cpu().detach().numpy()
        im_C0=img[0,2, ...].cpu().detach().numpy()
        blood=mask["blood"][0,0, ...].cpu().detach().numpy()
        muscle=(mask["muscle"][0,0, ...]).cpu().detach().numpy()
        edema=mask["edema"][0,0, ...].cpu().detach().numpy()
        scar=mask["scar"][0,0, ...].cpu().detach().numpy()
            
        seg=blood+2*muscle+3*edema+4*scar
        seg=seg-1
        seg = np.ma.masked_where(seg == -1, seg)
                
                
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(im_LGE, cmap='gray')
        plt.gca().set_title("LGE"+example[0])
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(im_T2, cmap='gray')
        plt.gca().set_title("T2")
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(im_C0, cmap='gray')
        plt.gca().set_title("C0")
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(im_LGE, cmap='gray')
        plt.gca().set_title("Lables")
        plt.axis('off')
        mat=plt.imshow(seg, 'jet', alpha=0.5, interpolation="none", vmin = 0, vmax = 3)
        plt.axis('off')
        plt.gca().set_title('Labels')
                
        values = np.array([0,1,2,3])
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
                
            #plt.savefig("C:/Users/A0067501/Desktop/Project3D/create_dataset/check_labels/val/"+example[0][0:-4]+"_"+str(i)+".png", bbox_inches='tight', dpi=500)
        plt.show()
        
            # if np.max(mask[cls].cpu().detach().numpy())>1:
            #     print('Problem')
            #     test=mask[cls].cpu().detach().numpy()[0,0,...]
            #     for i in range(len(test)):
            #         plt.figure()
            #         plt.imshow(test[i])
            #         plt.colorbar()
            #         plt.show()
                    
                
            
            
            #plt.savefig(p+cls +'/image-'+str(l)+'.png', bbox_inches='tight')
            
            #plt.close()
        l=l+1
            

