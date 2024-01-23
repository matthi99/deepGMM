# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:54:10 2022

@author: A0067501
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches 
import logging
import sys
import shutil
import matplotlib.backends.backend_pdf
import random
from skimage import measure


class Histogram():
    def __init__(self, classes, metrics):
        self.classes = classes
        self.metrics = metrics
        self.hist = self.prepare_hist()
        
    def prepare_hist(self):
        hist={}
        hist['loss'] = []
        for m in self.metrics:
            hist[m]={}
            hist[m]["train_mean"] = [] 
            hist[m]["val_mean"] = []
            for cl in self.classes:
                hist[m][f"train_{cl}"] = [] 
                hist[m][f"val_{cl}"] = []
        return hist
    
    def append_hist(self):
        self.hist['loss'].append(0)
        for m in self.metrics:
            self.hist[m]["train_mean"].append(0)
            self.hist[m]["val_mean"].append(0)
            for cl in self.classes:
                self.hist[m][f"train_{cl}"].append(0) 
                self.hist[m][f"val_{cl}"].append(0)
        
    def add_loss(self, loss):
        self.hist['loss'][-1]+=loss.item()
    
    def add_train_metrics(self, out, gt):
        #make output binary
        with torch.no_grad():
            temp=torch.argmax(out,1).long()
            temp=torch.nn.functional.one_hot(temp,5)
            out=torch.moveaxis(temp, -1, 1)
        for m in self.metrics:
            with torch.no_grad():
                values=self.metrics[m](out,gt).cpu().detach().numpy()
            if len(values)== len(self.classes):
                self.hist[m]["train_mean"][-1]+=values.mean()
                for cl, i in zip(self.classes, range(len(values))):
                    self.hist[m][f"train_{cl}"][-1]+=values[i]
            else:
                self.hist[m]["train_mean"][-1]+=values.mean()
                
    def add_val_metrics(self, out, gt):
        #make output binary
        with torch.no_grad():
            temp=torch.argmax(out,1).long()
            temp=torch.nn.functional.one_hot(temp,5)
            out=torch.moveaxis(temp, -1, 1)
        for m in self.metrics:
            with torch.no_grad():
                values=self.metrics[m](out,gt).cpu().detach().numpy()
            if len(values)== len(self.classes):
                self.hist[m]["val_mean"][-1]+=values.mean()
                for cl, i in zip(self.classes, range(len(values))):
                    self.hist[m][f"val_{cl}"][-1]+=values[i]
            else:
                self.hist[m]["val_mean"][-1]+=values.mean()
                
    def scale_train(self,steps):
        self.hist["loss"][-1]/=steps
        for m in self.metrics:
            self.hist[m]["train_mean"][-1]/=steps
            for cl in self.classes:
                self.hist[m][f"train_{cl}"][-1]/=steps
    
    def scale_val(self,steps):
        for m in self.metrics:
            self.hist[m]["val_mean"][-1]/=steps
            for cl in self.classes:
                self.hist[m][f"val_{cl}"][-1]/=steps
    
    def print_hist(self,logger):
        logger.info("Training:")
        loss=self.hist["loss"][-1]
        logger.info(f"Loss = {loss}")
        for m in self.metrics:
            value = self.hist[m]["train_mean"][-1]
            logger.info(f"Mean {m} = {value}")
            for cl in self.classes:
                value = self.hist[m][f"train_{cl}"][-1]
                logger.info(f"{cl} {m} = {value}")
        logger.info("Validation:")
        for m in self.metrics:
            value = self.hist[m]["val_mean"][-1]
            logger.info(f"Mean {m} = {value}")
            for cl in self.classes:
                value = self.hist[m][f"val_{cl}"][-1]
                logger.info(f"{cl} {m} = {value}")
    
    def plot_hist(self ,savefolder):
        #plot loss
        plt.figure()
        plt.plot(np.array(self.hist["loss"]))
        plt.title("Loss")
        plt.savefig(savefolder+f"Loss.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        for m in self.metrics:
            # plot mean metric
            plt.figure()
            plt.plot(np.array(self.hist[m]["train_mean"]))
            plt.plot(np.array(self.hist[m]["val_mean"]))
            plt.title(f"Mean {m}")
            plt.savefig(savefolder+f"Mean {m}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            #plot metric for different classes
            for cl in self.classes:
                plt.figure()
                plt.plot(np.array(self.hist[m][f"train_{cl}"]))
                plt.plot(np.array(self.hist[m][f"val_{cl}"]))
                plt.savefig(savefolder+f"{m} {cl}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.close()
            
                
            
       
def plot_examples(im, out, epoch, savefolder, train=True)    :
    dir=os.path.join(savefolder,f"Epoch_{epoch}")  
    if train:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)   
    if len(im.shape)==5:
        im=im[0,0,...]
        out= out[0,...]
    else:
        im=im[:,0,...]
        out=torch.moveaxis(out, 0, 1)
    img= im.cpu().detach().numpy()
    prob=out.cpu().detach().numpy()
    classes=np.argmax(prob,0)
    #plot a maximum of 6 images per epoch
    for i in range(min(classes.shape[0],6)):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i,...], cmap="gray")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(classes[i])
        plt.colorbar()
        plt.axis('off')
    
        if train:
            path=os.path.join(dir, f"train{i}.png")
        else:
            path=os.path.join(dir, f"val{i}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()    
    
    
        

def save_checkpoint(net, checkpoint_dir, fold, name="weights", savepath=False, z_dim=False):
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(net.state_dict(), checkpoint_path)
    if savepath:
        if not os.path.exists('paths/'):
            os.makedirs('paths/')
        if z_dim==True:
            with open(f"paths/best_weights3d_{fold}.txt", "w") as text_file:
                text_file.write(checkpoint_dir+"/")
        else:
            with open(f"paths/best_weights2d_{fold}.txt", "w") as text_file:
                text_file.write(checkpoint_dir+"/")
            
        
    
def isNaN(num):
    return num!= num

        
def plot_test_data(im,out, mask, patient, dicecoeff, classes, folder):      
    im_np=im[0, 0, ...].cpu().detach().numpy()
    out_np=out.cpu().detach().numpy()
    mask_blood=mask["blood"].float().cpu().detach().numpy()
    mask_muscle=((mask["heart"]*(1-mask["blood"]))*(1-mask["scar"])).float().cpu().detach().numpy()
    mask_scar=mask["scar"].float().float().cpu().detach().numpy()
             
    seg=mask_blood[0,0,...]+2*mask_muscle[0,0,...]+3*mask_scar[0,0,...]
    seg=seg-1
    seg=np.ma.masked_where(seg ==-1, seg)
             
    seg_pred=out_np
    seg_pred=np.ma.masked_where(seg_pred >=3, seg_pred)
    pdf = matplotlib.backends.backend_pdf.PdfPages(folder+'testplots/'+patient[0][0:-4]+'.pdf')
    for i in range(len(seg)):
        fig, axs = plt.subplots(1,3,  constrained_layout=True, dpi=500)
                 
        axs[0].imshow(im_np[i], cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("Input")
        axs[1].imshow(im_np[i], cmap='gray')
        axs[1].imshow(seg_pred[i],'jet', interpolation='none', alpha=0.5, vmin = 0, vmax = 2)
        axs[1].set_title("NN Segmentation")
        axs[1].axis('off')
        axs[2].imshow(im_np[i], cmap='gray')
        mat=axs[2].imshow(seg[i], 'jet',  interpolation='none', alpha=0.5, vmin = 0, vmax = 2)
        axs[2].set_title("Ground truth")
        axs[2].axis('off')
        
        values = np.array([0,1,2])
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
        plt.suptitle("Dice coefficient: " + str(round(dicecoeff,2)))
        pdf.savefig()
        plt.show()
        plt.close()

    pdf.close()
    plt.close()

def binary(mask):
    arg=torch.argmax(mask,1)
    
    blood=torch.zeros_like(arg)
    blood[arg==1]=1
    blood=blood[:,None,...]
    
    muscle=torch.zeros_like(arg)
    muscle[arg==2]=1
    muscle=muscle[:,None,...]
    
    scar=torch.zeros_like(arg)
    scar[arg==3]=1
    scar=scar[:,None,...]
    
    mvo=torch.zeros_like(arg)
    mvo[arg==4]=1
    mvo=mvo[:,None,...]
    return blood, muscle, scar, mvo

class CascadeAugmentation():
    def __init__(self, probs):
        self.probs = probs
        
    @staticmethod
    def delete_class(out2d):
        sl=np.random.randint(0,out2d.shape[1])
        if np.random.uniform() < 0.4:
            out2d[0,sl,...]=0
        else:
            out2d[1,sl,...]=0
        return out2d
        
    @staticmethod
    def delete_slices(out2d):
        slice_nrs=random.choice(range(1,4))
        slices=random.sample(range(out2d.shape[1]),slice_nrs)
        for sl in slices:
            out2d[:,sl,...]=0
        return out2d
        
    @staticmethod
    def delete_all(out2d):
        out2d=torch.zeros_like(out2d)
        return out2d
    
    @staticmethod
    def add_scar(out2d, muscle):
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sl=np.random.randint(0,out2d.shape[1])
        muscle= muscle[sl].cpu().detach().numpy()
        temp=muscle[muscle!=0]
        if len(temp)!=0:
            per = np.percentile(temp, 85)
            muscle[muscle<per]=0
            muscle[muscle!=0]=1
            labels = measure.label(muscle)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
            created_scar = largestCC*1
            created_scar = torch.from_numpy(created_scar).to(device)
            out2d[0,sl,...]+=((1-out2d[0,sl,...])*created_scar)
        return out2d
    
    @staticmethod
    def add_mvo(out2d):
        scar= out2d[0,...].clone().detach()
        mvo = out2d[1,...].clone().detach()
        if torch.sum(scar)!=0:
            indices= scar.nonzero()
            index=indices[np.random.randint(0,len(indices))].cpu().detach().numpy()
            n=[]
            for i in range(-1,2):
                for j in range(-1,2):
                    temp=torch.tensor(index)
                    temp[-1]+=i
                    temp[-2]+=j
                    n.append(temp)
            for pixel in n:
                if np.random.uniform() < 0.5:
                    scar[pixel[0], pixel[1], pixel[2]]=0
                    mvo[pixel[0], pixel[1], pixel[2]]=1
            out2d[0,...]=scar
            out2d[1,...]=mvo
            # plt.figure()
            # plt.imshow(out2d[0,pixel[0],...].cpu()+2*out2d[1,pixel[0],...].cpu())
            # plt.show()
            # plt.close()
        
        return out2d
     
        
        
    @staticmethod
    def nothing(out2d):
        return out2d
    
    def augment(self, net2d, im, gt, epoch, per_batch =True, do_nothing=False)  :
        disruption_list = [self.delete_class, self.delete_slices, self.delete_all, self.add_scar, self.add_mvo,  self.nothing]
        in2d=torch.moveaxis(im,2,0)
        out2d=[]
        with torch.no_grad():
            for b in range(im.shape[0]):
                temp=net2d(in2d[:,b,...])[0]
                temp=torch.moveaxis(temp,0,1)
                temp=torch.argmax(temp,0).long()
                temp=torch.nn.functional.one_hot(temp,5)
                temp=torch.moveaxis(temp,-1,0)[3:,...].float()
                if do_nothing:
                    index=-1
                else:
                    index=np.random.choice(np.arange(6), p=self.probs)
                fun= disruption_list[index]
                if index == 3:
                    muscle= gt[b,2,...]*im[b,0,...]
                    out2d.append(fun(temp,muscle))
                else:
                    out2d.append(fun(temp))
            out2d=torch.stack(out2d,0)
        return out2d


    
    
    
# def calculate_center(patient, net, folder="download/"):
#     #device = "cuda" if torch.cuda.is_available() else "cpu"
#     device="cpu"
#     datadir = os.path.join(folder, patient)
#     files = [x for x in os.walk(datadir)][-1]
#     root = files[0]
#     filenames = files[-1]
#     data=[]
    
#     for file in filenames:
#         data.append(pydicom.dcmread(root + "/" + file))
#     data = sorted(data, key = lambda k: k.get('InstanceNumber'))
#     X=np.zeros((len(data),256,256))
#     for i in range(len(data)):
#         X[i,...]=Normalize(constant_pad(data[i].pixel_array))
#     X=np.expand_dims(X, axis=0)
#     X=np.expand_dims(X, axis=0)
#     X =torch.from_numpy(X)
#     X=X.type(torch.FloatTensor)
#     X=X.to(device)
#     pred=net(X)
#     mask = pred >= 0.5
#     middle=mask[0,0,(len(data)//2)+1,:,:].cpu().numpy()
#     labels = label(middle)
#     #print(np.bincount(labels.flat))
#     middle = labels == np.argmax(np.bincount(labels.flat)[1:])+1
#     # plt.figure()
#     # plt.imshow(X[0,0,(len(data)//2)+1,:,:].cpu().detach().numpy())
#     # plt.imshow(middle, alpha=0.5)
#     # plt.colorbar()
#     # plt.show()
#     center=ndimage.measurements.center_of_mass(middle*1)
#     if isNaN(center[0]):
#         print("problem:", patient)
#         center=np.array([116, 105])
#     return (int(center[0]), int(center[1]))
    

# def calculate_center_2d(patient, net, folder="download/"):
#     device="cpu"
#     datadir = os.path.join(folder, patient)
#     files = [x for x in os.walk(datadir)][-1]
#     root = files[0]
#     filenames = files[-1]
#     data=[]
    
#     for file in filenames:
#         data.append(pydicom.dcmread(root + "/" + file))
#     data = sorted(data, key = lambda k: k.get('InstanceNumber'))
#     middle=Normalize(constant_pad(data[(len(data)//2)+1].pixel_array))
#     if patient == "Patient34":
#         middle=np.rot90(middle, k=-1).copy()
#     X=np.expand_dims(middle, axis=0)
#     X=np.expand_dims(X, axis=0)
#     X =torch.from_numpy(X)
#     X=X.type(torch.FloatTensor)
#     X=X.to(device)
#     pred=net(X)
#     mask=pred[0,0,:,:]
#     mask[mask>0.5]=1
#     mask[mask<1]=0
#     mask=mask.cpu().detach().numpy()
#     center=ndimage.measurements.center_of_mass(mask)
    
#     # plt.figure()
#     # plt.imshow(X[0,0,:,:].cpu().detach().numpy(), cmap="gray")
#     # ma= np.ma.masked_where(mask < 1, mask)
#     # plt.imshow(ma, 'jet', interpolation='none',alpha=0.5)
#     # plt.plot(center[1], center[0], "og", markersize=4) 
#     # plt.axis('off')
#     # plt.show()
#     if isNaN(center[0]):
#         print("problem:", patient)
#         center=np.array([116, 105])
#     return (int(center[0]), int(center[1]))





def Normalize(img):
    return (img - np.mean(img)) / np.std(img)

def constant_pad(x, c=2048):
     padding_size = ((0, 256 - x.shape[0]), (0, 256 - x.shape[1]))
     return np.pad(x, padding_size, mode='constant', constant_values=c)



def get_logger(name, level=logging.INFO, formatter = '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.handler_set = True
    return logger