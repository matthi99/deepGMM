# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:23:30 2024

@author: A0067501
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

 


folder = "DATA/preprocessed/traindata2d/"

files= os.listdir(folder)

np.random.seed(42)
np.random.shuffle(files)
files = files[0:10]


modalities = ["LGE", "T2", "C0"]
classes = ["blood", "muscle", "edema", "scar"]


means={}
probs={}
stds={}
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
        plt.figure()
        plt.imshow(np.argmax(mask,0))
        plt.title(f"{file}_{i}")
        plt.show()
        
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
        

plt.figure()
plt.title("Blood")
sns.kdeplot(np.array(means["LGE"]["blood"]), bw=0.5, )
sns.kdeplot(np.array(means["T2"]["blood"]), bw=0.5)
sns.kdeplot(np.array(means["C0"]["blood"]), bw=0.5)
plt.legend(labels=['LGE', 'T2', 'C0'])

plt.figure()
plt.title("Muscle")
sns.kdeplot(np.array(means["LGE"]["muscle"]), bw=0.5, )
sns.kdeplot(np.array(means["T2"]["muscle"]), bw=0.5)
sns.kdeplot(np.array(means["C0"]["muscle"]), bw=0.5)
plt.legend(labels=['LGE', 'T2', 'C0'])

plt.figure()
plt.title("Edema")
sns.kdeplot(np.array(means["LGE"]["edema"]), bw=0.5, )
sns.kdeplot(np.array(means["T2"]["edema"]), bw=0.5)
sns.kdeplot(np.array(means["C0"]["edema"]), bw=0.5)
plt.legend(labels=['LGE', 'T2', 'C0'])

plt.figure()
plt.title("Scar")
sns.kdeplot(np.array(means["LGE"]["scar"]), bw=0.5, )
sns.kdeplot(np.array(means["T2"]["scar"]), bw=0.5)
sns.kdeplot(np.array(means["C0"]["scar"]), bw=0.5)
plt.legend(labels=['LGE', 'T2', 'C0'])

print("Blood:", np.mean(probs["blood"]))
print("Muscle:", np.mean(probs["muscle"]))
print("Edema:", np.mean(probs["edema"]))
print("Scar:", np.mean(probs["scar"]))


mu=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        mu[i,j]=np.mean(means[modality][cl])
    
print(mu)


sigma=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        sigma[i,j]=np.mean(stds[modality][cl])
    
print(sigma)


mu=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        mu[i,j]=means[modality][cl][75]
    
print(mu)


sigma=np.zeros((4,3))
for cl,i in zip(classes, range(4)):
    for modality,j in zip(modalities, range(3)):
        sigma[i,j]=np.nanmean(stds[modality][cl])
    
print(sigma)