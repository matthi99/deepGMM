# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:05:37 2022

@author: A0067501
"""

import torch



class VariantGMM(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VariantGMM, self).__init__()
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self, predictions, inputs, heart):
        (B,K,X,Y)=predictions.shape
        M=inputs.shape[1]
        eps=1e-10
        likelylosses=torch.zeros(B).to(self.device)
        
        for b in range(B):
            pred = []
            for cl in range(K):
                pred.append(predictions[b,cl,...][heart[b,0,...]==1])
            pred = torch.stack(pred,dim=0)
            inp=[]
            for ch in range(M):
                inp.append(inputs[b,ch,...][heart[b,0,...]==1])
            inp = torch.stack(inp,dim=0)
            
            mu=torch.zeros((K,M)).to(self.device)
            var=torch.zeros((K,M)).to(self.device)
            for k in range(K):
                for m in range(M):
                    mu[k,m]=torch.sum(pred[k,...]*inp[m,...])/(torch.sum(pred[k,...])+eps)
                    var[k,m]=(torch.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(torch.sum(pred[k,...])+eps))+eps
            temp=torch.zeros((K,M,inp.shape[1])).to(self.device)
            for k in range(K):
                for m in range(M):
                    temp[k,m,...]=(1/(torch.sqrt(2*torch.pi*var[k,m])))*torch.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
            temp =  torch.prod(temp,1)
            temp= pred * temp
            likelylosses[b]=-torch.mean(torch.log(torch.sum(temp,axis=0)+eps))
        return torch.mean(likelylosses)


class NormalGMM(torch.nn.Module):
    def __init__(self, **kwargs):
        super(NormalGMM, self).__init__()
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self, predictions, inputs, heart):
        (B,K,X,Y)=predictions.shape
        M=inputs.shape[1]
        eps=1e-10
        likelylosses=torch.zeros(B).to(self.device)
        
        for b in range(B):
            pred = []
            for cl in range(K):
                pred.append(predictions[b,cl,...][heart[b,0,...]==1])
            pred = torch.stack(pred,dim=0)
            inp=[]
            for ch in range(M):
                inp.append(inputs[b,ch,...][heart[b,0,...]==1])
            inp = torch.stack(inp,dim=0)
            
            alpha = torch.mean(pred,1)
            mu=torch.zeros((K,M)).to(self.device)
            var=torch.zeros((K,M)).to(self.device)
            for k in range(K):
                for m in range(M):
                    mu[k,m]=torch.sum(pred[k,...]*inp[m,...])/(torch.sum(pred[k,...])+eps)
                    var[k,m]=(torch.sum(pred[k,...]*(inp[m,...]-mu[k,m])**2)/(torch.sum(pred[k,...])+eps))+eps

            temp=torch.zeros((K,M,inp.shape[1])).to(self.device)
            for k in range(K):
                for m in range(M):
                    temp[k,m,...]=(1/(torch.sqrt(2*torch.pi*var[k,m])))*torch.exp(-((inp[m,...]-mu[k,m])**2/(2*var[k,m])))
            temp =  torch.prod(temp,1)
            temp = alpha[:,None]*temp
            
            likelylosses[b]=-torch.mean(torch.log(torch.sum(temp,axis=0)+eps))
            #print(alpha)
        return torch.mean(likelylosses)




class Mu_data(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Mu_data, self).__init__()
    def forward(self, predictions, inputs, heart, mu_data):
        (B,K,X,Y)=predictions.shape
        M=inputs.shape[1]
        eps=1e-10
        mu_mean=torch.zeros_like(mu_data)
        for b in range(B):
            pred = []
            for cl in range(K):
                pred.append(predictions[b,cl,...][heart[b,0,...]==1])
            pred = torch.stack(pred,dim=0)
            inp=[]
            for ch in range(M):
                inp.append(inputs[b,ch,...][heart[b,0,...]==1])
            inp = torch.stack(inp,dim=0)
            mu=torch.zeros_like(mu_data)
            for k in range(K):
                for m in range(M):
                    mu[k,m]=torch.sum(pred[k,...]*inp[m,...])/(torch.sum(pred[k,...])+eps)

            mu_mean+=mu    
        mu_mean = mu_mean/B
        return torch.sum((mu_data-mu_mean)**2)



class Probs(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Probs, self).__init__()
    def forward(self, predictions, heart):    
        (B,K,X,Y)=predictions.shape
        l_blood=predictions[:,1,...][heart[:,0,...]==1]
        l_muscle=predictions[:,2,...][heart[:,0,...]==1]
        l_edema=predictions[:,3,...][heart[:,0,...]==1]
        l_scar=predictions[:,4,...][heart[:,0,...]==1]
        pred=torch.stack((l_blood, l_muscle, l_edema, l_scar))
        
        N=len(l_blood)
        probs = (torch.sum(pred,axis=1)/N)
        
        # a=probs[0]
        # probs=probs[1:]*(1/(1-a))
        #print(probs.sum())
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #probs_gt=torch.tensor([0.3839563844238861, 0.4023091477825316,  0.098236304180275, 0.11549816361330738]).to(device)
        probs_gt = (torch.ones(4)/4).to(device)
        #print(probs_gt)
        probs_loss = torch.mean((probs-probs_gt)**2)
        return probs_loss
    


class DiceLoss(torch.nn.Module):
    """
    Dice loss function for training a multi class segmentation network.
    The prediction of the network should be of type softmax(...) and the target should be one-hot encoded.
    """

    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.num_classes = kwargs.get("num_classes", 1)
        self.weights = kwargs.get("weights", self.num_classes * [1])
        self.smooth = kwargs.get("smooth", 1.)
        self.p = kwargs.get("p", 2)

    def _single_class(self, prediction, target):
        bs = prediction.size(0)
        p = prediction.reshape(bs, -1)
        t = target.reshape(bs, -1)

        intersection = (p * t).sum(1)
        total = (p.pow(self.p) + t.pow(self.p)).sum(1)

        loss = 1 - (2 * intersection + self.smooth) / (total + self.smooth)
        return loss.mean()

    def forward(self, prediction, target):
        assert prediction.shape == target.shape
        loss = 0
        for c in range(self.num_classes):
            loss += self._single_class(prediction[:, c, ...], target[:, c, ...]) * self.weights[c]
        return loss / sum(self.weights)


def get_loss(crit="NormalGMM", **kwargs):
    if crit == "dice":
        return DiceLoss()
    elif crit == "mu_data":
        return Mu_data()
    elif crit == "probs":
        return Probs()
    elif crit == "NormalGMM":
        return NormalGMM()
    elif crit == "VariantGMM":
        return VariantGMM()
    else:
        return print("wrong crit!")

