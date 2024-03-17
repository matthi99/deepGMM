# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:56:33 2024

@author: A0067501
"""


from sklearn.mixture import GaussianMixture as GMM
import numpy as np


class VariantGMM():
    def __init__(self, 
                 n_components = 1, 
                 tol=0.001, 
                 means_init = None, 
                 init_params = "kmeans", 
                 weights_init = None, 
                 covariances_init=None, 
                 max_iter = 100):
        self.tol = tol
        self.means = means_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.n_components = n_components
        self.weights_ = weights_init
        self.means_ = means_init
        self.covariances_ = covariances_init
        self.reg_covar = 1e-6
        self.max_iter = max_iter
        
    
    def initialize_parameters(self, X):
        gmm = GMM(n_components=self.n_components, covariance_type="diag", max_iter=1, 
                  init_params = "random" ,means_init = self.means_init)
        #gmm._initialize_parameters(X,gmm.random_state)
        gmm.fit(X)
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_ = gmm.predict_proba(X)
        
    def M_step(self, X):
        mu=np.zeros_like(self.means_)
        var=np.zeros_like(self.covariances_)
        n_samples, n_mods = X.shape
        eps= 1e-10
        for k in range(self.n_components):
            for m in range(n_mods):
                mu[k,m]=np.sum(self.weights_[:,k]*X[:,m])/(np.sum(self.weights_[:,k])+eps)
                var[k,m]=(np.sum(self.weights_[:,k]*(X[:,m]-mu[k,m])**2)/(np.sum(self.weights_[:,k])+eps))+eps
        
        self.means_ = mu
        self.covariances_ = var
        
    def E_step(self, X):
        n_samples, n_mods = X.shape
        
        temp=np.zeros((n_samples, self.n_components, n_mods))
        for k in range(self.n_components):
            for m in range(n_mods):
                temp[:,k,m]=(1/(np.sqrt(2*np.pi*self.covariances_[k,m])))*np.exp(-((X[:,m]-self.means_[k,m])**2/(2*self.covariances_[k,m])))
        temp =  np.prod(temp,2)
        temp= self.weights_ * temp
        self.weights_ = temp/temp.sum(axis=1)[:, np.newaxis]
        
    def compute_neg_log_likely(self,X):
        n_samples, n_mods = X.shape
        
        temp=np.zeros((n_samples, self.n_components, n_mods))
        for k in range(self.n_components):
            for m in range(n_mods):
                temp[:,k,m]=(1/(np.sqrt(2*np.pi*self.covariances_[k,m])))*np.exp(-((X[:,m]-self.means_[k,m])**2/(2*self.covariances_[k,m])))
        temp =  np.prod(temp,2)
        temp= self.weights_ * temp
        neg_log_likely = -np.mean(np.log(np.sum(temp,axis=1)))
        return neg_log_likely
        
    
    def fit(self, X):
        self.initialize_parameters(X)
        for n_iter in range(1, self.max_iter + 1):
            prev_neg_log_likely = self.compute_neg_log_likely(X)
            self.E_step(X)
            self.M_step(X)
            neg_log_likely = self.compute_neg_log_likely(X)
            change = neg_log_likely - prev_neg_log_likely
            if abs(change) < self.tol:
                break
        self.n_iter = n_iter
        self.neg_log_likely = neg_log_likely
        
    

