# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:07:13 2023

@author: Utente
"""
import numpy as np
import fingerprints_dataset
from scipy import linalg

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 

def dataset_mean(dataset):
    return dataset.mean(1).reshape((dataset.shape[0],1)) #dataset.shape[0] è numero features per rendere codice generico

def num_samples(dataset):
    return dataset.shape[1];

def compute_Sb(dataset,labels,num_classes):
    N = num_samples(dataset)
    Sb = 0
    mu = dataset_mean(dataset)
    for label in range(num_classes):
        data_per_class = dataset[:, labels==label]
        nc = num_samples(data_per_class)
        muc = dataset_mean(data_per_class)
        Sb += nc*(np.dot((muc-mu),(muc-mu).T)) 
        
    return Sb/N
    
def compute_Sw(dataset,labels,num_classes): 
    Sw = 0
    N = num_samples(dataset)
    for label in range(num_classes):
        data_per_class = dataset[:, labels==label]
        muc = dataset_mean(data_per_class)
        covariance_matrix = (np.dot((data_per_class-muc),(data_per_class-muc).T))/num_samples(data_per_class)
        Swc = num_samples(data_per_class)*covariance_matrix
        Sw += Swc
        
    return Sw/N    
    

def lda_gen_eig(dataset,labels,num_classes,m):
    if(m>=num_classes):
        print("Attention LDA can reduce until num_classes-1 dimensions")
        return
    SB = compute_Sb(dataset, labels, num_classes)
    SW = compute_Sw(dataset, labels, num_classes)    
    s, U = linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = np.linalg.svd(W) #perchè W di default può non essere ortogonale
    U = UW[:, 0:m]
    return np.dot(W.T,dataset) 
    
def lda_joint_diag(dataset,labels,num_classes,m):
    if(m>=num_classes):
        print("Attention LDA can reduce until num_classes-1 dimensions")
        return   
    SB = compute_Sb(dataset, labels, num_classes)
    SW = compute_Sw(dataset, labels, num_classes)
    U, s, _ = np.linalg.svd(SW)
    P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T)
    SBT = np.dot(np.dot(P1,SB),P1.T)
    s, U = linalg.eigh(SBT)
    P2 = U[:, ::-1][:, 0:m]
    W = np.dot(P1.T,P2)
    return np.dot(W.T,dataset)
  
if __name__ == '__main__':
  
    dataset, labels = fingerprints_dataset.load("Train.txt");
    dataset_with_LDA = lda_joint_diag(dataset, labels, 2, 2)
    fingerprints_dataset.plot_scatter(dataset_with_LDA, labels)







