# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:07:13 2023

@author: Utente
"""
import numpy
import fingerprints_dataset

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 

def dataset_mean(dataset):
    return dataset.mean(1).reshape((dataset.shape[0],1)) #dataset.shape[0] è numero features per rendere codice generico

def num_samples(dataset):
    return dataset.shape[1];

def pca(dataset,m): #m è la dimensione di output della PCA
    mu = dataset_mean(dataset)
    zero_mean_dataset = dataset - mu
    covariance_matrix = (numpy.dot(zero_mean_dataset, zero_mean_dataset.T))/num_samples(dataset)
    s, U = numpy.linalg.eigh(covariance_matrix)
    P = U[:, ::-1][:, 0:m] #prima inverto ordine poi prendo primi m autovettori
    print(P)
    return numpy.dot(P.T, dataset),P #ritorno già il dataset proiettato sugli assi della PCA

def pca_svd(dataset,m): #m è la dimensione di output della PCA
    mu = dataset_mean(dataset)
    zero_mean_dataset = dataset - mu
    covariance_matrix = (numpy.dot(zero_mean_dataset, zero_mean_dataset.T))/num_samples(dataset)
    U, s, Vh = numpy.linalg.svd(covariance_matrix)
    P = U[:, 0:m]
    DP = numpy.dot(P.T, dataset)
    return DP

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 

if __name__ == '__main__':
    dataset, labels = fingerprints_dataset.load("Train.txt");
    
    dataset_with_PCA,_ = pca(dataset,5)
    #fingerprints_dataset.plot_scatter(dataset_with_PCA, labels)







