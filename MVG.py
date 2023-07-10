# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:14:07 2023

@author: Utente
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt
import math


  

def dataset_mean(dataset):
    return dataset.mean(1).reshape((dataset.shape[0],1)) #dataset.shape[0] è numero features per rendere codice generico

def num_samples(dataset):
    return dataset.shape[1];

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 

def logpdf_GAU_ND_single_sample(x, mu, C): #here x is a single sample
    _, log_det_cov_mat = numpy.linalg.slogdet(C)
    inv_cov_mat = numpy.linalg.inv(C)
    elem = numpy.dot((x-mu).T,numpy.dot(inv_cov_mat,(x-mu)))
    
    return -(mu.shape[0]/2)*math.log(2*math.pi)-(1/2)*(log_det_cov_mat)-(1/2)*elem

def logpdf_GAU_ND(X, mu, C): #here X is a set of samples
    Y = []
    for index_sample in range(X.shape[1]):
        y = logpdf_GAU_ND_single_sample(vcol(X[:,index_sample]), mu, C) #scorro tutti i samples e per ciascuno ci calcolo la log-likelihood
        Y.append(y)
    return numpy.array(Y)[:,0,0]        
        
def ML_estimate(X):
    mu_ml = dataset_mean(dataset=X)
    covariance_matrix_ml = (numpy.dot(((X-mu_ml)), (X-mu_ml).T))/num_samples(X)
    
    return mu_ml,covariance_matrix_ml

def loglikelihood(X,m_ML,C_ML):
    return numpy.sum(logpdf_GAU_ND(X, m_ML, C_ML))
    






    