# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:01:57 2023

@author: Utente
"""

from sklearn.datasets import load_iris
import GaussianModels

import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def whitening(data):
    # Compute the covariance matrix
    covariance = np.cov(data)
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Compute the whitening matrix
    whitening_matrix = np.dot(np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))), eigenvectors.T)
    # Whiten the data
    whitened_data = np.dot(whitening_matrix, data)
    return whitened_data

def z_score_normalization(dataset):
    mean = np.mean(dataset, axis=1)
    std = np.std(dataset, axis=1)
    normalized_dataset = (dataset - mcol(mean)) / mcol(std)
    return normalized_dataset

#-------

def mcol(v):
    return v.reshape((v.size, 1))

def load_iris_bis():
    D, L = load_iris()['data'].T, load_iris() ['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 1.0 / 10.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTC = D[:, idxTrain]
    DTR = D[:, idxTest]
    LTC = L[idxTrain]
    LTR = L[idxTest]
    return (DTR, LTR), (DTC, LTC)



def compute_confusion_matrix(actual, predicted, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(confusion_matrix, (predicted, actual), 1)
    return confusion_matrix

def compute_results_with_diff_triplet(llr,p1,c1,c2):
    t = -np.log((p1*c1)/((1-p1)*c2))
    preds = llr > t
    
    return preds.astype(int)   

def compute_DCFun(p1,cfn,cfp,confusion_matrix):
    FNR = confusion_matrix[0,1]/(confusion_matrix[0,1]+confusion_matrix[1,1])
    FPR = confusion_matrix[1,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
    return p1*cfn*FNR + (1-p1)*cfp*FPR
def compute_DCF(p1,cfn,cfp,confusion_matrix):
    
    return compute_DCFun(p1, cfn, cfp, confusion_matrix)/min(p1*cfn,(1-p1)*cfp)

def compute_DCFmin(scores,labels,p1,cfn,cfp):
    thresholds = sorted(scores)
    DCF_list = []
    for t in thresholds:
        predicted_labels = scores > t
        predicted_labels = predicted_labels.astype(int)
        confusion_matrix = compute_confusion_matrix(labels, predicted_labels, num_classes=2)
        curr_DCF = compute_DCF(p1, cfn, cfp, confusion_matrix)
        DCF_list.append(curr_DCF)
    DCFmin = min(DCF_list)
    return DCFmin    

def plot_ROC(scores,labels):
    thresholds = sorted(scores)
    FPRs = []
    TPRs = []
    
    for t in thresholds:
        predicted_labels = scores > t
        predicted_labels = predicted_labels.astype(int)
        confusion_matrix = compute_confusion_matrix(labels, predicted_labels, num_classes=2)
        TPR = 1 - confusion_matrix[0,1]/(confusion_matrix[0,1]+confusion_matrix[1,1])
        FPR = confusion_matrix[1,0]/(confusion_matrix[1,0]+confusion_matrix[0,0])
        FPRs.append(FPR)
        TPRs.append(TPR)
    # Plotting the ROC curve
    plt.plot(FPRs, TPRs, marker='*')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Plotting the diagonal (random guess)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show() 
    
def plot_Bayes_error(scores,labels):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    DCFs = []
    minDCFs = []
    
    for p in effPriorLogOdds:
        eff_p = 1/(1+math.exp(-p))
        preds = compute_results_with_diff_triplet(scores, eff_p, 1, 1)
        confusion_matrix = compute_confusion_matrix(labels, preds, num_classes = 2)
        DCFs.append(compute_DCF(eff_p, 1, 1, confusion_matrix))
        minDCFs.append(compute_DCFmin(scores, labels, eff_p, 1, 1))
        
    plt.plot(effPriorLogOdds, DCFs, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title('Bayes Error')
    plt.show() 
      
        

if __name__ == '__main__':
    #D, L = load_iris_bis()
        
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    #(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    #model = GaussianModels.MVG_tied_classifier(DTR, LTR, num_classes = 3)
    #predicted_labels = GaussianModels.predict_log(model, DTE, prior=[1/3,1/3,1/3])
    
    #ll = np.load("commedia_ll.npy")
    #labels = np.load("commedia_labels.npy")
    #logPrior = np.log( np.ones(ll.shape[0]) / float(ll.shape[0]) )
    #J = ll + mcol(logPrior) # Compute joint probability
    #ll = scipy.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    #P = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    #P = np.exp(P)
    #predicted_labels = np.argmax(P, axis=0)
    
    llr = np.load("commedia_llr_infpar.npy")
    labels = np.load("commedia_labels_infpar.npy")    
    predicted_labels = compute_results_with_diff_triplet(llr, 0.8, 1, 10)
    
    cm = compute_confusion_matrix(labels, predicted_labels, num_classes=2)
    DCFmin = compute_DCFmin( llr,labels,0.5, 1, 1)
    #plot_ROC(llr, labels)
    plot_Bayes_error(llr, labels)
    