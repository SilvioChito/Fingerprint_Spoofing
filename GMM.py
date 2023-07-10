# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 19:26:55 2023

@author: Utente
"""

import sklearn.datasets
import matplotlib as plt

from MVG import *
import fingerprints_dataset
import metrics
import PCA

import numpy as np
import scipy


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 

def logpdf_GMM(X, gmm): #gmm list of tuple
    num_components = len(gmm)
    logS = np.empty((num_components, X.shape[1]))
    
    for component_index in range(num_components):
        logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
        logS[component_index, :] += np.log(gmm[component_index][0]) #computing the joint loglikelihood densities
    logSMarginal = scipy.special.logsumexp(logS, axis=0)
    
    return logSMarginal

def EM_GMM(X, gmm,psi): #gmm list of tuple
    while True:
        num_components = len(gmm)
        logS = np.empty((num_components, X.shape[1]))
        #---START of E step---
        for component_index in range(num_components):
            logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
            logS[component_index, :] += np.log(gmm[component_index][0]) #computing the joint loglikelihood densities
        logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
        logSPost = logS - logSMarginal
        SPost = numpy.exp(logSPost)
        
        old_loss = logSMarginal
        #---END of E step---
        #---START of M step---
        Z = np.sum(SPost,axis=1)
        
        
        
        F = np.zeros((X.shape[0],num_components))
        for component_index in range(num_components):
            F[:,component_index] = np.sum(SPost[component_index,:]*X,axis=1)
            
        S = np.zeros((X.shape[0],X.shape[0],num_components))
        for component_index in range(num_components):
            S[:,:,component_index] = np.dot(SPost[component_index,:]*X,X.T)
            
        mu = F/Z
        cov_mat = S/Z 
        
        for component_index in range(num_components):
            cov_mat[:,:,component_index] -= np.dot(vcol(mu[:,component_index]),vrow(mu[:,component_index]))
            U, s, _ = np.linalg.svd(cov_mat[:,:,component_index]) #avoiding degenerate solutions
            s[s<psi] = psi
            cov_mat[:,:,component_index] = np.dot(U, vcol(s)*U.T) #what is mcol?
        w = Z/np.sum(Z)
        
        
        
        
        gmm = [(w[index_component], vcol(mu[:,index_component]), cov_mat[:,:,index_component]) for index_component in range(num_components)]
        new_loss = logpdf_GMM(X, gmm)
        #---END of M step---
        if np.mean(new_loss)-np.mean(old_loss)>0:
            print("Bene")
        if np.mean(new_loss)-np.mean(old_loss)<0:
            print("Male")
        
        if np.mean(new_loss)-np.mean(old_loss)<1e-6:
            print("EM terminato")
            break
    print(np.mean(new_loss))
    return gmm 

def EM_Diag_GMM(X, gmm,psi): #gmm list of tuple
    while True:
        num_components = len(gmm)
        logS = np.empty((num_components, X.shape[1]))
        #---START of E step---
        for component_index in range(num_components):
            logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
            logS[component_index, :] += np.log(gmm[component_index][0]) #computing the joint loglikelihood densities
        logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
        logSPost = logS - logSMarginal
        SPost = numpy.exp(logSPost)
        
        old_loss = logSMarginal
        #---END of E step---
        #---START of M step---
        Z = np.sum(SPost,axis=1)
        
        
        
        F = np.zeros((X.shape[0],num_components))
        for component_index in range(num_components):
            F[:,component_index] = np.sum(SPost[component_index,:]*X,axis=1)
            
        S = np.zeros((X.shape[0],X.shape[0],num_components))
        for component_index in range(num_components):
            S[:,:,component_index] = np.dot(SPost[component_index,:]*X,X.T)
            
        mu = F/Z
        cov_mat = S/Z 
        
        for component_index in range(num_components):
            cov_mat[:,:,component_index] -= np.dot(vcol(mu[:,component_index]),vrow(mu[:,component_index]))
            
        w = Z/np.sum(Z)
        
        for index_component in range(num_components):
            cov_mat[:,:,index_component] = np.diag(np.diag(cov_mat[:,:,index_component])) #diag cov mat
            U, s, _ = np.linalg.svd(cov_mat[:,:,index_component]) #avoiding degenerate solutions
            s[s<psi] = psi
            cov_mat[:,:,index_component] = np.dot(U, vcol(s)*U.T)
            
            
        gmm = [(w[index_component], vcol(mu[:,index_component]), cov_mat[:,:,index_component]) for index_component in range(num_components)]
        new_loss = logpdf_GMM(X, gmm)
        #---END of M step---
        if np.mean(new_loss)-np.mean(old_loss)>0:
            print("BENE")
        if np.mean(new_loss)-np.mean(old_loss)<0:
            print("MALE")    
        if np.mean(new_loss)-np.mean(old_loss)<1e-6:
            print("EM terminato")
            break
    print(np.mean(new_loss))
    return gmm     

def EM_TiedCov_GMM(X, gmm,psi): #gmm list of tuple
    while True:
        num_components = len(gmm)
        logS = np.empty((num_components, X.shape[1]))
        #---START of E step---
        for component_index in range(num_components):
            logS[component_index, :] = logpdf_GAU_ND(X, gmm[component_index][1], gmm[component_index][2])
            logS[component_index, :] += np.log(gmm[component_index][0]) #computing the joint loglikelihood densities
        logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
        logSPost = logS - logSMarginal
        SPost = numpy.exp(logSPost)
        
        old_loss = logSMarginal
        #---END of E step---
        #---START of M step---
        Z = np.sum(SPost,axis=1)
        
        
        
        F = np.zeros((X.shape[0],num_components))
        for component_index in range(num_components):
            F[:,component_index] = np.sum(SPost[component_index,:]*X,axis=1)
            
        S = np.zeros((X.shape[0],X.shape[0],num_components))
        for component_index in range(num_components):
            S[:,:,component_index] = np.dot(SPost[component_index,:]*X,X.T)
            
        mu = F/Z
        cov_mat = S/Z 
        
        for component_index in range(num_components):
            cov_mat[:,:,component_index] -= np.dot(vcol(mu[:,component_index]),vrow(mu[:,component_index]))
        w = Z/np.sum(Z)
        
        
        cov_updated = np.zeros( cov_mat[:,:,0].shape)
        for component_index in range(num_components):
            cov_updated += (1/num_samples(X)) * (Z[component_index]*cov_mat[:,:,component_index])
            
        U, s, _ = np.linalg.svd(cov_updated) #avoiding degenerate solutions
        s[s<psi] = psi
        cov_updated = np.dot(U, vcol(s)*U.T)
        gmm = [(w[index_component], vcol(mu[:,index_component]), cov_updated) for index_component in range(num_components)]
        
        new_loss = logpdf_GMM(X, gmm)
        #---END of M step---
        if np.mean(new_loss)-np.mean(old_loss)>0:
            print("BENE")
        if np.mean(new_loss)-np.mean(old_loss)<0:
            print("MALE")      
        if np.mean(new_loss)-np.mean(old_loss)<1e-6:
            print("EM terminato")
            break
    print(np.mean(new_loss))
    return gmm    

def LBG(X,psi,alpha=0.1,num_components_max=4):
    w_start = 1
    mu_start,C_start = ML_estimate(X)
    
    U, s, _ = np.linalg.svd(C_start) #avoiding degenerate solutions
    s[s<psi] = psi
    C_start = np.dot(U, vcol(s)*U.T) #what is mcol
    
        
    gmm_start = []
    if num_components_max==1:
        gmm_start.append((w_start,vcol(mu_start),C_start))
        return gmm_start
    new_w = w_start/2
    U, s, Vh = np.linalg.svd(C_start)
    d = U[:, 0:1] * s[0]**0.5 * alpha
    mu_new_1 = mu_start+d
    mu_new_2 = mu_start-d
    gmm_start.append((new_w,vcol(mu_new_1),C_start))
    gmm_start.append((new_w,vcol(mu_new_2),C_start))
    
    while True:
        gmm_start = EM_GMM(X, gmm_start,psi)
        num_components = len(gmm_start)
        
           
            
        if num_components==num_components_max:
            break
        new_gmm = []
        for index_component in range(num_components):
            new_w = gmm_start[index_component][0]/2
            U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            mu_new_1 = gmm_start[index_component][1]+d
            mu_new_2 = gmm_start[index_component][1]-d
            new_gmm.append((new_w,vcol(mu_new_1),gmm_start[index_component][2]))
            new_gmm.append((new_w,vcol(mu_new_2),gmm_start[index_component][2]))
        gmm_start = new_gmm    
    return gmm_start

def Diag_LBG(X,psi,alpha=0.1,num_components_max=4):
    w_start = 1
    mu_start,C_start = ML_estimate(X)
    
    C_start = np.diag(np.diag(C_start))
    
    U, s, _ = np.linalg.svd(C_start) #avoiding degenerate solutions
    s[s<psi] = psi
    C_start = np.dot(U, vcol(s)*U.T)

    gmm_start = []
    
    if num_components_max==1:
        gmm_start.append((w_start,vcol(mu_start),C_start))
        return gmm_start
    
    new_w = w_start/2
    U, s, Vh = np.linalg.svd(C_start)
    d = U[:, 0:1] * s[0]**0.5 * alpha
    mu_new_1 = mu_start+d
    mu_new_2 = mu_start-d
    gmm_start.append((new_w,vcol(mu_new_1),C_start))
    gmm_start.append((new_w,vcol(mu_new_2),C_start))
    
    while True:
        gmm_start = EM_Diag_GMM(X, gmm_start,psi)
        num_components = len(gmm_start)
        
        
            
            
            
        if num_components==num_components_max:
            break
        new_gmm = []
        for index_component in range(num_components):
            new_w = gmm_start[index_component][0]/2
            U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            mu_new_1 = gmm_start[index_component][1]+d
            mu_new_2 = gmm_start[index_component][1]-d
            new_gmm.append((new_w,vcol(mu_new_1),gmm_start[index_component][2]))
            new_gmm.append((new_w,vcol(mu_new_2),gmm_start[index_component][2]))
        gmm_start = new_gmm 
        print("GMMMM START....")
        print(len(gmm_start))
    return gmm_start

def TiedCov_LBG(X,psi,alpha=0.1,num_components_max=4):
    w_start = 1
    mu_start,C_start = ML_estimate(X)
    
    U, s, _ = np.linalg.svd(C_start) #avoiding degenerate solutions
    s[s<psi] = psi
    C_start = np.dot(U, vcol(s)*U.T)

    gmm_start = []
    
    if num_components_max==1:
        gmm_start.append((w_start,vcol(mu_start),C_start))
        return gmm_start
    
    new_w = w_start/2
    U, s, Vh = np.linalg.svd(C_start)
    d = U[:, 0:1] * s[0]**0.5 * alpha
    mu_new_1 = mu_start+d
    mu_new_2 = mu_start-d
    gmm_start.append((new_w,vcol(mu_new_1),C_start))
    gmm_start.append((new_w,vcol(mu_new_2),C_start))
    while True:
        gmm_start = EM_TiedCov_GMM(X, gmm_start, psi)
        num_components = len(gmm_start)
        
        
        
            
            
        if num_components==num_components_max:
            break
        new_gmm = []
        for index_component in range(num_components):
            new_w = gmm_start[index_component][0]/2
            U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            mu_new_1 = gmm_start[index_component][1]+d
            mu_new_2 = gmm_start[index_component][1]-d
            new_gmm.append((new_w,vcol(mu_new_1),gmm_start[index_component][2]))
            new_gmm.append((new_w,vcol(mu_new_2),gmm_start[index_component][2]))
        gmm_start = new_gmm    
    return gmm_start


def GMM_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    model = []
    for index_class in range(num_classes):
        class_set = train_set[:,labels==index_class]
        gmm_c = LBG(class_set,psi,alpha,num_components_max[index_class])
        model.append(gmm_c)
        
    return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

def GMM_Diag_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    model = []
    for index_class in range(num_classes):
        class_set = train_set[:,labels==index_class]
        gmm_c = Diag_LBG(class_set,psi,alpha,num_components_max[index_class])
        model.append(gmm_c)
        
    return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

def GMM_TiedCov_classifier(train_set, labels, num_classes,psi=0.01,alpha=0.1,num_components_max=[4,4]):
    model = []
    for index_class in range(num_classes):
        class_set = train_set[:,labels==index_class]
        gmm_c = TiedCov_LBG(class_set,psi,alpha,num_components_max[index_class])
        model.append(gmm_c)
        
    return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (gmm1)...(gmm-n)

def predict_log(model, test_samples, prior): #in the project USE THIS function to avoid numerical problems
    num_classes = len(model)
    likelihoods = []
    logSJoint = []
    class_posterior_probs = []
    for index_class in range(num_classes):
        gmm_c = model[index_class]
        likelihood_c = logpdf_GMM(test_samples,gmm_c)
        likelihoods.append(likelihood_c)
        logSJoint.append(likelihood_c+np.log(prior[index_class]))
        
    
    
    
    
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - np.log(prior[1]/prior[0])
    SPost = np.exp(logSPost)    
             
    predicted_label = np.argmax(np.array(SPost),axis=0)
    
    return predicted_label.ravel(),llr #ravel per appiattire il vettore e farlo combaciare con labels per calcolo accuracy

def accuracy(predicted_labels,original_labels):
    if len(predicted_labels)!=len(original_labels):
        return
    total_samples = len(predicted_labels)
    
    
    
    correct = (predicted_labels==original_labels).sum()
    return (correct/total_samples)*100


def error_rate(predicted_labels,original_labels):
    return 100 - accuracy(predicted_labels,original_labels)

def K_fold(K,train_set,labels,num_classes,prior):
    
    # Set random seed for reproducibility
    np.random.seed(0)
    # Create an index array
    indices = np.arange(train_set.shape[1])
    
    # Shuffle the index array
    np.random.shuffle(indices)
    # Rearrange the features and labels based on the shuffled indices
    train_set = train_set[:, indices]
    labels = labels[indices]
    
    
    fold_size = train_set.shape[1]//K
    #results = numpy.array((4,train_set.shape[1])) #per ogni riga gli score del modello testato per noi proviamo 4 modelli
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    results5 = []
    results6 = []    
    preds = []
    for i in range(K):
        print("This is the "+str(i+1)+"-th FOLD")
        start = i*fold_size
        end = (i+1)*fold_size
        
        val_range = np.in1d(range(train_set.shape[1]),range(int(start),int(end)))
        
        curr_tr_set = train_set[:,np.invert(val_range)]
        curr_tr_labels = labels[np.invert(val_range)]
        
        curr_val_set = train_set[:,val_range]
        curr_val_labels = labels[val_range]
        
        model1 = GMM_TiedCov_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[2,1])
        #model2 = GMM_Diag_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[4,2])
        #model3 = GMM_TiedCov_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[4,2])
        #model4 = GMM_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[8,2])
        #model5 = GMM_Diag_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[8,2])
        #model6 = GMM_TiedCov_classifier(curr_tr_set, curr_tr_labels, num_classes,num_components_max=[8,2])
        
        
        
        
        pred_labels1,llr1 = predict_log(model1, curr_val_set, prior)
        #pred_labels2,llr2 = predict_log(model2, curr_val_set, prior)
        #pred_labels3,llr3 = predict_log(model3, curr_val_set, prior)
        #pred_labels4,llr4 = predict_log(model4, curr_val_set, prior)
        #pred_labels5,llr5 = predict_log(model5, curr_val_set, prior)
        #pred_labels6,llr6 = predict_log(model6, curr_val_set, prior)
        
        
        '''
        acc1 = accuracy(pred_labels1, curr_val_labels)
        acc2 = accuracy(pred_labels2, curr_val_labels)
        acc3 = accuracy(pred_labels3, curr_val_labels)
        acc4 = accuracy(pred_labels4, curr_val_labels)
        '''
        
        
        results1.append(llr1)
        preds.append(pred_labels1)
        #results2.append(llr2)
        #results3.append(llr3)
        #results4.append(llr4)
        #results5.append(llr5)
        #results6.append(llr6)
        
        
        
    #results = numpy.array(results)    
    #results = results.reshape((K,4))    #sulle righe l'esito della validation per il k_esimo-fold per ciascun modello 
    
    #results_per_model = results.mean(0) #chiedi al prof se la logica è corretta
    
    
    #best_model_id = numpy.argmax(results_per_model)
    
    confusion_matrix = metrics.compute_confusion_matrix(labels[0:2320], np.concatenate(preds), num_classes)
    dcf1 = metrics.compute_DCF(1/11, 1, 1, confusion_matrix)
    metrics.plot_Bayes_error(np.concatenate(results1), labels[0:2320])
    #dcfmin2 = metrics.compute_DCFmin(np.concatenate(results2), labels[0:2320], prior[1], 1, 1)
    #dcfmin3 = metrics.compute_DCFmin(np.concatenate(results3), labels[0:2320], prior[1], 1, 1)
    #dcfmin4 = metrics.compute_DCFmin(np.concatenate(results4), labels[0:2320], prior[1], 1, 1)
    #dcfmin5 = metrics.compute_DCFmin(np.concatenate(results5), labels[0:2320], prior[1], 1, 1)
    #dcfmin6 = metrics.compute_DCFmin(np.concatenate(results6), labels[0:2320], prior[1], 1, 1)
    
    
    #predicted_labels = numpy.where(numpy.concatenate(results1) > -1*numpy.log(prior[1]/prior[0]), 1, 0)
    #print(error_rate(predicted_labels, labels[0:2320]))
    print('GMM DCF = '+str(dcf1))
    
    

if __name__ == '__main__':
    DTR, LTR = fingerprints_dataset.load("Train.txt")
    DTE, LTE = fingerprints_dataset.load("Test.txt")
    '''
    model = GMM_classifier(DTR, LTR, num_classes=2,num_components_max=2)
    preds = predict_log(model, DTE, prior=[0.5, 0.5])
    print(preds)
    print(LTE)
    print(error_rate(preds, LTE))'''
    DTR,P = PCA.pca(DTR, 9)
    DTE = np.dot(P.T, DTE)
    
    eff_prior=1/11 #(0.5,cfn=1,cfp=10)
    '''
    model1 = GMM_classifier(DTR, LTR, num_classes=2,num_components_max=[8,1])
    model2 = GMM_Diag_classifier(DTR, LTR, num_classes=2,num_components_max=[8,1])
    model3 = GMM_TiedCov_classifier(DTR, LTR, num_classes=2,num_components_max=[8,1])
    
    pred_labels1,llr1 = predict_log(model1, DTE, [1-eff_prior,eff_prior])
    confusion_matrix1 = metrics.compute_confusion_matrix(LTE, pred_labels1, num_classes=2)
    
    dcf1 = metrics.compute_DCF(1/11, 1, 1, confusion_matrix1)
    print("-------")
    print("DCF GMM: "+str(dcf1))
    
    dcfmin1 = metrics.compute_DCFmin(llr1, LTE, 1/11, 1, 1)
    print("MIN DCF GMM: "+str(dcfmin1))
    #----------------------------------------
    
    
    pred_labels2,llr2 = predict_log(model2, DTE, [1-eff_prior,eff_prior])
    confusion_matrix2 = metrics.compute_confusion_matrix(LTE, pred_labels2, num_classes=2)
    
    dcf2 = metrics.compute_DCF(1/11, 1, 1, confusion_matrix2)
    print("-------")
    print("DCF DIAG: "+str(dcf2))
    
    dcfmin2 = metrics.compute_DCFmin(llr2, LTE, 1/11, 1, 1)
    print("MIN DCF DIAG: "+str(dcfmin2))
    #-----------------------------------------
    
    
    pred_labels3,llr3 = predict_log(model3, DTE, [1-eff_prior,eff_prior])
    confusion_matrix3 = metrics.compute_confusion_matrix(LTE, pred_labels3, num_classes=2)
    
    dcf3 = metrics.compute_DCF(1/11, 1, 1, confusion_matrix3)
    print("-------")
    print("DCF TIED: "+str(dcf3))
    
    dcfmin3 = metrics.compute_DCFmin(llr3, LTE, 1/11, 1, 1)
    print("MIN DCF TIED: "+str(dcfmin3))
    '''
    K_fold(10, DTR, LTR, num_classes=2, prior=[1-eff_prior,eff_prior])
    