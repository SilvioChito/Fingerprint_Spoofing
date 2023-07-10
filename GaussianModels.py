# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:09:21 2023

@author: Utente
"""

import fingerprints_dataset
import numpy
from scipy import special

import MVG

import PCA
from LDA import compute_Sw #within-covariance matrix to compute cov.matrix of tied gaussian model

import metrics

def vcol(vector):
    return vector.reshape((vector.shape[0],1)) 

def vrow(vector):
    return vector.reshape((1,vector.shape[0])) 


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def MVG_classifier(train_set, labels, num_classes):
    model = []
    for index_class in range(num_classes):
        class_set = train_set[:,labels==index_class]
        mu_c, C_c = MVG.ML_estimate(class_set)
        model.append(mu_c)
        model.append(C_c)
    return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (media1,covmat1)...(median,covmatn)
    
def MVG_tied_classifier(train_set, labels, num_classes):
    model = []
    within_class_cov_mat = compute_Sw(train_set, labels, num_classes)
    for index_class in range(num_classes):
        class_set = train_set[:,labels==index_class]
        mu_c, _ = MVG.ML_estimate(class_set)
        model.append(mu_c)
        model.append(within_class_cov_mat)
    return model  #ritorna una lista con tutti i model parametrs per ciascuna classe (media1,covmat1)...(median,covmatn),uguale a MVG_classifier per usare lo stesso metodo di inferenza    


def NaiveBayes_classifier(train_set, labels, num_classes):
    model = MVG_classifier(train_set, labels, num_classes)
    
    for i in range(num_classes):
        model[2*i+1] = numpy.diag(numpy.diag(model[2*i+1]))
        

        
    return model 

def MVG_tied_NaiveBayes_classifier(train_set, labels, num_classes):
    model = MVG_tied_classifier(train_set, labels, num_classes)
    
    for i in range(num_classes):
        model[2*i+1] = numpy.diag(numpy.diag(model[2*i+1]))
        

        
    return model     

def predict(model, test_samples, prior): #prior is thought as a vector with every prior prob. for each class
    num_classes = len(model)//2
    likelihoods = []
    SJoint = []
    class_posterior_probs = []
    for index_class in range(num_classes):
        mu_c = model[index_class*2]
        C_c = model[index_class*2+1]
        likelihood_c = numpy.exp(MVG.logpdf_GAU_ND(test_samples, mu_c, C_c))
        likelihoods.append(likelihood_c)
        SJoint.append(likelihood_c*prior[index_class])
        
    
    
    
    SMarginal = vrow(numpy.array(SJoint).sum(0))
    SMarginalSol = numpy.load("Solution/SJoint_MVG.npy")
    print(numpy.abs(SMarginal - SMarginalSol).max())
    SPost = []
    for index_class in range(num_classes):
        SPost.append(SJoint[index_class]/SMarginal)                 
    predicted_label = numpy.argmax(numpy.array(SPost),axis=0)
    return predicted_label.ravel() #ravel per appiattire il vettore e farlo combaciare con labels per calcolo accuracy


def predict_log(model, test_samples, prior): #in the project USE THIS function to avoid numerical problems
    num_classes = len(model)//2
    likelihoods = []
    logSJoint = []
    class_posterior_probs = []
    
    for index_class in range(num_classes):
        mu_c = model[index_class*2]
        C_c = model[index_class*2+1]
        likelihood_c = MVG.logpdf_GAU_ND(test_samples, mu_c, C_c)
        likelihoods.append(likelihood_c)
        logSJoint.append(likelihood_c+numpy.log(prior[index_class]))
        
    
    
    
    
    logSMarginal = vrow(special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    SPost = numpy.exp(logSPost)    
             
    predicted_label = numpy.argmax(numpy.array(SPost),axis=0)
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(prior[1]/prior[0])
    
    #predicted_labels_bis = numpy.where(llr > 0, 1, 0)
    
    return predicted_label.ravel(),llr #ravel per appiattire il vettore e farlo combaciare con labels per calcolo accuracy

def accuracy(predicted_labels,original_labels):
    if len(predicted_labels)!=len(original_labels):
        return
    total_samples = len(predicted_labels)
    
    
    
    correct = (predicted_labels==original_labels).sum()
    return (correct/total_samples)*100


def error_rate(predicted_labels,original_labels):
    return 100 - accuracy(predicted_labels,original_labels)


def z_score_normalization(dataset):
    # Compute mean and standard deviation along the axis 0 (columns)
    mean = numpy.mean(dataset, axis=1)
    std = numpy.std(dataset, axis=1)
    
    # Apply Z-score normalization
    normalized_dataset = (dataset - vcol(mean)) / vcol(std)
    
    return normalized_dataset

def K_fold(K,train_set,labels,num_classes,prior):
    
    # Set random seed for reproducibility
    numpy.random.seed(0)
    # Create an index array
    indices = numpy.arange(train_set.shape[1])
    
    # Shuffle the index array
    numpy.random.shuffle(indices)
    # Rearrange the features and labels based on the shuffled indices
    train_set = train_set[:, indices]
    labels = labels[indices]
    
    
    fold_size = train_set.shape[1]//K
    #results = numpy.array((4,train_set.shape[1])) #per ogni riga gli score del modello testato per noi proviamo 4 modelli
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    for i in range(K):
        start = i*fold_size
        end = (i+1)*fold_size
        
        val_range = numpy.in1d(range(train_set.shape[1]),range(int(start),int(end)))
        
        curr_tr_set = train_set[:,numpy.invert(val_range)]
        curr_tr_labels = labels[numpy.invert(val_range)]
        
        curr_val_set = train_set[:,val_range]
        curr_val_labels = labels[val_range]
        
        model1 = MVG_classifier(curr_tr_set, curr_tr_labels, num_classes)
        model2 = NaiveBayes_classifier(curr_tr_set, curr_tr_labels, num_classes)
        model3 = MVG_tied_classifier(curr_tr_set, curr_tr_labels, num_classes)
        model4 = MVG_tied_NaiveBayes_classifier(curr_tr_set, curr_tr_labels, num_classes)
        
        
        pred_labels1,llr1 = predict_log(model1, curr_val_set, prior)
        pred_labels2,llr2 = predict_log(model2, curr_val_set, prior)
        pred_labels3,llr3 = predict_log(model3, curr_val_set, prior)
        pred_labels4,llr4 = predict_log(model4, curr_val_set, prior)
        
        '''
        acc1 = accuracy(pred_labels1, curr_val_labels)
        acc2 = accuracy(pred_labels2, curr_val_labels)
        acc3 = accuracy(pred_labels3, curr_val_labels)
        acc4 = accuracy(pred_labels4, curr_val_labels)
        '''
        
        
        results1.append(llr1)
        results2.append(llr2)
        results3.append(llr3)
        results4.append(llr4)
        
        
    #results = numpy.array(results)    
    #results = results.reshape((K,4))    #sulle righe l'esito della validation per il k_esimo-fold per ciascun modello 
    
    #results_per_model = results.mean(0) #chiedi al prof se la logica è corretta
    
    
    #best_model_id = numpy.argmax(results_per_model)
    
    
    dcfmin1 = metrics.compute_DCFmin(numpy.concatenate(results1), labels[0:2320], prior[1], 1, 1)
    dcfmin2 = metrics.compute_DCFmin(numpy.concatenate(results2), labels[0:2320], prior[1], 1, 1)
    dcfmin3 = metrics.compute_DCFmin(numpy.concatenate(results3), labels[0:2320], prior[1], 1, 1)
    dcfmin4 = metrics.compute_DCFmin(numpy.concatenate(results4), labels[0:2320], prior[1], 1, 1)
    
    #predicted_labels = numpy.where(numpy.concatenate(results1) > -1*numpy.log(prior[1]/prior[0]), 1, 0)
    #print(error_rate(predicted_labels, labels[0:2320]))
    print('MVG = '+str(dcfmin1))
    print('NaiveBayes = '+str(dcfmin2))
    print('Tied = '+str(dcfmin3))
    print('Tied Naive Bayes= '+str(dcfmin4))
    

if __name__ == '__main__':
        
    DTR, LTR = fingerprints_dataset.load("Train.txt")
    DTE, LTE = fingerprints_dataset.load("Test.txt")
        
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    #(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    '''
    model = MVG_classifier(DTR, LTR, num_classes=2)
    
    
    predicted_labels,llr = predict_log(model, DTE, prior=[1-1/11,1/11])
    dcf_min = metrics.compute_DCFmin(llr, LTE, 1/11, 1, 1)
    cm = metrics.compute_confusion_matrix(LTE, predicted_labels, num_classes=2)
    dcf=metrics.compute_DCF(1/11, 1, 1, cm)
    print(dcf)
    metrics.plot_Bayes_error(llr, LTE)
    
    '''
    #acc = accuracy(predicted_labels,LTE)
    #err = error_rate(predicted_labels,LTE)
    #print(err)
    #print(LTR.shape)
    eff_prior=1/11 #(0.5,cfn=1,cfp=10)
    
    
    #DTR_PCA = PCA.pca(DTR, 10)
    DTR_norm = z_score_normalization(DTR)
    DTR_PCA,_ = PCA.pca(DTR_norm, 9)
    K_fold(10,DTR_PCA, LTR, 2, prior = [1-eff_prior,eff_prior])
    
    #predicted_labels = predict_log(best_val_model, DTE, prior=[0.5,0.5])
    #err = error_rate(predicted_labels,LTE)
    #print(err)
    
    
    
    #qua poi dopo k-fold traino il modello che mi è stato consigliato con TUTTI i training data
    '''
    model = MVG_classifier(DTR, LTR, num_classes=2)
    predicted_labels = predict_log(model, DTE, prior=[0.5,0.5])
    err = error_rate(predicted_labels,LTE)
    print(err)
    '''
    '''
    model = NaiveBayes_classifier(DTR, LTR, num_classes=2)
    predicted_labels = predict_log(model, DTE, prior=[0.5,0.5])
    err = error_rate(predicted_labels,LTE)
    print(err)
    '''
    
    '''
    model = MVG_tied_NaiveBayes_classifier(DTR, LTR, num_classes=2)
    predicted_labels = predict_log(model, DTE, prior=[0.5,0.5])
    err = error_rate(predicted_labels,LTE)
    print(err)
    '''
    