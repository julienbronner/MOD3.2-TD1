# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:26:18 2020

@author: julbr
"""
import pickle
import numpy as np
from random import sample

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def lecture_cifar(path):
    dico_tot = unpickle(path)
    data_X = np.array(dico_tot[b'data'])
    data_X = data_X.astype(float)
    label_Y = np.array(dico_tot[b'labels'])
    return data_X, label_Y

def decoupage_donnes(X, Y):
    n = len(X)
    nbr_aleatoires_test = sample(range(0,n), int(n/5))
    nbr_app = list(set(range(0,n)) - set(nbr_aleatoires_test))
    Xtest = np.take(X, nbr_aleatoires_test, 0)
    Ytest = np.take(Y, nbr_aleatoires_test, 0)
    Xapp = np.take(X, nbr_app, 0)
    Yapp = np.take(Y, nbr_app, 0)
    return Xapp, Yapp, Xtest, Ytest

def kppv_distances(Xtest, Xapp): #ça tourne mais j'espère que c'est juste x)
    N = np.shape(Xapp)[0]
    M = np.shape(Xtest)[0]
    
    diag_xapp = np.diag(Xapp.dot(np.transpose(Xapp)))
    diag_xapp = np.reshape(diag_xapp, (N,1))
    mat_ligne_m = np.ones((1,M))
    terme1_somme = diag_xapp.dot(mat_ligne_m)
    
    diag_xtest = np.diag(Xtest.dot(np.transpose(Xtest)))
    diag_xtest = np.reshape(diag_xtest, (1,M))
    mat_colonne_n = np.ones((N,1))
    terme2_somme = mat_colonne_n.dot(diag_xtest)
    
    terme3_somme = Xapp.dot(np.transpose(Xtest))
    
    dist = terme1_somme + terme2_somme - 2*terme3_somme
    
    return dist
    
    