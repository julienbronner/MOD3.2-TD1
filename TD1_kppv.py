# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:26:18 2020

@author: julbr
"""
import pickle
import numpy as np

seed = 2
np.random.seed(seed) # pour que l'exécution soit déterministe

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
    np.random.seed(seed)
    n = len(X)
    nbr_aleatoires_test = np.random.choice(n, int(n/5), replace=False)
    # nbr_aleatoires_test = sample(range(0,n), int(n/5))
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
    
def kppv_predict(dist, Yapp, K): # utilisationde np.argpartition(A,k) qui donne les indices pour que jusqu'à k, on ait les valeurs les  plus petites
    N,M = np.shape(dist)
    sort_indices = np.argpartition(dist, K-1, axis = 0) #K-1 car on part de 0
    Yapp_mat = Yapp.dot(np.ones(1,M)) # pour dupliquer Yapp dans toutes les colonnes
    Yapp_mat_sort = np.take_along_axis(Yapp_mat, sort_indices, axis=0)
    Yapp_mat_sort_tronque = Yapp_mat_sort[:K, : ]
    #trouver comment avoir l'element le plus present de chaque colonne et après on aura Ypred
    return ''
    