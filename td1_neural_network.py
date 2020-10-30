#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:40:01 2020

@author: raphael
"""

#%% Imports
import pickle
import numpy as np
from random import sample

np.random.seed(1) # pour que l'exécution soit déterministe

#%% Code Julien Bronner

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
    
def kppv_predict(dist, Yapp, K): # utilisationde np.argpartition(A,k) qui donne les indices pour que jusqu'à k, on ait les valeurs les  plus petites
    N,M = np.shape(dist)
    sort_indices = np.argpartition(dist, K-1, axis = 0) #K-1 car on part de 0
    Yapp_mat = Yapp.dot(np.ones(1,M)) # pour dupliquer Yapp dans toutes les colonnes
    Yapp_mat_sort = np.take_along_axis(Yapp_mat, sort_indices, axis=0)
    Yapp_mat_sort_tronque = Yapp_mat_sort[:K, : ]
    #trouver comment avoir l'element le lus present de chaque colonne et après on aura Ypred
    return ''

#%% Chargement des données

arbo_ia = "/home/raphael/Documents/Centrale Lyon/Apprentissage profond et IA/"
data_X, label_Y = lecture_cifar(arbo_ia + "cifar-10-batches-py/data_batch_1")
Xapp, Yapp, Xtest, Ytest = decoupage_donnes(data_X, label_Y)

# dict_batch_1 = unpickle(arbo_ia + "cifar-10-batches-py/data_batch_1")
# dict_batch_2 = unpickle(arbo_ia + "cifar-10-batches-py/data_batch_2")
# dict_batch_3 = unpickle(arbo_ia + "cifar-10-batches-py/data_batch_3")
# dict_batch_4 = unpickle(arbo_ia + "cifar-10-batches-py/data_batch_4")
# dict_batch_5 = unpickle(arbo_ia + "cifar-10-batches-py/data_batch_5")

#%% Réseau de neurones

def matrice_stochastique(Y):
    # On suppose que les valeurs sont entières entre 0 et 9
    n = len(Y)
    eye_10 = np.eye(10)
    Y_stochastique = np.zeros((n, 10))
    for i in range(n):
        # Les classes sont codées sous forme de lignes dans une matrice
        # Chaque élément vaut 0, sauf en la colonne de la classe où la valeur est 1.
        Y_stochastique[i] = eye_10[Y[i]]
    return Y_stochastique

def evaluation_classifieur(Ytest, Ypred):
    """
    Parameters
    ----------
    Ytest : Array
        Matrice des classes auxquelles appartiennent les données.
    Ypred : Array
        Matrice de prédiction des classes.

    Returns
    -------
    Float
        La précision du modèle, c'est-à-dire la fraction des images correctement
        classifiées.

    """
    n = len(Ytest) # Nombre de données d'entrée
    s = np.sum(np.argmax(Ytest, axis=1) == np.argmax(Ypred, axis=1)) # Nombre de données correctement prédites
    accuracy = s/n
    return accuracy

#%% Génération des données

# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
# N, D_in, D_h, D_out = 30, 2, 10, 3
N, D_in = np.shape(Xapp)
D_h, D_out = 30, 10

# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
# X = np.random.random((N, D_in))
# Y = np.random.random((N, D_out))
X = Xapp
Y = matrice_stochastique(Yapp)

# Initialisation aléatoire des poids du réseau
W1 = 2*np.random.random((D_in, D_h)) - 1
b1 = np.zeros((1, D_h))
W2 = 2*np.random.random((D_h, D_out))-1
b2 = np.zeros((1, D_out))

for i in range(100):
    ####################################################
    # Passe avant : calcul de la sortie prédite Y_pred #
    ####################################################
    I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
    O1 = 1/(1 + np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
    O2 = 1/(1 + np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
    Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie
    
    if i == 0:
        print("O2", O2)
    ########################################################
    # Calcul et affichage de la fonction perte de type MSE #
    ########################################################
    loss = np.square(Y_pred - Y).sum()/2
    if i%10 == 0 :
        print(i, "loss : ", loss)
    # print("loss : ", loss)
    
    ########################################
    # Passe arrière : mis à jour des poids #
    ########################################
    grad_Y_pred = 2*(Y_pred - Y)
    grad_O2 = grad_Y_pred*(O2*(1 - O2))
    grad_w2 = O1.T.dot(grad_O2)
    
    grad_O1 = O1*(1 - O1)
    grad_w1 = X.T.dot((grad_O2.dot(W2.T))*grad_O1)
    
    W1 -= 1e-4*grad_w1
    W2 -= 1e-4*grad_w2

print("Accuracy : ", evaluation_classifieur(Y, O2))






