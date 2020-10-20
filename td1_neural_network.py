#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:40:01 2020

@author: raphael
"""

## Imports
import numpy as np

np.random.seed(1) # pour que l'exécution soit déterministe

##########################
# Chargement des données #
##########################
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict_batch_1 = unpickle("cifar-10-batches-py/data_batch_1")
dict_batch_2 = unpickle("cifar-10-batches-py/data_batch_2")
dict_batch_3 = unpickle("cifar-10-batches-py/data_batch_3")
dict_batch_4 = unpickle("cifar-10-batches-py/data_batch_4")
dict_batch_5 = unpickle("cifar-10-batches-py/data_batch_5")

########################
## Réseau de neurones ##
########################

##########################
# Génération des données #
##########################

# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
N, D_in, D_h, D_out = 30, 2, 10, 3

# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
X = np.random.random((N, D_in))
Y = np.random.random((N, D_out))

# Initialisation aléatoire des poids du réseau
W1 = 2*np.random.random((D_in, D_h)) - 1
b1 = np.zeros((1, D_h))
W2 = 2*np.random.random((D_h, D_out))-1
b2 = np.zeros((1, D_out))

####################################################
# Passe avant : calcul de la sortie prédite Y_pred #
####################################################
I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
O1 = 1/(1 + np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
O2 = 1/(1 + np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie

########################################################
# Calcul et affichage de la fonction perte de type MSE #
########################################################
loss = np.square(Y_pred - Y).sum()/2
print(loss)

########################################
# Passe arrière : mis à jour des poids #
########################################
grad_Y_pred = 2*(Y_pred - Y)
grad_O2 = grad_Y_pred*((O2*(1 - O2)))
grad_w2 = O1.T.dot(grad_O2)

grad_O1 = O1*(1 - O1)
grad_w1 = X.T.dot(grad_O2.dot(W2.T)*grad_O1)








