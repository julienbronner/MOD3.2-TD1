# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:48:10 2020

@author: julbr
"""

#%% Imports
import pickle
import numpy as np
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed) # pour que l'exécution soit déterministe

#%% Fonctions d'imports de données 

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

#%% Génération de données
    
# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
N, D_in, D_h, D_out = 30, 2, 10, 3

# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
X = np.random.random((N, D_in))
Y = np.random.random((N, D_out))
# Initialisation aléatoire des poids du réseau
W1 = 2 * np.random.random((D_in, D_h)) - 1
b1 = np.zeros((1,D_h))
W2 = 2 * np.random.random((D_h, D_out)) - 1
b2 = np.zeros((1,D_out))


#%% Passe avant : calcul de la sortie prédite Y_pred

I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
O2 = 1/(1+np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie


#%% Calcul et affichage de la fonction perte de type MSE

loss = np.square(Y_pred - Y).sum() / 2
print(loss)

#%% Backward

# delta_O2 = (O2-I2)
# delta_I2 = (1-O2)*O2 *delta_O2
# delta_O1 = W2 * delta_I2
# delta_W2 = O1 * delta_I2
# delta_I1 = (1-O1)*O1 *delta_O1
# delta_W1 = Xi * delta_I1
# delta_Xi = W1 * delta_I1

# db2 = np.sum(delta_I2, axis=0) envoyé, ça doit correspondre a delta b2 ?
