#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:40:01 2020

@author: raphael
"""

#%% Imports
import pickle
import numpy as np
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed) # pour que l'exécution soit déterministe

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
    """
    Parameters
    ----------
    Y : Array
        Vecteur de taille (N,1) contenant les classes comprises entre 0 et 9.

    Returns
    -------
    Y_stochastique : Array
        Matrice de taille (N, 10). Chaque classe est codée sous la forme d'un
        vecteur de taille 10, composé de 0 et d'un 1 à l'indice de la classe.
        Par exemple, la classe 6 est codée par le vecteur [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.].

    """
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

def sigmoid(X, W):
    I = X.dot(W) # Potentiel d'entrée de la couche
    O = np.where(I >= 0, 1/(1 + np.exp(-I)), np.exp(I)/(1 + np.exp(I)))
    # O = 1/(1 + np.exp(-I)) # Sortie de la couche (fonction d'activation de type sigmoïde)
    return O

def d_sigmoid(X):
    return X*(1 - X)

def neural_network_donnees_aleatoires():
    np.random.seed(seed)
    
    # N est le nombre de données d'entrée
    # D_in est la dimension des données d'entrée
    # D_h le nombre de neurones de la couche cachée
    # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
    N, D_in, D_h, D_out = 30, 2, 10, 3
    
    nb_iterations = 100000
    gradient_step = 1e-3
    
    # Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
    X = np.random.random((N,D_in))
    Y = np.random.random((N,D_out))
    
    # Vecteurs pour mesurer la performance du modèle
    vect_loss = np.zeros(nb_iterations)
    vect_accuracy = np.zeros(nb_iterations)
    
    # Initialisation aléatoire des poids du réseau
    W1 = 2*np.random.random((D_in, D_h)) - 1
    W1 = np.array(W1, dtype="float32")
    b1 = np.zeros((1, D_h), dtype="float32")
    W2 = 2*np.random.random((D_h, D_out))-1
    W2 = np.array(W2, dtype="float32")
    b2 = np.zeros((1, D_out), dtype="float32")
    
    for i in range(nb_iterations):
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
        O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
        O2 = 1/(1+np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie
            
        ########################################################
        # Calcul et affichage de la fonction perte de type MSE #
        ########################################################
        loss = np.square(Y_pred - Y).sum()/2
        vect_loss[i] = loss
        
        ######################################################
        # Calcul et affichage de la précision du classifieur #
        ######################################################
        accuracy = evaluation_classifieur(Y, Y_pred)
        vect_accuracy[i] = accuracy
        
        ########################################
        # Passe arrière : mis à jour des poids #
        ########################################
        grad_Y_pred = 2*(Y_pred - Y)
        grad_O2 = grad_Y_pred*d_sigmoid(O2)
        grad_w2 = O1.T.dot(grad_O2)
        
        grad_O1 = d_sigmoid(O1)
        grad_w1 = X.T.dot((grad_O2.dot(W2.T))*grad_O1)
        
        W1 -= gradient_step*grad_w1
        W2 -= gradient_step*grad_w2
    
    vect_accuracy*= 100
    
    #######################################
    # Afficher les performances du modèle #
    #######################################
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    fig.suptitle("Fonction de perte en fonction des itérations")
    ax1.set_xlabel(r'Itération $i$')
    ax1.set_ylabel("Perte")
    ax1.grid()
    ax1.plot(vect_loss)
    
    ax2 = fig.add_subplot(212)
    fig.suptitle("Précision en fonction des itérations")
    ax2.set_xlabel(r'Itération $i$')
    ax2.set_ylabel("Précision")
    ax2.grid()
    ax2.plot(vect_accuracy)
    
    return Y_pred, W1, W2, vect_loss, vect_accuracy

def neural_network_classification_cifar_10_train(Xapp,
                                                 Yapp,
                                                 gradient_step=1e-3,
                                                 nb_iterations = 1000):
    
    np.random.seed(seed)
    
    # N est le nombre de données d'entrée
    # D_in est la dimension des données d'entrée
    # D_h le nombre de neurones de la couche cachée
    # D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
    N, D_in = np.shape(Xapp)
    D_h, D_out = 20, 10
    
    # Création d'une matrice d'entrée X et de sortie Y
    
    X = np.array(Xapp/255, dtype="float32") #Normalisation des données
    Y = matrice_stochastique(Yapp)
    
    # Vecteurs pour mesurer la performance du modèle
    vect_loss = np.zeros(nb_iterations)
    vect_accuracy = np.zeros(nb_iterations)
    
    # Initialisation aléatoire des poids du réseau
    W1 = 2*np.random.random((D_in, D_h)) - 1
    W1 = np.array(W1, dtype="float32")
    # b1 = np.zeros((1, D_h))
    W2 = 2*np.random.random((D_h, D_out))-1
    W2 = np.array(W2, dtype="float32")
    # b2 = np.zeros((1, D_out))
    
    for i in range(nb_iterations):
        ####################################################
        # Passe avant : calcul de la sortie prédite Y_pred #
        ####################################################
        O1 = sigmoid(X, W1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
        O2 = sigmoid(O1, W2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
        O2 = np.divide(O2, np.sum(O2, axis=1)[:,np.newaxis], dtype="float32")
        # O2/np.sum(O2, axis=1)[:,np.newaxis] # Normalisation des coefficients sur chaque ligne
        Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie
        
        # if i == 0:
        #     print("O2", O2)
        ########################################################
        # Calcul et affichage de la fonction perte de type MSE #
        ########################################################
        loss = np.square(Y_pred - Y).sum()/2
        vect_loss[i] = loss
        # if i%int(nb_iterations/10) == 0 :
        #     print(i, "loss :", loss)
        
        ######################################################
        # Calcul et affichage de la précision du classifieur #
        ######################################################
        accuracy = evaluation_classifieur(Y, Y_pred)
        vect_accuracy[i] = accuracy
        
        ########################################
        # Passe arrière : mis à jour des poids #
        ########################################
        grad_Y_pred = 2*(Y_pred - Y)
        grad_O2 = grad_Y_pred*d_sigmoid(O2)
        grad_w2 = O1.T.dot(grad_O2)
        
        grad_O1 = d_sigmoid(O1)
        grad_w1 = X.T.dot((grad_O2.dot(W2.T))*grad_O1)
        
        W1 -= gradient_step*grad_w1
        W2 -= gradient_step*grad_w2
    
    vect_accuracy*= 100
    return Y_pred, W1, W2, vect_loss, vect_accuracy

def neural_network_classification_cifar_10_test(Xtest, 
                                                Ytest, 
                                                W1, 
                                                W2, 
                                                gradient_step=1e-3, 
                                                nb_iterations = 1000):
    N, D_in = np.shape(Xtest)
    D_h, D_out = 20, 10
    
    # Création d'une matrice d'entrée X et de sortie Y
    X = np.array(Xtest/255, dtype="float32")  #Normalisation des données
    
    # Initialisation des poids du réseau
    b1 = np.zeros((1, D_h))
    b2 = np.zeros((1, D_out))
    
    ####################################################
    # Passe avant : calcul de la sortie prédite Y_pred #
    ####################################################
    I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
    O1 = 1/(1 + np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
    O2 = 1/(1 + np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
    O2 = O2/np.sum(O2, axis=1)[:,np.newaxis] # Normalisation des coefficients sur chaque ligne
    Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie
    
    return Y_pred

#%% Recherche du meilleur gradient_step et nombre d'itérations

##################################################
# ATTENTION : Met plusieurs minutes à s'exécuter #
##################################################

liste_gradient_step = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
liste_nb_iterations = [100, 200, 500, 1000]
nb_gradient_steps = len(liste_gradient_step)
nb_liste_nb_iterations = len(liste_nb_iterations)
accuracy_matrix = np.zeros((nb_gradient_steps, nb_liste_nb_iterations), dtype=float)

Y = matrice_stochastique(Yapp)

for i, g_step in enumerate(liste_gradient_step):
    for j, nb_iter in enumerate(liste_nb_iterations):
        Y_pred, W1, W2, vect_loss, vect_accuracy = neural_network_classification_cifar_10_train(Xapp, Yapp, gradient_step = g_step, nb_iterations = nb_iter)
        accuracy = evaluation_classifieur(Y, Y_pred)
        accuracy *= 100
        accuracy_matrix[i,j] = accuracy
        print("Gradient step :", g_step, "and Nb iterations :", nb_iter)
        print("Accuracy :", round(accuracy, 2), "%")

# accuracy_matrix
# array([[10.4875, 10.025 , 10.025 , 10.025 ],
#        [10.4875, 10.05  ,  9.625 , 10.025 ],
#        [10.025 , 10.4875, 10.1625,  9.9   ],
#        [22.05  , 21.375 , 20.4   , 27.875 ],
#        [22.725 , 22.2875, 18.825 , 20.7125],
#        [16.3625, 20.7   , 22.35  , 24.425 ],
#        [16.7875, 18.1   , 18.7375, 21.55  ],
#        [11.45  , 13.425 , 16.05  , 18.425 ]])
# The gradient steps 1e-3 and 5e-4 appears to be the best.

#%% 
gradient_step = 1e-3
nb_iterations = 400

Yapp_stochastique = matrice_stochastique(Yapp)
Y_pred, W1, W2, vect_loss, vect_accuracy = neural_network_classification_cifar_10_train(Xapp, Yapp, gradient_step=gradient_step, nb_iterations=nb_iterations)

fig = plt.figure()
ax1 = fig.add_subplot(211)
fig.suptitle("Fonction de perte en fonction des itérations")
ax1.set_xlabel(r'Itération $i$')
ax1.set_ylabel("Perte")
# ax1.set_xticks(np.arange(0, 25, step=2))
ax1.grid()
ax1.plot(vect_loss)

ax2 = fig.add_subplot(212)
fig.suptitle("Précision en fonction des itérations")
ax2.set_xlabel(r'Itération $i$')
ax2.set_ylabel("Précision")
# ax.set_xticks(np.arange(0, 25, step=2))
ax2.grid()
ax2.plot(vect_accuracy)

# accuracy = evaluation_classifieur(Yapp_stochastique, Y_pred)
# accuracy *= 100
# print("Gradient step :", gradient_step, "and Nb iterations :", nb_iterations)
# print("Accuracy :", round(accuracy, 2), "%")

Y_test_stochastique = matrice_stochastique(Ytest)
Y_pred_test = neural_network_classification_cifar_10_test(Xtest, Ytest, W1, W2, gradient_step=gradient_step, nb_iterations=nb_iterations)
accuracy_test = evaluation_classifieur(Y_test_stochastique, Y_pred_test)
accuracy_test *= 100
print("Gradient step :", gradient_step, "and Nb iterations :", nb_iterations)
print("Maximum accuracy :", round(np.amax(vect_accuracy), 2), "% reached at iteration ", np.where(vect_accuracy == np.amax(vect_accuracy))[0][0])
print("Accuracy test data:", round(accuracy_test, 2), "%")





