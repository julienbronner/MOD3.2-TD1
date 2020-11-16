# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:26:18 2020

@author: julbr
"""
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import skimage
import PIL.Image

seed = 2
np.random.seed(seed) # pour que l'exécution soit déterministe
fraction_test = 1/5

#%% Fonctions
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
    nbr_aleatoires_test = np.random.choice(n, int(n*fraction_test), replace=False)
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
    
def kppv_distances_bala(Xtest, Xapp):
    a_2 = (Xtest**2).sum(axis=1)
    a_2 = a_2.reshape((-1,1))
    print(a_2)
    b_2 = (Xapp**2).sum(axis=1)
    b_2 = b_2.reshape((1,-1))
    print(b_2)
    dist = -2 * Xtest.dot(Xapp.T) + a_2 + b_2
    print(Xtest.dot(Xapp.T))
    return dist

def kppv_predict(dist, Yapp, K): # utilisationde np.argpartition(A,k) qui donne les indices pour que jusqu'à k, on ait les valeurs les  plus petites
    N,M = np.shape(dist)
    #print(N,M)
    sort_indices = np.argpartition(dist, K-1, axis = 0) #K-1 car on part de 0
    Yapp = np.reshape(Yapp, (N,1))
    Yapp_mat = Yapp.dot(np.ones((1,M))) # pour dupliquer Yapp dans toutes les colonnes
    Yapp_mat_sort = np.take_along_axis(Yapp_mat, sort_indices, axis=0)
    Yapp_mat_sort_tronque = Yapp_mat_sort[:K, : ]
    Ypred = stats.mode(Yapp_mat_sort_tronque)[0][0] #permet d'avoir l'element le plus présent
    return Ypred

def evaluation_classifieur(Ytest, Ypred):
    Ybool = (Ytest == Ypred)
    nbr_true = sum(Ybool)
    nbr_tot = len(Ybool)
    return nbr_true/nbr_tot*100

#%% Test

K = 10
Kmax = 1000
path = 'D:/julbr/Documents/ecole/ECL/3A\MOD 3.2 Deep Learning & IA/TD1/cifar-10-batches-py/data_batch_1'
def fonction_test(K, path):
    data_X, label_Y = lecture_cifar(path)
    Xapp, Yapp, Xtest, Ytest = decoupage_donnes(data_X, label_Y)
    #print(np.shape(Xapp))
    dist = kppv_distances(Xtest, Xapp)
    Ypred = kppv_predict(dist, Yapp, K)
    accuracy = evaluation_classifieur(Ytest, Ypred)
    return(accuracy)

#%% Expérimentations
    
def influence_K(Kmax, path):
    data_X, label_Y = lecture_cifar(path)
    Xapp, Yapp, Xtest, Ytest = decoupage_donnes(data_X, label_Y)
    dist = kppv_distances(Xtest, Xapp)
    K_liste = []
    accuracy_liste = []
    for K in range(1,Kmax, int(Kmax/100)):
        K_liste.append(K)
        Ypred = kppv_predict(dist, Yapp, K)
        accuracy_liste.append(evaluation_classifieur(Ytest, Ypred))
        
#    fig = plt.figure()
#    fig.title("Précisions en fonction du nombre de voisins")
#    fig.set_xlabel("Nombre de voisins")
#    fig.set_ylabel("Précision")
#    fig.plot(K_liste, accuracy_liste)
    plt.plot(K_liste, accuracy_liste)
    plt.title("Précisions en fonction du nombre de voisins")
    plt.xlabel("Nombre de voisins")
    plt.ylabel("Précision (%)")
    plt.show()
    
#print(fonction_test(K, path))
#influence_K(Kmax, path)

#%% Affichage
    
X, Y = lecture_cifar(path)
test = X[0]
X_r = test[:32*32].reshape((32, 32))
X_g = test[32*32:2*32*32].reshape((32, 32))
X_b = test[2*32*32:].reshape((32, 32))
X_tot = np.array([X_r, X_g, X_b])
X_tot = np.transpose(X_tot, [1,2,0])
#print(np.shape(X_tot))

image = PIL.Image.fromarray(X_tot, "RGB")
plt.imshow(image)



#Visualizing CIFAR 10 --> fonctionne
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#fig, axes1 = plt.subplots(5,5,figsize=(3,3))
#for j in range(5):
#    for k in range(5):
#        i = np.random.choice(range(len(X)))
#        axes1[j][k].set_axis_off()
##        axes1[j][k].imshow(X[i:i+1][0])  
#        axes1[j][k].imshow(X[i])  

