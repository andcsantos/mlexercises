#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:41:34 2018

@author: lasiand
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset
dataset = pd.read_csv("svm1.txt", sep = "\s+", header = None)
X = dataset.iloc[:,0:2].values
y = dataset.iloc[:,2].values

#plot
plt.scatter(X[:,0], X[:,1], c=y)

#normalizacao        
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
Xs = sc.fit_transform(X)

#classificador
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(Xs, y)

#equacao
w = classifier.coef_
intercept = classifier.intercept_

#plot2
def draw_svm(X, y, C=1.0):
    # Plot
    plt.scatter(X[:,0], X[:,1], c=y)
    
    # classificador
    clf = SVC(kernel='linear', C=C)
    clf_fit = clf.fit(X, y)
    
    # limita os eixos
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # cria os vetores
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # Plota os limites
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
                        alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], 
                clf.support_vectors_[:, 1], 
                s=100, linewidth=1, facecolors='none')
    plt.show()
    # retorna classificado
    return clf_fit

clf_arr = []
clf_arr.append(draw_svm(Xs, y, 1))
plt.title("C=1")
plt.show()
clf_arr.append(draw_svm(Xs, y, 100))
plt.title("C=100")
plt.show()


###PARTE 2
from sklearn.svm import SVC
dataset_treinamento = pd.read_csv("svm2_treinamento.txt", sep = "\s+", header = None)
dataset_teste = pd.read_csv("svm2_teste.txt", sep = "\s+", header = None)
X = dataset_treinamento.iloc[:,0:2].values
y = dataset_treinamento.iloc[:,2].values
X_teste = dataset_teste.iloc[:,0:2].values
y_teste = dataset_teste.iloc[:,2].values

plt.scatter(X[:,0], X[:,1], c=y)
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
Xs = sc.fit_transform(X)
Xs_teste = sc.fit_transform(X_teste)

def tx_acerto(gamma, count = 0):
    classifier = SVC(gamma = gamma)
    classifier.fit(Xs, y)
    pred = classifier.predict(Xs_teste)
    for i in range(len(X_teste)):
        if pred[i]==y_teste[i]:
            count += 1
    return count/len(X_teste)

tx_acerto(0.01)
tx_acerto(0.1)
tx_acerto(1)
tx_acerto(10)
tx_acerto(100)
        
    

