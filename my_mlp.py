#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:11:36 2018

@author: lasiand
"""

##valido somente para uma arquitetura de ce1=10 e ce2=1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

dataset = pd.read_csv("rna_treinamento4.txt", header=None, sep="\s+")
X = dataset.iloc[:,:-1].values
d = dataset.iloc[:,-1].values
X = np.insert(X, 0, -1, axis=1) #bias
n_neurons1 = 10
n_inputs = 3
n_inputs = n_inputs + 1
n_out = 1

w1 = np.random.rand(n_neurons1, n_inputs)
w1i = np.copy(w1)
w2 = np.random.rand(n_neurons1)
w2i = np.copy(w2)

n = 0.1
epson = 1e-6

epocas = 0
erro = 1

I1 = np.zeros(n_neurons1)
y1 = np.zeros(n_neurons1)
delta1 = np.zeros(n_neurons1)
dery1 = np.zeros(n_neurons1)

yerro = []
xepoc = []

while True:
    em = erro
    erro = 0
    for j in range(len(X)):
        for i in range(n_neurons1):
            I1[i] = np.inner(X[j,:], w1[i,:]) #produto escalar
            y1[i] = logistic.cdf(I1[i])
        I2 = np.inner(y1, w2)
        y2 = logistic.cdf(I2)
        dery2 = y2*(1-y2)    
        delta2 = (d[j]-y2)*dery2
        for i in range(n_neurons1):
            w2[i] = w2[i] + n*delta2*y1[i]
        for i in range(n_neurons1):
            dery1[i] = y1[i]*(1-y1[i])
            delta1[i] = np.inner(delta2,w2[i])*dery1[i]
            w1[i,:] = w1[i,:] + n*delta1[i]*X[j,:]
        erro = erro + (d[j]-y2) ** 2
    erro = erro/(2*len(X))
    yerro.append(erro)
    epocas += 1
    xepoc.append(epocas)
    if abs(em-erro)<epson:
        break

#operacao
teste = pd.read_csv("rna_teste4.txt", header=None, sep="\s+")
x = teste.iloc[:,:-1].values
x = np.insert(x, 0, -1, axis=1)
y = np.array([])
esperado = teste.iloc[:,-1].values
erro_y = 0
for j in range(len(x)):
    for i in range(n_neurons1):
        I1[i] = np.inner(x[j,:], w1[i,:]) #produto escalar
        y1[i] = logistic.cdf(I1[i])
    I2 = np.inner(y1, w2)
    y2 = logistic.cdf(I2)
    y = np.append(y, y2)
    erro_y = erro_y + abs((esperado[j]-y2)/esperado[j])
    
erro_y = erro_y/len(x)
var = np.var(y)

plt.plot(xepoc, yerro)
    
        
    
