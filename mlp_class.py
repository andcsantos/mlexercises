#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:03:14 2018

@author: Anderson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

dataset = pd.read_csv("rna_treinamento5.txt", header=None, sep="\s+")
X = dataset.iloc[:,1:-3].values
d = dataset.iloc[:,-3:].values
X = np.insert(X, 0, -1, axis=1) #bias
n_neurons1 = 15
n_inputs = 4
n_inputs = n_inputs + 1
n_out = 3

w1 = np.random.rand(n_neurons1, n_inputs)
w1i = np.copy(w1)
w2 = np.random.rand(n_neurons1,n_out)
w2i = np.copy(w2)

n = 0.1
epsilon = 1e-6

epocas = 0
erro = 1

I1 = np.zeros(n_neurons1)
y1 = np.zeros(n_neurons1)
delta1 = np.zeros(n_neurons1)
dery1 = np.zeros(n_neurons1)
I2 = np.zeros(n_out)
y2 = np.zeros(n_out)
delta2 = np.zeros(n_out)
dery2 = np.zeros(n_out)

yerro = []
xepoc = []

while True:
    em = erro
    erro = 0
    for j in range(len(X)):
        for i in range(n_neurons1):
            I1[i] = np.inner(X[j,:], w1[i,:]) #produto escalar
            y1[i] = logistic.cdf(I1[i])
        for i in range(n_out):
            I2[i] = np.inner(y1, w2[:,i])
            y2[i] = logistic.cdf(I2[i])
        for i in range(n_out):
            dery2[i] = y2[i]*(1-y2[i])
            delta2[i] = (d[j,i]-y2[i])*dery2[i]
        for k in range(n_out):
            for i in range(n_neurons1):
                w2[i,k] = w2[i,k] + n*delta2[k]*y1[i]
        for k in range(n_out):
            for i in range(n_neurons1):
                dery1[i] = y1[i]*(1-y1[i])
                delta1[i] = np.inner(delta2[k],w2[i,k])*dery1[i]
                w1[i,:] = w1[i,:] + n*delta1[i]*X[j,:]
        for i in range(n_out):
            erro = erro + (d[j,i]-y2[i]) ** 2
    erro = erro/(2*len(X)*n_out)
    yerro.append(erro)
    epocas += 1
    xepoc.append(epocas)
    if abs(em-erro)<epsilon:
        break

#operacao
teste = pd.read_csv("rna_teste5.txt", header=None, sep="\s+")
x = teste.iloc[:,1:-3].values
x = np.insert(x, 0, -1, axis=1)
y = np.zeros([len(x),n_out])
esperado = teste.iloc[:,-3:].values
erro_y = 0
for j in range(len(x)):
    for i in range(n_neurons1):
        I1[i] = np.inner(x[j,:], w1[i,:]) #produto escalar
        y1[i] = logistic.cdf(I1[i])
    for i in range(n_out):
        I2[i] = np.inner(y1, w2[:,i])
        y2[i] = logistic.cdf(I2[i])
    for i in range(n_out):
            if y2[i]>=0.5:
                y[j,i]=1
            else: y[j,i]=0

#chk
chk=0
for i in range(len(x)):
    for j in range(n_out):
        if esperado[i,j] == 1 and y[i,j] == 1:
            chk += 1
acc = chk/len(x) * 100

plt.plot(xepoc, yerro)