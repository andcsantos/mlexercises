#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Mon May 21 11:40:58 2018

@author: lasiand
=======
Created on Sun May 20 15:27:27 2018

@author: Anderson
>>>>>>> 859a664badf763ca7e03556e5bce9aca805ea73d
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logistic

dataset = pd.read_csv("treinamento.txt", header=None, sep="\s+")
inp = dataset.iloc[:,1:].values
<<<<<<< HEAD
n_inputs = 15
d = dataset.iloc[n_inputs:,1:].values
n_neurons1 = 25
=======
d = dataset.iloc[5:,1:].values
n_neurons1 = 10
n_inputs = 5
>>>>>>> 859a664badf763ca7e03556e5bce9aca805ea73d
n_out = 1

w1 = np.random.rand(n_neurons1, n_inputs+1)
w1p = np.random.rand(n_neurons1, n_inputs+1)
w1i = np.copy(w1)
w2 = np.random.rand(n_neurons1,n_out)
w2p = np.random.rand(n_neurons1,n_out)
w2i = np.copy(w2)

n = 0.1
epsilon = 0.5e-6
momento = 0.8

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

##entradas
X = np.zeros([len(d),n_inputs])
for i in range(len(d)):
    for j in range(n_inputs):
        X[i,j] = inp[j+i]
    
X = np.insert(X, 0, -1, axis=1) #bias
n_inputs += 1

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
        temp = np.copy(w2)
        for k in range(n_out):
            for i in range(n_neurons1):
                w2[i,k] = w2[i,k] + momento*(w2[i,k]-w2p[i,k]) + n*delta2[k]*y1[i]
        w2p = np.copy(temp)
        temp = np.copy(w1)
        for k in range(n_out):
            for i in range(n_neurons1):
                dery1[i] = y1[i]*(1-y1[i])
                delta1[i] = np.inner(delta2[k],w2[i,k])*dery1[i]
                w1[i,:] = w1[i,:] + momento*(w1[i,:]-w1p[i,:]) + n*delta1[i]*X[j,:]
        w1p = np.copy(temp)
        for i in range(n_out):
            erro = erro + (d[j,i]-y2[i]) ** 2
    erro = erro/(2*len(X)*n_out)
    yerro.append(erro)
    epocas += 1
    xepoc.append(epocas)
    if abs(em-erro)<epsilon:
        break

#operacao
teste = pd.read_csv("teste.txt", header=None, sep="\s+")
inpp = teste.iloc[:,1:].values
inp = np.concatenate((inp,inpp), axis=0)

x = np.zeros([len(inp)-n_inputs+1,n_inputs-1])
for i in range(len(inp)-n_inputs+1):
    for j in range(n_inputs-1):
        x[i,j] = inp[j+i]
    
x = np.insert(x, 0, -1, axis=1) #bias

y = np.zeros([len(x)-len(X),n_out])
esperado = teste.iloc[:,1:].values
erro_y = 0
k=0
for j in range(len(X),len(x)):
    for i in range(n_neurons1):
        I1[i] = np.inner(x[j,:], w1[i,:]) #produto escalar
        y1[i] = logistic.cdf(I1[i])
    for i in range(n_out):
        I2[i] = np.inner(y1, w2[:,i])
        y2[i] = logistic.cdf(I2[i])
        y[k,i] = y2[i]
    k +=1

<<<<<<< HEAD
for i in range(len(y)):
     erro_y = erro_y + abs((esperado[i]-y[i])/esperado[i])
erro_y = erro_y/len(x)
var = np.var(y)
    
plt.plot(xepoc, yerro)
plt.title("EQM X Epocas")
plt.ylabel("EQM")
plt.xlabel("Epocas")
plt.show()

plt.plot(range(len(y)), y, label = 'predicao')
plt.plot(range(len(inpp)), inpp, label = 'esperado')
plt.xlabel("tempo (100s+)")
plt.legend()
plt.show()
=======
#chk
chk=0
for i in range(len(x)):
    for j in range(n_out):
        if esperado[i,j] == 1 and y[i,j] == 1:
            chk += 1
acc = chk/len(x) * 100

plt.plot(xepoc, yerro)
>>>>>>> 859a664badf763ca7e03556e5bce9aca805ea73d
