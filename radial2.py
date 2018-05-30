#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:02:58 2018

@author: lasiand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('treinamento.txt', sep='\s+',header=None)
X = dataset.values[:,1:-1]
d = dataset.values[:,-1:]

for i in range(len(d)):
    if d[i] == -1:
        d[i] = 0

k = 15
n_in = 3
n_out = 1

w1 = X[:k,:]

Xl = np.insert(X, n_in, -1, axis=1)
dis = np.zeros(shape = (len(X), k))
itr = 0

while True:
    chk = np.copy(Xl[:,-1])
    for j in range(k):
        for i in range(len(X)):
            dis[i, j] = np.linalg.norm(X[i,:]-w1[j,:])
    for i in range(len(X)):
        for j in range(k):
            if dis[i, j] - dis[i, :].min() < 1e-6:
                Xl[i, -1] = j
    for i in range(k):
        cal_cx = []
        cal_cy = []
        for j in range(len(X)):
            if i == Xl[j, -1]:
                cal_cx.append(Xl[j,0])
                cal_cy.append(Xl[j,1])
        w1[i, 0] = np.mean(cal_cx)
        w1[i, 1] = np.mean(cal_cy)
    itr += 1
    count = 0
    for i in range(len(X)):
        if Xl[i,-1] == chk[i]:
            count += 1
    if count == len(X):
        break
    
var = np.zeros(k)
for j in range(k):
    c = 0
    for i in range(len(Xl)):
        if Xl[i,-1] == j:
            for l in range(n_in):
                var[j] = var[j] + (X[i,l] - w1[j,l]) ** 2
            c += 1
    var[j] = var[j]/c

plt.scatter(X[:,0], X[:,1], c=Xl[:,2])
plt.scatter(w1[:,0],w1[:,1], color = 'red', s=200)


z = np.zeros([len(X),k])
g = np.zeros(k)
for i in range(len(X)):
    for j in range(k):
            p = sum((X[i,:] - w1[j,:]) ** 2)
            g[j] = 1/np.exp(p/(2*var[j]))
    z[i,:] = g

g = np.insert(z, 0, -1, axis=1) #bias
g = np.transpose(g)
k1 = k
k = k+1
w2 = np.random.rand(k,n_out)
w2i = np.copy(w2)
n = 0.01
epsilon = 1e-7
epocas = 0
erro = 1
y2 = np.zeros(n_out)
delta2 = np.zeros(n_out)

yerro = []
xepoc = []

while True:
    em = erro
    erro = 0
    for j in range(len(X)):
        for i in range(n_out):
            y2[i] = np.inner(g[:,j], w2[:,i])
        for i in range(n_out):
            delta2[i] = (d[j,i]-y2[i])
        for l in range(n_out):
            for i in range(k):
                w2[i,l] = w2[i,l] + n*delta2[l]*g[i,j]
        for i in range(n_out):
            erro = erro + (d[j,i]-y2[i]) ** 2
    erro = erro/(2*len(X)*n_out)
    yerro.append(erro)
    epocas += 1
    xepoc.append(epocas)
    if abs(em-erro)<epsilon:
        break
    
###operacao
amostras = pd.read_csv('teste.txt', sep='\s+',header=None)
x = amostras.iloc[:,1:-1].values
D = amostras.iloc[:,-1].values
for i in range(len(D)):
    if D[i] == -1:
        D[i] = 0

g1 = np.zeros(k1)
z1 = np.zeros([len(x),k1])
for i in range(len(x)):
    for j in range(k1):
            p = sum((x[i,:] - w1[j,:]) ** 2)
            g1[j] = 1/np.exp(p/(2*var[j]))
    z1[i,:] = g1

g1 = np.insert(z1, 0, -1, axis=1) #bias
g1 = np.transpose(g1)
y = np.zeros(len(x))
yp = np.zeros(len(x))
for j in range(len(x)):
    for i in range(n_out):
        y[j]=np.inner(g1[:,j],w2[:,i])
        
error = 0
for i in range(len(x)):
    error = error + abs(D[i]-y[i])/D[i]
error = error/len(x)
   