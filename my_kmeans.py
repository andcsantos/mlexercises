#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:08:53 2018

@author: lasiand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('remedios.txt', sep='\s+',header=None)
X = dataset.values[:,:]
k = 5
Xl = np.insert(X, 2, -1, axis=1)
c = []
for i in range(k):
    c.append(random.choice(X))
c = np.array(c)
dis = np.zeros(shape = (len(X), k))
itr = 0

while True:
    chk = np.copy(Xl[:,-1])
    for j in range(k):
        for i in range(len(X)):
            dis[i, j] = np.linalg.norm(X[i,:]-c[j,:])
    for i in range(len(X)):
        for j in range(k):
            if dis[i, j] - dis[i, :].min() < 1e-9:
                Xl[i, -1] = j
    for i in range(k):
        cal_cx = []
        cal_cy = []
        for j in range(len(X)):
            if i == Xl[j, -1]:
                cal_cx.append(Xl[j,0])
                cal_cy.append(Xl[j,1])
        c[i, 0] = np.mean(cal_cx)
        c[i, 1] = np.mean(cal_cy)

    itr += 1
    count = 0
    for i in range(len(X)):
        if Xl[i,-1] == chk[i]:
            count += 1
    if count == len(X):
        break


plt.scatter(Xl[:,0],Xl[:,1], c=Xl[:,-1])
plt.scatter(c[:,0],c[:,1], color = 'red', s=200)
plt.show()

plt.scatter(X[:,0],X[:,1])

qtd = []
for i in range(k):
    ckt = 0
    for j in range(len(X)):
        if Xl[j,-1]==i:
            ckt += 1
    qtd.append(ckt)

dm1 = np.mean(dis[0,:])
dm2 = np.mean(dis[1,:])
dm3 = np.mean(dis[2,:])
dm4 = np.mean(dis[3,:])
dm5 = np.mean(dis[4,:])
    
            
    
                    