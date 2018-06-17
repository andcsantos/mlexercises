#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:49:40 2018

@author: lasiand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######## originais ########

fig1  = np.matrix([[0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])

fig2 = np.matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                 [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0],
                 [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

fig3 = np.matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

fig4 = np.matrix([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                 [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

def converttominus(x):
    for i in range(9):
        for j in range(5):
            if x[i, j] == 0:
                x[i, j] = -1

converttominus(fig1)
converttominus(fig2)
converttominus(fig3)
converttominus(fig4)

##########################

####### com ruido #######
fig1r_1 = np.matrix([[0, 0, 1, 0, 1], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 1, 1, 1]])


fig1r_2 = np.random.permutation(fig1)
fig1r_3 = np.random.permutation(fig1)

fig2r_1 = np.matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                 [1, 0, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0],
                 [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
fig2r_2 = np.random.permutation(fig2)
fig2r_3 = np.random.permutation(fig2)

fig3r_1 = np.matrix([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

fig3r_2 = np.random.permutation(fig3)
fig3r_3 = np.random.permutation(fig3)

fig4r_1 = np.matrix([[1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1],
                 [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
fig4r_2 = np.random.permutation(fig4)
fig4r_3 = np.random.permutation(fig4)

converttominus(fig1r_1)
converttominus(fig2r_1)
converttominus(fig3r_1)
converttominus(fig4r_1)



##########################

n = 45
p = 4
fig1f = fig1.flatten()
fig2f = fig2.flatten()
fig3f = fig3.flatten()
fig4f = fig4.flatten()
z = np.concatenate((fig1f, fig2f, fig3f, fig4f), axis=0)
z = np.transpose(z)

w = np.matrix(np.zeros([n, n]))
for i in range(p):
    w = w + z[:, i] * np.transpose(z[:, i])    
w = (1/n * w) - (p/n * np.identity(n))


def operacao(x, epocas = 0):
    x = x.reshape(45,1)
    v = np.copy(x)
    beta = 100
    while True:
        va = np.copy(v)
        u = w * va
        v = ((1-np.exp(-u*beta))/(1+np.exp(-u*beta)))
        epocas += 1
        if np.array_equal(v, va) == True:
            return epocas, v
    
epocas1, fig1r_1a = operacao(fig1r_1)
fig1r_1a = fig1r_1a.reshape(9,5)
epocas2, fig1r_2a = operacao(fig1r_2)
fig1r_2a = fig1r_2a.reshape(9,5)
epocas3, fig1r_3a = operacao(fig1r_3)
fig1r_3a = fig1r_3a.reshape(9,5)

epocas4, fig2r_1a = operacao(fig2r_1)
fig2r_1a = fig2r_1a.reshape(9,5)
epocas5, fig2r_2a = operacao(fig2r_2)
fig2r_2a = fig2r_2a.reshape(9,5)
epocas6, fig2r_3a = operacao(fig2r_3)
fig2r_3a = fig2r_3a.reshape(9,5)

epocas7, fig3r_1a = operacao(fig3r_1)
fig3r_1a = fig3r_1a.reshape(9,5)
epocas8, fig3r_2a = operacao(fig3r_2)
fig3r_2a = fig3r_2a.reshape(9,5)
epocas9, fig3r_3a = operacao(fig3r_3)
fig3r_3a = fig3r_3a.reshape(9,5)

epocas10, fig4r_1a = operacao(fig4r_1)
fig4r_1a = fig4r_1a.reshape(9,5)
epocas11, fig4r_2a = operacao(fig4r_2)
fig4r_2a = fig4r_2a.reshape(9,5)
epocas12, fig4r_3a = operacao(fig4r_3)
fig4r_3a = fig4r_3a.reshape(9,5)

####################### plots #######################

fig = plt.figure(figsize = (20, 20))
fig.add_subplot(1, 4, 1)
plt.imshow(fig1)
fig.add_subplot(1, 4, 2)
plt.imshow(fig2)
fig.add_subplot(1, 4, 3)
plt.imshow(fig3)
fig.add_subplot(1, 4, 4)
plt.imshow(fig4)
plt.show()

figs1 = plt.figure(figsize = (20, 20))
figs1.add_subplot(2, 3, 1)
plt.imshow(fig1r_1)
figs1.add_subplot(2, 3, 2)
plt.imshow(fig1r_2)
figs1.add_subplot(2, 3, 3)
plt.imshow(fig1r_3)
figs1.add_subplot(2, 3, 4)
plt.imshow(fig1r_1a)
figs1.add_subplot(2, 3, 5)
plt.imshow(fig1r_2a)
figs1.add_subplot(2, 3, 6)
plt.imshow(fig1r_3a)
plt.show()

figs2 = plt.figure(figsize = (20, 20))
figs2.add_subplot(2, 3, 1)
plt.imshow(fig2r_1)
figs2.add_subplot(2, 3, 2)
plt.imshow(fig2r_2)
figs2.add_subplot(2, 3, 3)
plt.imshow(fig2r_3)
figs2.add_subplot(2, 3, 4)
plt.imshow(fig2r_1a)
figs2.add_subplot(2, 3, 5)
plt.imshow(fig2r_2a)
figs2.add_subplot(2, 3, 6)
plt.imshow(fig2r_3a)
plt.show()


figs3 = plt.figure(figsize = (20, 20))
figs3.add_subplot(2, 3, 1)
plt.imshow(fig3r_1)
figs3.add_subplot(2, 3, 2)
plt.imshow(fig3r_2)
figs3.add_subplot(2, 3, 3)
plt.imshow(fig3r_3)
figs3.add_subplot(2, 3, 4)
plt.imshow(fig3r_1a)
figs3.add_subplot(2, 3, 5)
plt.imshow(fig3r_2a)
figs3.add_subplot(2, 3, 6)
plt.imshow(fig3r_3a)
plt.show()

figs4 = plt.figure(figsize = (20, 20))
figs4.add_subplot(2, 3, 1)
plt.imshow(fig4r_1)
figs4.add_subplot(2, 3, 2)
plt.imshow(fig4r_2)
figs4.add_subplot(2, 3, 3)
plt.imshow(fig4r_3)
figs4.add_subplot(2, 3, 4)
plt.imshow(fig4r_1a)
figs4.add_subplot(2, 3, 5)
plt.imshow(fig4r_2a)
figs4.add_subplot(2, 3, 6)
plt.imshow(fig4r_3a)
plt.show()