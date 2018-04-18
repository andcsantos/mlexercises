#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:16:43 2018

@author: lasiand
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## treinamento
dataset = pd.read_csv('treinamento.csv', header=None)
amostras = dataset.iloc[:, :-1].values
amostras = np.insert(amostras, 0, -1, axis=1 ) #com bias
saidas = dataset.iloc[:, -1].values

w = np.random.rand(1,5)
w_in = w
n = 0.0025
epocas = 0
eps = 1e-6
x = np.matrix(amostras)
x_t = np.transpose(x)
x_in = x_t[:, 0]
u_in = w_in * x_in
eqm = 0

py = np.array([])
px = np.array([])

def erroqm(size, eqm = 0):
    for i in range(size):
        x_t_i = x_t[:, i]
        u = w * x_t_i
        eqm = eqm + (saidas[i] - u) ** 2
    eqm = eqm/size
    return eqm

while True:
    erro = eqm
    for i in range(len(amostras)):
        x_t_i = x_t[:, i]
        u = w * x_t_i
        w = w + (n * (saidas[i] - u) * amostras[i, :])
    epocas += 1
    eqm = erroqm(len(amostras))
    px = np.append(px, epocas)
    py = np.append(py, eqm)
    if abs(erro - eqm) <= eps:
        break

#gráficos
plt.plot(px, py)

## operacao
dataset_teste = pd.read_csv('teste.csv', header=None)
entradas = dataset_teste.iloc[:,:].values
entradas = np.insert(entradas, 0, -1, axis=1 ) #com bias
entradas_t = np.transpose(entradas)
v = w * entradas_t
respostas = np.sign(v)