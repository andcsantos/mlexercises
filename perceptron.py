#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:21:26 2018

@author: lasiand
"""

import pandas as pd
import numpy as np

## treinamento
dataset = pd.read_csv('treinamento.csv', header=None)
amostras = dataset.iloc[:, :-1].values
amostras = np.insert(amostras, 0, -1, axis=1 ) #com bias
saidas = dataset.iloc[:, -1].values

w = np.random.rand(1,4)
w_in = w
n = 0.01
epocas = 0
x = np.matrix(amostras)
x_t = np.transpose(x)
x_in = x_t[:, 0]
u_in = w_in * x_in

while True:
        erro = False
        for i in range(len(amostras)):
            x_t_i = x_t[:, i]
            u = w * x_t_i
            y = np.sign(u)
            if y != saidas[i]:
                w = w + (n * (saidas[i] - y) * amostras[i, :])
                erro = True
        epocas += 1
        if erro == False:
            break

## operacao
dataset_teste = pd.read_csv('teste.csv', header=None)
entradas = dataset_teste.iloc[:,:].values
entradas = np.insert(entradas, 0, -1, axis=1 ) #com bias
entradas_t = np.transpose(entradas)
v = w * entradas_t
respostas = np.sign(v)

