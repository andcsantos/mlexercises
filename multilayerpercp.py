#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:49:25 2018

@author: lasiand
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv('treinamento.txt', sep='\s+',header=None)
amostras = dataset.iloc[:, :-3].values
saidas = dataset.iloc[:, 4:].values

net = MLPClassifier(hidden_layer_sizes=(15,), tol=1e-6, learning_rate_init=0.1)
net.fit(amostras, saidas)

dataset_teste = pd.read_csv('teste.txt', sep='\s+',header=None)
entradas = dataset_teste.iloc[:,:-3].values

respostas = net.predict(entradas) #binario
coeficientes = net.predict_proba(entradas) #reais

plt.plot(net.loss_curve_)


