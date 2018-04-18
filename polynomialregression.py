#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:31:13 2018

@author: lasiand
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values ##deixar no formato de matriz 1:2
y = dataset.iloc[:, 2].values

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
#                                                    random_state = 0)

##fitting Linear
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

##fitting Polynomial
##tranformando em um polinomio de n-grau
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(X)

#criando uma regressao linear multipla encaixando o polinomio de n-grau
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizacao
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Checking")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') ##usar o polinomio
plt.title("Checking")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

#predicao
lin_reg.predict(6.5)
lin_reg2.predict(poly_reg.fit_transform(6.5))

