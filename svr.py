#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:20:59 2018

@author: lasiand
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#define dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #atencao
y = dataset.iloc[:, 2:3].values

#feature scaling ##SVR NAO POSSUI NORMALIZACAO AUTOMATICA
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#cria o regressor
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicao (prever y usando X!!!)
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#visualizacao
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('user xp')
plt.ylabel('salary')
plt.show()
