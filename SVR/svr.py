# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:27:35 2021

@author: Kevin
"""
# SVR Maquinas de Soporte Vectorial

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Importar el Data Set

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Escalado de variables
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresion con el dataset

regresion = SVR(kernel="rbf")
regresion.fit(X,y)

# Predeccion del modelo con SVR

experiencia = np.array(6.5)
experiencia = experiencia.reshape(1,-1) # transformar a array de 1x1
y_pred = regresion.predict(sc_X.transform(experiencia))
y_pred = sc_y.inverse_transform(y_pred)

plt.scatter(X, y, color = "red")
plt.plot(X, regresion.predict(X), color="blue")
plt.title("Modelo de regresion SVR")
plt.xlabel("Nivel del empleado")
plt.ylabel("Sueldo en $")
plt.show()