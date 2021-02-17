# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:06:58 2021

@author: Kevin
"""

# Arboles de decision

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importar el Data Set

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Ajustar la regulacion con el dataset

regresion = DecisionTreeRegressor(random_state=0)
regresion.fit(X,y)

dato = np.array(6.5).reshape(1,-1)
y_pred = regresion.predict(dato)

# Visualizar los datos

plt.scatter(X, y, color = "red")
plt.plot(X, regresion.predict(X), color="blue")
plt.title("Modelo de regresion Arboles de Decision")
plt.xlabel("Nivel del empleado")
plt.ylabel("Sueldo en $")
plt.show()


