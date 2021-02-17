# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:29:35 2021

@author: Kevin
"""

# Regresion Polinomica

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importar el Data Set

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

"""
# Dividir en data set de entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)
"""
# Escalado de variables de valores numericos muy grandes, normalizacion

"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
# Ajustar la regresion lineal con el dataset

regresionLineal = LinearRegression()
regresionLineal.fit(X, y)

# Ajustar la regresion polinomica con el dataset

regresionPolinomica = PolynomialFeatures(degree=4)
X_poly = regresionPolinomica.fit_transform(X)

regresionLineal2 = LinearRegression()
regresionLineal2.fit(X_poly,y)


# Visualizacion de resultados modelo lineal

plt.scatter(X, y, color = "red")
plt.plot(X, regresionLineal.predict(X), color="blue")
plt.title("Modelo de regresion Lineal")
plt.xlabel("Nivel del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# Visualizacion de resultados modelo polinomico
x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(x_grid, regresionLineal2.predict(regresionPolinomica.fit_transform(x_grid)), color="blue")
plt.title("Modelo de regresion Polinomica")
plt.xlabel("Nivel del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# Prediccion de los modelos
experiencia = np.array(6.5)
experiencia = experiencia.reshape(1,-1) # transformar a array de 1x1

regresionLineal.predict(experiencia)
regresionLineal2.predict(regresionPolinomica.fit_transform(experiencia))
