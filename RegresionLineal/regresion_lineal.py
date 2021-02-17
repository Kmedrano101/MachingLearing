# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:37:46 2021

@author: Kevin
"""

# Regresion Lineal Simple

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Importar el Data Set

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# Dividir en data set de entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state=0)

# Escalado de variables de valores numericos muy grandes, normalizacion

"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Crear modelo de Regresion Lineal Simple con datos de entrenamiento
# No es necesario normalizar las variables en este tipo de modelo
regresion = LinearRegression()
regresion.fit(X_train, y_train)

# Predecir el conjunto de test

y_prediccion = regresion.predict(X_test)

# Visualiar los datos
print(X_train)

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regresion.predict(X_train), color="blue")
plt.title("Sueldos vs Tiempo Experiencia Entrenamiento")
plt.xlabel("Tiempo de experiencia")
plt.ylabel("Sueldo en$")
plt.show()

plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regresion.predict(X_train), color="blue")
plt.title("Sueldos vs Tiempo Experiencia Test")
plt.xlabel("Tiempo de experiencia")
plt.ylabel("Sueldo en$")
plt.show()








