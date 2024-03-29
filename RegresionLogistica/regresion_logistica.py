# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 07:43:11 2021

@author: Kevin
"""
# Regresion Logistica

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
# Importamos el dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Dividir en data set de entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)

# Escalado de variables de valores numericos muy grandes, normalizacion

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el modelo de Regresion Logistica en entrenamiento

clasificador = LogisticRegression(random_state=0)
clasificador.fit(X_train, y_train)

# Prediccion de los resultados con el conjunto de Test

y_pred = clasificador.predict(X_test)

# Elaborar la matriz de confusion

cmatriz = confusion_matrix(y_test, y_pred)

# Visualizar los resultados del modelo

X_set, y_set = X_train, y_train

# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clasificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()






