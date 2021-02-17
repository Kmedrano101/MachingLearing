# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:05:10 2021

@author: Kevin
"""

# Plantilla de preprocesado

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importar el Data Set

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


# Dividir en data set de entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

# Escalado de variables de valores numericos muy grandes, normalizacion

"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


