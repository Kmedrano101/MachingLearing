# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:36:12 2021

@author: Kevin
"""

# Plantilla de Pre procesado - datos faltantes

# Importar las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

# Importar el Data Set

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Tratamiendo de los datos faltantes

imputer = Imputer(missing_values = "NaN", strategy="mean",axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])