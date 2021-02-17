# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:35:42 2021

@author: Kevin
"""
# Plantilla de Pre procesado - datos categoricos

# Importar las librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importar el Data Set

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


# Codificar datos categoricos

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()

#labelencoder_y = LabelEncoder()
#sy = labelencoder_y.fit_transform(y)