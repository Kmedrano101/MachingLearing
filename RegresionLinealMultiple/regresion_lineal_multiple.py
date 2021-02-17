# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 08:59:57 2021

@author: Kevin
"""
# Regresion Lienal Multiple

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Metodo automatico para eliminacion hacia atras
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
# Importar el Data Set

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Codificar datos categoricos

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Evitar el problema de las variables dummy, eliminar uno columna

X = X[:,1:]

# Dividir en data set de entrenamiento y test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1)

# Ajustar el modelo de regresion lineal multiple

regresion = LinearRegression()
regresion.fit(X_train, y_train)

# Prediccion de los resultados con los datos de test

y_pred = regresion.predict(X_test) 

# Optimizar el modelo de RLM con eliminacion hacia atras
# axis >> 1 columnas 0 filas
X = np.append(arr=np.ones((50,1)).astype(int), values=X , axis= 1)
SL = 0.05
# Reajustar el modelo
X_optimo = X[:,[0,1,2,3,4,5]]
regresion_OLS = sm.OLS(endog = y,exog = X_optimo).fit()
regresion_OLS.summary()

X_optimo = X[:,[0,1,3,4,5]]
regresion_OLS = sm.OLS(endog = y,exog = X_optimo).fit()
regresion_OLS.summary()

X_optimo = X[:,[0,3,4,5]]
regresion_OLS = sm.OLS(endog = y,exog = X_optimo).fit()
regresion_OLS.summary()

X_optimo = X[:,[0,3,5]]
regresion_OLS = sm.OLS(endog = y,exog = X_optimo).fit()
regresion_OLS.summary()

X_optimo = X[:,[0,3]]
regresion_OLS = sm.OLS(endog = y,exog = X_optimo).fit()
regresion_OLS.summary()

# llamada al metodo automatico
SL = 0.05
X_modelo = backwardElimination(X, SL)

# Volvemos a trabajar con el nuevo modelo ajustado

X_train_modelo, X_test_modelo, y_train, y_test = train_test_split(X_modelo,y, test_size = 0.2, random_state=0)

# Ajustar el modelo de regresion lineal multiple

regresion_modelo = LinearRegression()
regresion_modelo.fit(X_train_modelo, y_train)

# Prediccion de los resultados con los datos de test

y_pred_modelo = regresion_modelo.predict(X_test_modelo) 


# Visualiar los datos
""" # Modelo con todas  las variables
plt.scatter(X_train[:,3], y_train, color = "red")
plt.plot(X_train[:,3], regresion.predict(X_train), color="blue")
plt.title("Start Ups Ingresos Entrenamiento")
plt.xlabel("Inversiones de las empresas")
plt.ylabel("Ingresos en $")
plt.show()

plt.scatter(X_test[:,3], y_test, color = "red")
plt.plot(X_train[:,3], regresion.predict(X_train), color="blue")
plt.title("Start Ups Ingresos Test")
plt.xlabel("Inversiones de las empresas")
plt.ylabel("Ingresos en $")
plt.show()"""

# Modelo con metodo de eliminacion hacia atras

plt.scatter(X_train_modelo[:,2], y_train, color = "red")
plt.plot(X_train_modelo[:,2], regresion_modelo.predict(X_train_modelo), color="blue")
plt.title("Start Ups Ingresos Entrenamiento Modelo Optimizado")
plt.xlabel("Inversiones de las empresas")
plt.ylabel("Ingresos en $")
plt.show()

plt.scatter(X_test_modelo[:,2], y_test, color = "red")
plt.plot(X_train_modelo[:,2], regresion_modelo.predict(X_train_modelo), color="blue")
plt.title("Start Ups Ingresos Test Modelo Optimizado")
plt.xlabel("Inversiones de las empresas")
plt.ylabel("Ingresos en $")
plt.show()


