# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:52:42 2021

@author: Kevin
"""
# Concurso Kaggle Titanic ML

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Definicion de metodos

# Importamos el dataset

datasetTRAIN = pd.read_csv('train.csv')
X_train = datasetTRAIN.iloc[:,[2,4,5,6,7,9]].values
y_train = datasetTRAIN.iloc[:,[1]].values

#Descripcion basica de los datos
descripcionBasica = datasetTRAIN.describe()

datasetTEST = pd.read_csv('test.csv')
X_test = datasetTEST.iloc[:,[1,3,4,5,6,8]].values
#y_test = datasetTEST.iloc[:,:].values
# Determinar el porcentaje de hobres y mujeres

def  graficarBar(variable):
    # Obtener caracteristica
    var = datasetTRAIN [variable]
    # Cantidad de categorias variable(value/sample)
    varValue = var.value_counts()
    
    # Visualizar
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequencia")
    plt.title(variable)
    plt.show()

def graficarBarNumeros(variable):
    plt.figure(figsize = (9,3))
    plt.hist(datasetTRAIN[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequencia")
    plt.title(" {} Histrograma de distribucion".format(variable))
    plt.show()
    
ListaSex = list(datasetTRAIN['Sex'].values)
num_mujeres,num_hombres = 0,0
#num_hombres = datasetTRAIN.apply(lambda x:True if x[4]=='male' else False, axis='Sex')

for i in ListaSex:
    if i == 'male':
        num_hombres +=1
    else:
        num_mujeres += 1
porc_hombres = num_hombres/len(ListaSex) * 100
porc_mujeres = num_mujeres/len(ListaSex) * 100

print("Hombres: "+str(num_hombres)+" Porcentaje: "+str(porc_hombres))
print("Mujeres: "+str(num_mujeres)+" Porcentaje: "+str(porc_mujeres))

# Visualizar los datos
# Analisar datos categoricos
columnas = ["Survived","Sex","Pclass","Embarked"]
for i in columnas:
    graficarBar(i)
# Analizar datos numericos
columnas = ["Fare","Age"]
for i in columnas:
    graficarBarNumeros(i)
#porcentaje_hombres = sum(lista_hombres)/889
#print(lista_hombres)

# Determinar el porcentaje de muertes de hombres y mujeres

mujeres_vive = datasetTRAIN.loc[datasetTRAIN.Sex == 'female']["Survived"]
rate_mujeres = sum(mujeres_vive)/len(mujeres_vive) * 100

hombres_vive = datasetTRAIN.loc[datasetTRAIN.Sex == 'male']["Survived"]
rate_hombres = sum(hombres_vive)/len(hombres_vive) * 100

print("% Hombres que sobreviven: "+str(rate_hombres))
print("% Mujeres que sobreviven: "+str(rate_mujeres))

# Datos faltantes
valoresNATrain = datasetTRAIN.isna().sum()
valoresNATest = datasetTEST.isna().sum()

# Age = 177 Cabin = 687 Embarked = 2 conjunto de train
# Age = 86 Cabin = 327 Fare = 1 conjunto de test

# Corregir datos faltantes de edad conjunto train
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train[:,[2]])
X_train[:,[2]] = imputer.transform(X_train[:,[2]])

# Corregir datos faltantes de edad conjunto test
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_test[:,[2]])
X_test[:,[2]] = imputer.transform(X_test[:,[2]])

imputer = imputer.fit(X_test[:,[5]])
X_test[:,[5]] = imputer.transform(X_test[:,[5]])

# Codificar datos categoricos

# Conjunto de Train
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X_train = onehotencoder.fit_transform(X_train).toarray()
#y = datasetTRAIN["Survived"]
# Conjunto de Test
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X_test = onehotencoder.fit_transform(X_test).toarray()

#y = datasetTRAIN["Survived"]

#Opciones = ["Pclass", "Sex", "SibSp", "Parch"]
#X = pd.get_dummies(datasetTRAIN[Opciones])
#X_test_2 = pd.get_dummies(datasetTEST[Opciones])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculando sin procesar los datos faltantes se obtiene 35.40669856% que sobreviven
# Calculando sin procesar los datos faltantes se obtiene 35.88516746% que sobreviven

valorSobrevividos = y_pred.sum() / len(y_pred) * 100
print("Porcentaje Vivos conjunto Test: "+str(valorSobrevividos))

salida = pd.DataFrame({'PassengerId': datasetTEST.PassengerId, 'Survived': y_pred})
salida.to_csv('gender_submission.csv', index=False)
#print("Tu entrega a sido guardado con exito!")
# Visualizar los datos
"""cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Prediccion")
plt.ylabel("Test")"""



