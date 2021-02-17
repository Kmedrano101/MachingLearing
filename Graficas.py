# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:49:26 2021

@author: Kevin
"""

# Desarrollo de defirentes graficas 

# Importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

# Importar los datos
datos = pd.read_csv('Titanic/train.csv')
df = pd.DataFrame(datos)

# Descripcion basica de los datos para un previo analisis 
descripcion_basica = df.describe()
# Agrupaciones de los sobrevivientes respecto a las variables categoricas
df.groupby('Sex')['Survived'].sum().plot(title="Sobrevivientes por Sexo",kind='bar',legend='Reverse',color='C3')
plt.show()
df.groupby('Pclass')['Survived'].sum().plot(title="Sobrevivientes por Clase",kind='bar',legend='Reverse',color='C3')
plt.show()
df.groupby('SibSp')['Survived'].sum().plot(title="Sobrevivientes por SibSP",kind='bar',legend='Reverse',color='C3')
plt.show()
df.groupby('Parch')['Survived'].sum().plot(title="Sobrevivientes por Parch",kind='bar',legend='Reverse',color='C3')
plt.show()
df.groupby('Embarked')['Survived'].sum().plot(title="Sobrevivientes por Embarked",kind='bar',legend='Reverse',color='C3')
plt.show()


# Otra forma de agrupar los datos
#df.Survived.groupby(df.Sex).sum().plot(title="Sobrevivientes por Sexo",kind='bar',legend='Reverse',color='C3')