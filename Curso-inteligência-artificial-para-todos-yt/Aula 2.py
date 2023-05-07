from matplotlib import pyplot as plt
import pandas as pd # Criação de dataframes para manipulação de dados
import pylab as pl
import numpy as np

# sklearn é um framework para aprendizado de máquina
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# TRATAMENTO DE DADOS ========================= #
# Leitura do CSV com o pandas - dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')
# Exibe o header
print(df.head())

motores = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())
print(co2.head())

# test_size: define que 0.2 das amostras serão utilizadas para teste, enquanto o restante para treino
# random_state:
motores_treino, motores_teste, co2_treino, co2_teste = train_test_split(motores, co2, test_size = 0.2, random_state=42)
print(type(motores_treino))

# Exibir a correlação entre as features do dataset de treinamento
plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão CO2")
plt.show()

# Exibir a correlação entre as features do dataset de teste
plt.scatter(motores_teste, co2_teste, color='red')
plt.xlabel("Motores")
plt.ylabel("Emissão de CO2")
plt.show()