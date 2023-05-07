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
#plt.show()

# Exibir a correlação entre as features do dataset de teste
plt.scatter(motores_teste, co2_teste, color='red')
plt.xlabel("Motores")
plt.ylabel("Emissão de CO2")
#plt.show()

# ================= CRIAÇÃO E TREINAMENTO DO MODELO =================
modelo = linear_model.LinearRegression()
modelo.fit(motores_treino, co2_treino)

# Exibindo as saídas do modelo
print('(A) Intercepto: ', modelo.intercept_) # valor de y quando x = 0
print('(B) Inclinação: ', modelo.coef_) # coeficiente angular da reta

# ================= PREDIÇÕES DO MODELO =================
predicoesCo2 = modelo.predict(motores_teste)
#print(predicoesCo2)

plt.scatter(motores_teste, co2_teste, color='purple')
plt.plot(motores_teste, modelo.coef_[0][0]*motores_teste + modelo.intercept_[0], '-r')
plt.xlabel('Motores')
plt.ylabel('Emissão de CO2')
plt.show()

# ================= MÉTRICAS DO MODELO =================
print('=================== RELATÓRIO FINAL ===================')
print('Soma dos erros ao quadrado (SSE): %.2f' % np.sum((predicoesCo2 - co2_teste)**2))
print('Erro quadrático médio (MSE): %.2f' % mean_squared_error(co2_teste, predicoesCo2))
print('Erro médio absoluto: %.2f' % mean_absolute_error(co2_teste, predicoesCo2))
print('Raiz do erro quadrático médio (RMSE): %.2f' % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print('R2-score: %.2f' % r2_score(predicoesCo2, co2_teste))