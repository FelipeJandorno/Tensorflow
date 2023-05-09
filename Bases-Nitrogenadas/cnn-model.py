import tensorflow as tf
import pandas as pd
import numpy as np
#import sklearn as sk
#import pyplot as plt

# ================ PRÉ PROCESSAMENTO DE DADOS ================
# Lê o arquivo CSV
df_c = pd.read_csv('./CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV', delimiter=";")
df_g = pd.read_csv('./CSV/Amostra G - Thu Apr 27 17-42-21 2023 (GMT-03-00).CSV', delimiter=";")
df_h = pd.read_csv('./CSV/Amostra H - Thu Apr 27 18-39-46 2023 (GMT-03-00).CSV', delimiter=";")

# Adicionando labels no dataframe
df_c.columns = ['Wavenumber', 'Adenine']
df_g.columns = ['Wavenumber', 'Citosine']
df_h.columns = ['Wavenumber', 'Ade + Cit']

# Trocando as vírgulas dos dados por ponto
df_c['Wavenumber'], df_c['Adenine'] = df_c['Wavenumber'].str.replace(',', '.'), df_c['Adenine'].str.replace(',', '.')
df_g['Wavenumber'], df_g['Citosine'] = df_g['Wavenumber'].str.replace(',', '.'), df_g['Citosine'].str.replace(',', '.')
df_h['Wavenumber'], df_h['Ade + Cit'] = df_h['Wavenumber'].str.replace(',', '.'), df_h['Ade + Cit'].str.replace(',', '.')

# Alterando o tipo de variável dos dataframes
df_c['Wavenumber'], df_c['Adenine'] = df_c['Wavenumber'].astype('float64'), df_c['Adenine'].astype('float64')
df_g['Wavenumber'], df_g['Citosine'] = df_g['Wavenumber'].astype('float64'), df_g['Citosine'].astype('float64')
df_h['Wavenumber'], df_h['Ade + Cit'] = df_h['Wavenumber'].astype('float64'), df_h['Ade + Cit'].astype('float64')

# Juntando dados em colunas diferentes
df = df_c
df['Citosine'] = df_g['Citosine']
df['Ade + Cit'] = df_h['Ade + Cit']
print(df.dtypes)

# ================ SEPARAÇÃO DE DADOS PARA TREINO E TESTE ================
#(x_train, y_train), (x_test, y_test) =

# ================ SEPARAÇÃO DAS FEATURES ================
y_class = ['Wavenumber', 'Adenine', 'Citosine', 'Ade + Cit']

# ================ CRIAÇÃO DO MODELO ================

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=35, activation='relu', kernel_size=3, padding='same'))

# ================ COMPILAÇÃO DO MODELO ================
model.compile()


