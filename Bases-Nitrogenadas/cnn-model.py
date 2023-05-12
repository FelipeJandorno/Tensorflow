import os
import tensorflow as tf
import pandas as pd
import numpy as np
#import sklearn as sk
#import pyplot as plt

# ================ PRÉ PROCESSAMENTO DE DADOS ================
# Lê o arquivo CSV

df_all = pd.DataFrame(dtype="float64")
path = "CSV/"

for filename in enumerate(os.listdir(path)):
    # Leitura do arquivo CSV
    df = pd.read_csv(path+filename[1])

    # Alterando o tipo de dado do dataframe
    df = df.astype('float64')

    # Adicionando a label das colunas no dataframe
    df.columns = ['Wavenumber', 'Amostra {}'.format(filename[0])]

    # Juntando todos os dados em colunas
    df_all['Amostra {}'.format(filename[0])] = df['Amostra {}'.format(filename[0])]

# Procurando o maior valor do dataFrame
df_all = df_all.to_numpy(dtype="float64")
max_value = 0

for row in range(df_all.shape[0]):
    for col in range(df_all.shape[1]):
        if max_value < df_all[row][col]:
            max_value = df_all[row][col]

# Normalizando os dados
for row in range(df_all.shape[0]):
    for col in range(df_all.shape[1]):
        df_all[row][col] = df_all[row][col]/max_value

# Transformando a matriz numpy normalizada em dataframe
df = pd.DataFrame(data=df_all, dtype="float64")

# Renomeando todas as colunas conforme sua numeração
for col in range(df.shape[1]):
    new_name = "Amostra {}".format(col)
    df = df.rename(columns={df.columns[col]: new_name})

# ================ SEPARAÇÃO DE DADOS PARA TREINO E TESTE ================
#(x_train, y_train), (x_test, y_test) =

# ================ SEPARAÇÃO DAS FEATURES ================
#y_class = ['Wavenumber', 'Adenine', 'Citosine', 'Ade + Cit']

# ================ CRIAÇÃO DO MODELO ================

#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Conv2D(filters=35, activation='relu', kernel_size=3, padding='same'))

# ================ COMPILAÇÃO DO MODELO ================
#model.compile()


