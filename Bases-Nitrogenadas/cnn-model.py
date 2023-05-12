import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import pyplot as plt

# ================ PRÉ PROCESSAMENTO DE DADOS ================ #
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

    # Juntando todos os dados em colunas no dataframe df_all
    df_all['Amostra {}'.format(filename[0])] = df['Amostra {}'.format(filename[0])]

# Procurando o maior valor do dataFrame
df_all = df_all.to_numpy(dtype="float64")
max_value = 0

for row in range(df_all.shape[0]):
    for col in range(df_all.shape[1]):
        if max_value < df_all[row][col]:
            max_value = df_all[row][col]

# Normalizando os dados do dataframe
for row in range(df_all.shape[0]):
    for col in range(df_all.shape[1]):
        df_all[row][col] = df_all[row][col]/max_value

# Transformando a matriz numpy normalizada em dataframe
df = pd.DataFrame(data=df_all, dtype="float64")

# Renomeando todas as colunas conforme sua numeração
for col in range(df.shape[1]):
    new_name = "Amostra {}".format(col)
    df = df.rename(columns={df.columns[col]: new_name})
print(df)
# ================ SEPARAÇÃO DE DADOS PARA TREINO E TESTE ================ #
train, test = train_test_split(df, train_size=0.8)

# ================ SEPARAÇÃO DAS FEATURES ================


# ================ CRIAÇÃO DO MODELO ================
print("======================TRAIN========================")
print(train)
print("===================TESTE===========================")
print(test)
print("===================DATAFRAME=======================")
print(df)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=23, activation='relu', kernel_size=3, padding='same', input_shape=(df.shape)))
model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
#model.add(tf.keras.layers.Conv2D(filters=66, activation='relu', kernel_size=6, padding='same'))
model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten(input_shape=(13062, 31)))
model.add(tf.keras.layers.Dense(units=330, activation='relu'))
model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

# ================ COMPILAÇÃO DO MODELO ================
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ================ TREINAMENTO DO MODELO ================
model.fit(train, epochs=3)

# ================ AVALIAÇÃO DO MODELO ================
#model.evaluate(test, verbose=2)