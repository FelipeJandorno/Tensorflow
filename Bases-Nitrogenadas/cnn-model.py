import tensorflow as tf
import pandas as pd
import os
import numpy as np
# import xarray as xr
from sklearn.model_selection import train_test_split


# ================ PRÉ PROCESSAMENTO DE DADOS ================
# Lê o arquivo CSV
def read_files(path):
    df1 = pd.read_csv("CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV")
    df1 = df1.to_numpy()

    arr = np.array([
        df1
    ])

    for filename in os.listdir(path):
        df = pd.read_csv(path + filename)
        df = df.to_numpy()
        arr = np.insert(arr, arr.shape[0], [df], axis=0)
    # arr = np.delete(arr, [1, 0, 0], axis=0)
    # print(arr.shape)

    return arr
df_all = read_files("CSV/")
df_all = df_all.astype("float64")

# Procurando o maior valor do dataFrame df_all
def normalize_array(arr, wv_max_value=0, abs_max_value=0):
    # Separando o maior valor de comprimento de onda e de absorbância
    # try:
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if (arr[x][y][z] < 400) and wv_max_value < arr[x][y][z]:
                    wv_max_value = arr[z][y][x]
                elif (arr[x][y][z] >= 400) and abs_max_value < arr[x][y][z]:
                    abs_max_value = arr[x][y][z]
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if arr[x][y][z] < 400:
                    arr[x][y][z] = arr[x][y][z] / abs_max_value
                else:
                    arr[x][y][z] = arr[x][y][z] / wv_max_value
    return arr
df = normalize_array(df_all)

# Reformulação dos dados
# df = np.reshape(df, (df.shape[0], 1, df.shape[1], df.shape[2]))
# df = np.expand_dims(df, axis=2)

# ================ SEPARAÇÃO DE DADOS PARA TREINO E TESTE ================ #
train, test = train_test_split(df, train_size=0.8)

def train_target_group(train):
    train_target = np.array([])
    for i in range(train.shape[0]):
        train_target = np.append(train_target, i)
    return train_target.astype("int64")

train_target = train_target_group(train)
num_classes = np.max(train_target) + 1

train_target = tf.keras.utils.to_categorical(train_target, num_classes=num_classes)

train_target = np.array([[0, 0, 1],
                         [1, 0]], dtype=object)
# train = np.array(train)
# train_target = np.array(train_target)

# train_target = np.reshape(train_target, (1, train_target.shape[0], train_target[1]))

# ================ SEPARAÇÃO DAS FEATURES ================
print('train: ', train.shape)
print('target: ', train_target.shape)

# ================ CRIAÇÃO DO MODELO ================
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(2, 3, activation='relu', input_shape=(train.shape[1:])))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# ================ COMPILAÇÃO DO MODELO ================
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(train, train_target, epochs=2)


# Adaptar o train_target para virar (1, 12821, 1) e tornar a matriz em unidimensional
# Alterar método de renomeação e normalização dos dados
