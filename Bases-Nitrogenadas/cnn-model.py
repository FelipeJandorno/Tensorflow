import tensorflow as tf
import pandas as pd
import os
import numpy as np
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
    arr = np.delete(arr, [1, 0, 0], axis=0)
    print(arr.shape)

    return arr
df_all = read_files("CSV/")

# Leitura do arquivo 2° tentativa
# Convertendo os dataframes para numpy
df_all = df_all.astype("float64")
# df_wv = df_wv.to_numpy(dtype="float64")

# Procurando o maior valor do dataFrame df_all
def normalize_data(df_all, max_value = 0):
    try:
        for row in range(df_all.shape[0]):
            for col in range(df_all.shape[1]):
                if max_value < df_all[row][col]:
                    max_value = df_all[row][col]

        # Normalizando os dados do dataframe df_all
        for row in range(df_all.shape[0]):
                for col in range(df_all.shape[1]):
                     df_all[row][col] = df_all[row][col]/max_value
    except IndexError:
        for row in range(df_all.shape[0]):
            if row > max_value:
                max_value = row

            # Normalizando os dados do dataframe df_wv
        for row in range(df_all.shape[0]):
            df_all[row] = df_all[row] / max_value
    except ValueError:
        print('error')
        return 0
    # RETOMAR A PARTIR DESTE PONTO !!!!!!!!!
    return df_all

df_all = normalize_data(df_all)
# df_wv = normalize_data(df_wv)

# Transformando a matriz numpy normalizada em dataframe
df = pd.DataFrame(data=df_all, dtype="float64")
# df_wv = pd.DataFrame(data=df_wv, dtype="float64")

# Renomeando todas as colunas conforme sua numeração
for col in range(df.shape[1]):
    new_name = "Amostra {}".format(col)
    df = df.rename(columns={df.columns[col]: new_name})

# Adicionando a coluna de Wavenumber no dataframe
# df['Wavenumber'] = df_wv

df = df.to_numpy()
new_arr = []

#for row in enumerate(df):
#new_arr = new_arr.concat(np.array([row[0]]))

#print(new_arr.shape)
#print(new_arr)

# print('dim: ', df.ndim)
# print(df)
# ================ SEPARAÇÃO DE DADOS PARA TREINO E TESTE ================ #
# train, test = train_test_split(df, train_size=0.8)
# print('train: ', train.shape, ' test: ', test.shape)

# ================ SEPARAÇÃO DAS FEATURES ================

# ================ CRIAÇÃO DO MODELO ================


# ================ COMPILAÇÃO DO MODELO ================
