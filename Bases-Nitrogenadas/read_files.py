import pandas as pd
import numpy as np
import os

def read_files(path):
    df = pd.DataFrame(dtype="float64")
    arr = np.zeros((12821, 2))

    df = pd.read_csv("CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV")
    df = df.to_numpy()
    arr = np.dstack((arr, df))
    # for file in enumerate(os.listdir(path)):
        # df = pd.read_csv(path+file[1])
        # df = df.to_numpy()
        # arr = np.dstack((arr, df))
    # print(arr.shape)
    return arr
def read_files2(path):
    # 1 - Armazenando os dados dos dataframes em dicionários
    new_df_dict = {}
    for filename in enumerate(os.listdir(path)):
        df_dict = {
            "df{}".format(filename[0]): pd.read_csv(path + filename[1])
        }
        new_df_dict.update(df_dict)
        # print(new_df_dict)

    # 2 - Convertendo o dicionário em uma matriz numpy
    df_array = np.array(list(new_df_dict.items()), dtype=object)
    print(df_array)

read_files2("CSV/")