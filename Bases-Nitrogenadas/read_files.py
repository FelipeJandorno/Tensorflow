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

def read_files3(path):
    arr = np.array([[]])
    for file in enumerate(os.listdir(path)):
        df = pd.read_csv(path+file[1])
        df = df.to_numpy()
        print(arr.shape)
        arr = np.insert(arr, df.shape[0], [df], axis=0)

def read_files4():
    df1 = pd.read_csv("CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV")
    df2 = pd.read_csv("CSV/Amostra H - Thu Apr 27 18-39-46 2023 (GMT-03-00).CSV")
    df3 = pd.read_csv("CSV/Amostra G - Thu Apr 27 17-42-21 2023 (GMT-03-00).CSV")

    df1 = df1.to_numpy()
    df2 = df2.to_numpy()
    df3 = df3.to_numpy()

    arr = np.array([
        df1,
        df2
    ])

    arr = np.insert(arr, arr.shape[0], [df3], axis=0)

    print(arr.shape)

# read_files4()

def read_file5(path):
    df1 = pd.read_csv("CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV")
    df1 = df1.to_numpy()

    arr = np.array([
        df1
    ])

    for filename in os.listdir(path):
        df = pd.read_csv(path+filename)
        df = df.to_numpy()
        arr = np.insert(arr, arr.shape[0], [df], axis=0)
    arr = np.delete(arr, [1, 0, 0], axis=0)
    print(arr.shape)

# ========================================= #
arr = np.array([
    #abs, wv
    [[1, 0],
    [4, 1],
    [5, 0]],
    [[3, 0],
     [8, 0],
     [2, 0]]
])

print('x: {}, y: {}, z: {}'.format(arr.shape[0], arr.shape[1], arr.shape[2]))
def norm_arr(arr, wv_max_value=0, abs_max_value=0):
# Normalizando os valores de absorbancia
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(arr.shape[2]):
                if x==1 and wv_max_value < arr[z][y][x]:
                    wv_max_value = arr[z][y][x]
                elif x==0 and abs_max_value < arr[z][y][x]:
                    abs_max_value = arr[z][y][x]
    return abs_max_value, wv_max_value
print(norm_arr(arr))