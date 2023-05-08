import tensorflow as tf
import pandas as pd
import numpy as np
#import pyplot as plt

# ================ PRÉ PROCESSAMENTO DE DADOS ================
# Lê o arquivo CSV
df_c = pd.read_csv('./CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV', delimiter=",")
df_g = pd.read_csv('./CSV/Amostra G - Thu Apr 27 17-42-21 2023 (GMT-03-00).CSV', delimiter=",")
df_h = pd.read_csv('./CSV/Amostra H - Thu Apr 27 18-39-46 2023 (GMT-03-00).CSV', delimiter=",")

#pd.set_option('float format', '{:.3f}'.format)

# Transforma os dataframes em matrizes numpy
df_c = df_c.to_numpy()
df_g = df_g.to_numpy()
df_h = df_h.to_numpy()

df_c = df_c.astype('float64')
df_g = df_g.astype('float64')
df_h = df_h.astype('float64')

# Adicionando as classes às matrizes
#df_c = np.vstack((["Wavenumber", "Absorbance"], df_c))
#df_g = np.vstack((["Wavenumber", "Absorbance"], df_g))
#df_h = np.vstack((["Wavenumber", "Absorbance"], df_h))

# Transformando as matrizes numpy em um dataframe (após o tratamento de dados)
df_c = pd.DataFrame(data=df_c)
df_g = pd.DataFrame(data=df_g)
df_h = pd.DataFrame(data=df_h)

# ================ SEPARAÇÃO DAS FEATURES ================
y_class = ['Adenina', 'Citosina']

# ================ CRIAÇÃO DO MODELO ================

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=60, input_shape=(1, 12821, 1), activation='relu', kernel_size=3, padding='same'))
model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Conv1D(filters='30', activation='relu', kernel_size=3, padding='same'))
model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

