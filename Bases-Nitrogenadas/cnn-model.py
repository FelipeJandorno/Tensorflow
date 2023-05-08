import tensorflow as tf
import pandas as pd
#import pyplot as plt

# ================ PRÉ PROCESSAMENTO DE DADOS ================
# Lê o arquivo CSV
df = pd.read_csv('./CSV/Amostra C - Thu Apr 27 17-28-37 2023 (GMT-03-00).CSV', delimiter=";")

# Separa os maiores valores de Absorbancia e de n° de onda e normaliza os dados
max_absorbtion_value = df.max(axis=0)
df['Absorbance'] = df['Absorbance']/max_absorbtion_value['Absorbance']

#