#Link de mapa de expressões do tensorflow 1.12 para o 2
# https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0

import tensorflow as tf
import numpy as np

# Dados importantes do tensor
tensor_02 = tf.constant([[23, 4], [32, 51]])
print("======== tensor =========")
print(tensor_02)
print("======== shape ==========")
print(tensor_02.shape)
print("======== dtype ==========")
print(tensor_02.dtype)

# Converte o tensor para o numpy
print("==== tensor to numpy ====")
print(tensor_02.numpy())

# Conversão de uma matriz numpy para tensor
numpy_array = np.array([[23, 4], [32, 51]])
tensor_from_np = tf.constant(numpy_array)
print("==== numpy to tensor ====")
print(tensor_from_np)

# ================================================= #

# Definindo uma variável
tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

# Acessando valores de uma variável: É necessário utilizar o np
tf2_variable.numpy()

# Alterando um valor específico de uma variável: A função assign é do tensor
tf2_variable[0, 2].assign(100)
print(tf2_variable)
# ================================================= #

