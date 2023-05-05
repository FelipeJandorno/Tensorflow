import tensorflow as tf
import numpy as np

# Operações com tensores
tensor = tf.constant([[1., 2.], [3., 4.]])
tensor_20 = tf.constant([[23., 4.],[32., 51.]])

# Adição entre um escalar e um tensor
print("Add escalar e tensor: ", tensor + 2)

# Subtração entre um escalar e um tensor
print("Sub escalar e tensor: ", tensor - 3)

# Multiplicação entre um escalar e um tensor
print("Mult escalar e tensor: ", tensor * 5)

# Divisão entre um escalar e um tensor
print("Div escalar e tensor: ", tensor/2)

# ================================================== #

# Usando funções numpy nos tensores do TensorFlow
print("Quadrado (^2): ", np.square(tensor))
print("Sqrt: ", np.sqrt(tensor))

# Dot product (produto escalar) entre dois tensores
print("dot: ", np.dot(tensor, tensor_20))
# ================================================== #
