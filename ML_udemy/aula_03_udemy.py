import tensorflow as tf
import numpy as np

tf_string = tf.constant("TensorFlow")

# Operações simples com strings
print("lenght: ", tf.strings.length(tf_string))
print("decode: ", tf.strings.unicode_decode(tf_string, 'UTF-8'))

# Armazenando arrays (vetores) de strings
tf_string_array = tf.constant(["TensorFlow", "Deep Learning", "AI"])

# Interação ao longo do tensor de string
for string in tf_string_array:
    print(string)

