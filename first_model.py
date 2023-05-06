import tensorflow as tf
import numpy as np

# Importa a base de dados que contém as imagens para realizar a classificação
mnist = tf.keras.datasets.mnist
# Separa o conjunto de dados para alimentar a IA e outro conjunto para testar
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Padroniza os dados como pontos flutuantes entre 0 e 1
x_train, x_test = x_train/255.0, x_test/255.0

# Criação do modelo de inteligência artificial
model = tf.keras.Sequential([
    # Transforma a matriz multidimensional em um vetor de 28x28 = 784
    tf.keras.layers.Flatten(input_shape = (28, 28)),

    # Cria 128 nós (neurônios)
    tf.keras.layers.Dense(128, activation='relu'),

    # Camada que define aleatoriamente certas unidades para zero com uma frequência
    # de 0.2 durante cada etapa do treino. Tal prática evita Overfitting, isto é,
    # um treinamento que funciona bem com dados antigos, contudo, mal com dados novos
    tf.keras.layers.Dropout(0.2),


    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose = 2)