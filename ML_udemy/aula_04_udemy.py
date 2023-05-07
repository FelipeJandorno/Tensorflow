import tensorflow as tf
import numpy as np

fs_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fs_mnist.load_data()

x_train, x_test = x_train/255, x_test/255

# Altera o formato da matriz para um vetor
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Construindo a rede neural
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

# Zera 0.2 neurônios da camada oculta anterior (128*0.2 = 25.6)
model.add(tf.keras.layers.Dropout(0.15))
# Como temos um problema de classificação com mais de duas calsses, utiliza-se o softmax
# Se fossem apenas duas classes, utilizariamos a função de ativação sigmoide
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compilando o modelo
# Loss: É a forma que faremos o cálculo do erro
# Metrics: Avaliação do percentual de acerto
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')
model.summary()

# Treinando o modelo
# epochs: número de ciclos de treinamento realizados pela IA
model.fit(x_train, y_train, epochs=10)

# Análise do modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_acc)) # Precisão
print("Test loss: {}".format(test_loss)) # Erro