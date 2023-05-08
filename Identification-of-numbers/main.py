import tensorflow as tf
import pandas as pd

# ============== PRÉ-PROCESSAMENTO DE DADOS ==============
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# ============== CRIAÇÃO DO MODELO ==============

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=784, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# ============== TREINAMENTO DO MODELO ==============
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['sparse_categorical_accuracy'])

# ============== TESTE DO MODELO ==============
model.fit(x_train, y_train, epochs=3)
model.summary()

# ============== MÉTRICAS DO MODELO ==============
model_loss, model_acc = model.evaluate(x_test, y_test)
print('loss: {}'.format(model_loss))
print('accuracy: {}'.format(model_acc))