# O tutorial utilizado para a realização deste código poderá ser encontrado
# em: https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br

# Este código busca treinar um modelo de rede neural para classificação de
# imagens de roupas, como tênis e camisetas

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importação de dados do banco de dados fashion_mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Pre-processamento dos dados
train_images, test_images = train_images/255, test_labels/255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Verificando conjunto de dados
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Construção do modelo
model = tf.keras.Sequential([
    # Transforma o formato da imagem de um array de duas dimensões em um vetor
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Cria 128 nós
    tf.keras.layers.Dense(128, activation='relu'),

    # É uma camada softmax de 10 nós que retorna uma array de 10 probabilidades
    # nessa array, cada elemento é atribuido com um valor entre 0 e 1 de modo que
    # a soma de todos os valores seja equivalente a 1. Dessa forma, o unitário que
    # obtiver maior valor dentre todos, é aquele que possui maior probabilidade de
    # ser o objeto lido
    tf.keras.layers.Dense(10, activation='softmax')
])

# Função Loss: mede a precisão do modelo durante o treinamento
# Optimizer: Como o modelo se atualiza com base no dado que está analisando
# Métricas: Monitora o desempenho do modelo ao longo do treinamento e do teste

# Compilando o modelo
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(train_images, train_labels, epochs=10)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

# Fazendo predições
predictions = model.predict(test_images)