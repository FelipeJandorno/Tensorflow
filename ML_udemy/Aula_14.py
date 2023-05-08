import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# ====================== PRÉ PROCESSAMENTO ======================

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255, x_test/255
#print(x_train.shape)
#print(x_test.shape)

# ====================== CONSTRUÇÃO DA REDE NEURAL ======================

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                                 activation='relu', input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                                  activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
#model.summary()

# ====================== COMPILANDO O MODELO ======================
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['sparse_categorical_accuracy'])

# ====================== TREINANDO O MODELO ======================
model.fit(x_train, y_train, epochs=5)

# ====================== AVALIAÇÃO DO MODELO ======================
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy {}'.format(test_acc))
print('Test loss {}'.format(test_loss))