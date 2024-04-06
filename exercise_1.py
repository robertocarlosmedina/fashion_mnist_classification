# imports que provalmente serao necessarios
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# constantes - dimensao das imagens
IMG_HEIGHT = 28
IMG_WIDTH = 28

# constantes - labels/classes
LABELS = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
N_CLASSES = 10

# callbacks
BEST_MODEL_CHECKPOINT = tf.keras.callbacks.ModelCheckpoint(
    filepath="tmp/best_model.weights.h5",      # ficheiro onde serao guardados os pesos do "melhor modelo"
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10)

# carregar o dataset
dataset = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# normalização
x_train = x_train / 255.0
x_test = x_test / 255.0

# transformar vetor das labels em matriz - adequado para rede multiclasse mas nao para a binaria
y_train = tf.keras.utils.to_categorical(y_train,N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test,N_CLASSES)

"""Continuar a partir da daqui. Resumo das tarefas:
   
   a)	Obter o conjunto de validação;
   b)	Construir o modelo;
   c)	Compilar a rede;
   d)	Treinar o modelo – max 50 épocas, de pref. a usar callbacks;
   e)	Gráfico que mostre a evolução do treino;
   f)	Cálculo dos acertos no conjunto de teste;
   g)	Mostrar a matriz de confusão."""

val_split = 0.2
val_size = int(len(x_train) * val_split)

x_val = x_train[-val_size:]
y_val = y_train[-val_size:]

x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# Construir o modelo
model = tf.keras.Sequential([
    layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1), input_shape=(IMG_HEIGHT, IMG_WIDTH)),  # Reshape to add channel dimension
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax')
])

# Compilar a rede
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train, epochs=50, batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[BEST_MODEL_CHECKPOINT, EARLY_STOPPING])

# Gráfico que mostra a evolução do treino
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Cálculo dos acertos no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Mostrar a matriz de confusão
y_pred = np.argmax(model.predict(x_test), axis=-1)
cm = confusion_matrix(np.argmax(y_test, axis=-1), y_pred, labels=np.arange(N_CLASSES))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot(cmap=plt.cm.Blues)
plt.show()