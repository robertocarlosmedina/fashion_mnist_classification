import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Constants - Image sizes
IMG_HEIGHT = 28
IMG_WIDTH = 28

# Constants - labels/classes
LABELS = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
N_CLASSES = 10
CHECKPOINT_PATH = "tmp/best_model_exerc_1.weights.h5"

# Callbacks
BEST_MODEL_CHECKPOINT = tf.keras.callbacks.ModelCheckpoint(
    # file where the weights of the "best model" will be stored
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15)

# Load dataset
dataset = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# transform label vector into matrix - suitable for multiclass network but not for binary
y_train = tf.keras.utils.to_categorical(y_train,N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test,N_CLASSES)

"""
Continue from here. Task summary:
   
   a) Obtain the validation set;
   b) Build the model;
   c) Compile the network;
   d) Train the model – max 50 epochs, preferably. to use callbacks;
   e) Graph showing the evolution of training;
   f) Calculation of hits in the test set;
   g) Show the confusion matrix.
"""

val_split = 0.2
val_size = int(len(x_train) * val_split)

x_val = x_train[-val_size:]
y_val = y_train[-val_size:]

x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# Build model structure
model = tf.keras.Sequential([
    # Reshape to add channel dimension
    layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1), input_shape=(IMG_HEIGHT, IMG_WIDTH)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax')
])

# Compile model Network
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
try:
    model.load_weights(CHECKPOINT_PATH)
except:
    print("Model weights file not found")

args_parser = argparse.ArgumentParser()

args_parser.add_argument(
    "-a", "--action", required=True, choices=["train", "test"],
    help="Actions regarding the model."
)

args = vars(args_parser.parse_args())

# Train Model
if args["action"] == "train":
    history = model.fit(x_train, y_train, epochs=50, batch_size=32,
                        validation_data=(x_val, y_val),
                        callbacks=[BEST_MODEL_CHECKPOINT, EARLY_STOPPING])

    # Graph that shows the evolution of training
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

# Calculation of hits on the test set
elif args["action"] == "test":
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(np.argmax(y_test, axis=-1), y_pred, labels=np.arange(N_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    
    # Plot confusion matrix with rotated x-labels
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)  # Rotate x-labels by 45 degrees
    plt.show()
