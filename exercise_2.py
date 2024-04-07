import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers  # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

args_parser = argparse.ArgumentParser()

args_parser.add_argument(
    "-a", "--action", required=True, choices=["train", "test"],
    help="Actions regarding the model."
)

args_parser.add_argument(
    "-mn", "--model-name", required=False, type=str,
    help="Specify model name you want to use."
)

args = vars(args_parser.parse_args())


BEST_MODEL_CHECKPOINT = tf.keras.callbacks.ModelCheckpoint(
    # file where the weights of the "best model" will be stored
    filepath=f"tmp/{'best_model_exerc_2.weights.h5' if not args['model_name'] else args['model_name'] + '.weights.h5'}",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15)

# Map original labels to new binary labels: "Apparel" (1) and "Footwear and Bags" (0)
# Apparel classes: 0 (T-Shirt/Top), 3 (Dress), 4 (Coat), 6 (Shirt)
# Footwear and Bags classes: 1 (Trouser), 2 (Pullover), 5 (Sandal), 7 (Sneaker), 8 (Bag), 9 (Boot)
binary_label_map = {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0}
y_train_binary = np.array([binary_label_map[label] for label in y_train])
y_test_binary = np.array([binary_label_map[label] for label in y_test])

# Define model architecture
model = tf.keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
try:
    model.load_weights(
        f"tmp/{'best_model_exerc_2.weights.h5' if not args['model_name'] else args['model_name'] + '.weights.h5'}")
except:
    print("Model weights file not found")

# Treinar o modelo
if args["action"] == "train":
    # Train the model
    history = model.fit(x_train, y_train_binary, epochs=50, batch_size=32,
                        validation_split=0.2, callbacks=[BEST_MODEL_CHECKPOINT, EARLY_STOPPING])

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

# Evaluate the model
elif args["action"] == "test":
    test_loss, test_acc = model.evaluate(x_test, y_test_binary)
    print('Test accuracy:', test_acc)

    # Confusion matrix
    y_pred = np.round(model.predict(x_test)).astype(int)
    cm = confusion_matrix(y_test_binary, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                  "Footwear and Bags", "Apparel"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
