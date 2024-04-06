import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

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

# Train the model
history = model.fit(x_train, y_train_binary, epochs=10, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test_binary)
print('Test accuracy:', test_acc)

# Confusion matrix
y_pred = np.round(model.predict(x_test)).astype(int)
cm = confusion_matrix(y_test_binary, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Footwear and Bags", "Apparel"])
disp.plot(cmap=plt.cm.Blues)
plt.show()