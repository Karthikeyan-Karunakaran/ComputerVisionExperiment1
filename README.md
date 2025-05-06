# 1: CNN FOR IMAGE CLASSIFICATION
## AIM:
 To implement a simple Convolutional Neural Network (CNN) for classifying images in the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 categories.
## PROCEDURE:
 1. Import and preprocess the CIFAR-10 dataset by normalizing the images.
 2. Build a CNN architecture including convolutional, pooling, and fully connected layers.
 3. Configure the model with the Adam optimizer and sparse categorical crossentropy as the loss function.
 4. Train the model using the training data.
5. Assess the model's performance on the test set and report the accuracy.
## CODE:
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Increased filter size
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Increased units
    layers.Dense(10, activation='softmax')  # Added softmax activation for output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Changed from_logits=False
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=12, validation_data=(x_test, y_test))  # Increased epochs

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')  # Display with 3 decimal points
```
## RESULT:
 The CNN model achieved an approximate test accuracy 80% on the CIFAR-10 dataset.
