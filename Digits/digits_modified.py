import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the MNIST dataset - a classic dataset of handwritten digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training data shape: {X_train.shape[0]} images")

# Reshape the data to 4D tensor as required by Conv2D layers
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalize the pixel values to be between 0 and 1
X_train /= 255
X_test /= 255

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the CNN model
model = Sequential()

# First convolutional layer with 48 filters and 3x3 kernel size
model.add(Conv2D(48, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Dropout(0.3))  # Slightly increased dropout to prevent overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling layer to reduce spatial dimensions

# Second convolutional layer with 36 filters
model.add(Conv2D(36, (3, 3), activation='relu', padding='valid'))

# Flatten the output to feed it into fully connected layers
model.add(Flatten())

# Fully connected layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Batch normalization to stabilize learning
model.add(BatchNormalization(axis=-1))

# Output layer with 10 units (one for each digit) and softmax activation
model.add(Dense(10, activation='softmax'))

# Split training data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Compile the model with Adam optimizer and a slightly lower learning rate
model.compile(Adam(learning_rate=0.0025), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 12 epochs with a batch size of 100
eval = model.fit(X_train, y_train, epochs=10, batch_size=150, validation_data=(X_val, y_val))

# Evaluate the model on the test data
testLoss, testAcc = model.evaluate(X_test, y_test)
print('Test Accuracy:', testAcc)

# Plotting the training and validation accuracy
plt.plot(eval.history['accuracy'], label='Training Accuracy')
plt.plot(eval.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predicting on test data and evaluating metrics
y_pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Identify and visualize some misclassified examples
error_id = np.where(y_pred != y_true)[0]

plt.figure(figsize=(10, 10))
for i, id in enumerate(error_id[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[id].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_true[id]} Pred: {y_pred[id]}')
    plt.axis('off')
plt.show()
