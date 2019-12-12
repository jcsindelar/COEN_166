import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
print(tf.__version__)
fashion_mnist = datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # the 1st 2d-convolutional layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation= 'relu'))
model.add(layers.Dense(10, activation= 'softmax'))
model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5,batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
y_test_hat_mat = model.predict(test_images)
y_test_hat = np.argmax(y_test_hat_mat, axis=1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_labels, y_test_hat, labels=range(10))
print(cm)