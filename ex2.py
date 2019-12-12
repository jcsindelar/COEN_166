import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
# load the data set
fashion_mnist = datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
# display the first 10 training images 
for t in range(10):  
  img = x_train[t,:,:]    
  plt.imshow(img,cmap="gray")    
  plt.show()

# normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(10))
model.add(layers.Dense(1568, activation='relu'))
model.add(layers.Dense(784))
model.add(layers.Reshape([784],input_shape=(784,)))
model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64)
loss_test, acc_test = model.evaluate(x_test, y_test)
y_test_hat_mat = model.predict(x_test)
y_test_hat = np.argmax(y_test_hat_mat, axis=1)

for t in range(10):  
  img = x_train[t,:,:]    
  plt.imshow(img,cmap="gray")    
  plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_test_hat, labels=range(10))
print(cm)