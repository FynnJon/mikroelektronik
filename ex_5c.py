import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import keras
from keras import layers


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5)),
        #layers.Flatten(),
        #layers.Dense(num_classes, activation="softmax"),
    ]
)

model.load_weights('saved_weights/my_weights')
model.summary()
weights = model.get_weights()
print(weights)

'''
x_example = x_test[0, :, :, 0]
x_example = np.reshape(x_example, (1, 28, 28, 1))
y_pred = model.predict_step(x_example)

plt.imshow(y_pred[0, :, :, 0], cmap=plt.get_cmap('gray'))
plt.show()
plt.imshow(y_pred[0, :, :, 1], cmap=plt.get_cmap('gray'))
plt.show()
plt.imshow(y_pred[0, :, :, 2], cmap=plt.get_cmap('gray'))
plt.show()
plt.imshow(y_pred[0, :, :, 31], cmap=plt.get_cmap('gray'))
plt.show()
'''
