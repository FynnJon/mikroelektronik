# Verwenden des conv2d Layers und exportieren von Daten in txt

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
        layers.Conv2D(32, kernel_size=(5, 5), use_bias=False),
        #layers.Flatten(),
        #layers.Dense(num_classes, activation="softmax"),
    ]
)

model.load_weights('saved_weights/my_weights')
model.summary()
weights = np.array(model.get_weights()[0])


x_example = x_test[0, :, :, 0]
x_example = np.reshape(x_example, (1, 28, 28, 1))
y_pred = model.predict_step(x_example)
plt.imshow(y_pred[0, :, :, 0], cmap=plt.get_cmap('gray'), vmin=-np.max(y_pred[0, :, :, 0]), vmax=np.max(y_pred[0, :, :, 0]))
plt.show()
#plt.imshow(y_pred[0, :, :, 1], cmap=plt.get_cmap('gray'))
#plt.show()
#plt.imshow(y_pred[0, :, :, 2], cmap=plt.get_cmap('gray'))
#plt.show()
#plt.imshow(y_pred[0, :, :, 31], cmap=plt.get_cmap('gray'))
#plt.show()


def integer_write_array_4da(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w', newline='') as file:
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    #for i4 in range(p_input_shape[3]):
                        csv.writer(file, delimiter=' ').writerow(p_integer_vector_4d[i1, i2, i3, :])


def integer_write_array_4db(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w', newline='') as file:
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                #for i3 in range(p_input_shape[2]):
                    for i4 in range(p_input_shape[3]):
                        csv.writer(file, delimiter=' ').writerow(p_integer_vector_4d[i1, i2, :, i4])


#x_example_integer = np.rint(x_example*255)
#integer_write_array_4da("input", x_example_integer, x_example_integer.shape)
#weights_integer = np.rint(weights*255)
#integer_write_array_4db("weights", weights_integer, weights_integer.shape)
