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
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


model.save_weights('saved_weights/my_weights')



def integer_write_array_4d(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w', newline='') as file:
        for i1 in p_input_shape[0]:
            for i2 in p_input_shape[1]:
                for i3 in p_input_shape[2]:
                    for i4 in p_input_shape[3]:
                        csv.writer(file, delimiter=' ').writerow(p_integer_vector_4d(i1, i2, i3, i4))


#integer_write_array_4d("input", input_n, input_shape)
