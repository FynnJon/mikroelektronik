import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
#x_train = np.reshape(x_train, (60000, 784))
#x_test = np.reshape(x_test, (10000, 784))
#x_test = x_test.batch()
#example = x_test[0]
#print(example.shape)
#plt.imshow(example, cmap=plt.get_cmap('gray'))
#plt.show()

# Random Bild erzeugen 64x64, 50 Batch, 3 Stück
#input_shape = (1, 10, 10, 1)
#input_n = np.random.randint(0, 255, input_shape)
#input_t = tf.constant(input_n)
#cnn_feature = tf.random.normal(input_shape)*255
#print(input_n)
#print(input_t)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(1, (5, 5), x_train.shape[1:])(x_train)
])

# Parameter für das Training festlegen
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics='accuracy')
# Modell trainieren mit 100 Durchläufen, kein Callback
history = model.fit(x_train, y_train, epochs=5)
# Modell testen und Bewertung nach Loss, Accuracy etc
model.evaluate(x_test, y_test, verbose=2)
# Übersicht über das Modell, Anzahl Parameter etc
model.summary()




# Conv Layer
#output = tf.keras.layers.Conv2D(1, (5, 5), input_shape=input_shape[1:])(input_t)
#print(output.shape)


def integer_write_array_4d(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w', newline='') as file:
        for i1 in p_input_shape[0]:
            for i2 in p_input_shape[1]:
                for i3 in p_input_shape[2]:
                    for i4 in p_input_shape[3]:
                        csv.writer(file, delimiter=' ').writerow(p_integer_vector_4d(i1, i2, i3, i4))


#integer_write_array_4d("input", input_n, input_shape)
