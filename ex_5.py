# Erstellen und Trainieren eines Neuronalen Netzes

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import keras
from keras import layers

# Importieren des Datensatzes und Vorbereiten der Daten
# Datensatz laden
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Wertebereich von 0-255 auf 0-1 normalisieren
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Ziffern 0-9 -> Zehn Klassen
num_classes = 10
input_shape = (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Entwerfen des Models
model = keras.Sequential(
    [
        # Input-Layer, um mehrdimensionales Array zu verarbeiten
        keras.Input(shape=input_shape),
        # Convolution mit 32 Channels und 5x5 Kernel. Es wird kein Bias trainiert,
        # da dieser in VHDL nicht implementiert wurde
        layers.Conv2D(32, kernel_size=(5, 5), use_bias=False),
        # Verringert die Dimensionen, da im nächsten Layer nur in einer Dimension gerechnet wird
        layers.Flatten(),
        # Normaler Fully-Connected-Layer mit zehn Ausgängen, um das Netz trainieren zu können
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
# Nur fünf Trainingsepochen, damit es schnell geht, Genauigkeit hier nicht von Wichtigkeit
epochs = 5
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Speichern der ermittelten Gewichte, um diese weiterzuverwenden
model.save_weights('saved_weights/my_weights')
