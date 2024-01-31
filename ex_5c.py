# Verwenden des Conv2D-Layers und exportieren von Daten in txt

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import keras
from keras import layers
from scipy import signal

# Importieren und Vorbereiten des Datensatzes
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
num_classes = 10
input_shape = (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Nachbauen des Modells, jedoch nur mit Convolution-Layer
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), use_bias=False),
    ]
)

# Laden der zuvor bestimmten Gewichte
model.load_weights('saved_weights/my_weights')
model.summary()
# Gewichte speichern (model.get_weights()[1] wären die Bias-Werte,
# welche hier nicht trainiert wurden und nicht für VHDL benutzt werden)
weights = np.array(model.get_weights()[0])
print(weights.shape)

# Beispielbild aus dem Testdatensatz
x_example = x_test[0, :, :, 0]
x_example = np.reshape(x_example, (1, 28, 28, 1))
print(x_example.shape)
# Beispielbild in das Netz hineingeben
y_pred = model.predict_step(x_example)
# Beispielhaftes Abspeichern der Ergebnisse aus dem ersten und letzten Channel
y_pred_1 = y_pred[0, :, :, 0]
y_pred_32 = y_pred[0, :, :, 31]
# Durchführen der ersten Faltung mit Python-Funktion (signal)
# Funktion spiegelt das eine Array. Das passt nicht zum Algorithmus aus VHDL oder Tensorflow,
# daher vorher schon einmal spiegeln
weights_flipped = np.flip(weights[:, :, 0, 0])
y_conv = signal.convolve2d(x_test[0, :, :, 0], weights_flipped, mode='valid')
# Vergleichen der beiden Ergebnisse, wegen Rundungsunterschieden mit .allclose
print(np.allclose(np.array(y_pred_1), y_conv))

# Plotten des Beispielbildes und der Ergebnisse
fig, axs = plt.subplots(1, 4, figsize=(9, 3))
axs[0].imshow(x_test[0, :, :, 0], cmap=plt.get_cmap('gray'))
axs[0].set_title('Input')
axs[1].imshow(y_pred_1, cmap=plt.get_cmap('gray'))
axs[1].set_title('Netz, Channel=0')
axs[2].imshow(y_conv, cmap=plt.get_cmap('gray'))
axs[2].set_title('Funktion, Channel=0')
axs[3].imshow(y_pred_32, cmap=plt.get_cmap('gray'))
axs[3].set_title('Netz, Channel=31')
plt.savefig('convolution_float.pdf')
plt.show()


# Exportieren in txt-Datei
# Funktion zum Schreiben eines 4D-Array in eine txt
def integer_write_array_4d(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w') as file:
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    for i4 in range(p_input_shape[3]):
                        file.write("".join(str(p_integer_vector_4d[i1, i2, i3, i4])) + "\n")


# Da in VHDL mit Integern gearbeitet wird, werden alle Daten mit 255 multipliziert
# Beispielbild exportieren
x_example_integer = np.rint(x_example*255)
integer_write_array_4d("input", x_example_integer, x_example_integer.shape)
# Alle Gewichte exportieren
weights_integer = np.rint(weights*255)
integer_write_array_4d("weights", weights_integer, weights_integer.shape)
