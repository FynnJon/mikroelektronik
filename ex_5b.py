# Einlesen und Überprüfen des Output aus VHDL

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal


# Funktion zum Lesen der txt
def integer_read_array_4d(fname, p_input_shape):
    p_integer_array_4d = np.zeros(p_input_shape)
    with open(fname + '.txt', 'r') as file:
        reader = list(csv.reader(file, delimiter=' '))
        ix = 0
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    for i4 in range(p_input_shape[3]):
                        p_reader = np.array(reader[ix])
                        p_integer_array_4d[i1, i2, i3, i4] = p_reader[0]
                        ix = ix + 1
    return p_integer_array_4d


# Einlesen der Gewichte und des Input-Bildes aus der txt
weights = integer_read_array_4d("weights", (5, 5, 1, 32))
inputs = integer_read_array_4d("input", (1, 28, 28, 1))

# Faltung auf Channel=0
# Dazu Umwandlung in 2D-Arrays
i = inputs[0, :, :, 0]
w = weights[:, :, 0, 0]
w = np.flip(w)
output_py1 = signal.convolve2d(i, w, mode='valid')
w = weights[:, :, 0, 31]
w = np.flip(w)
output_py32 = signal.convolve2d(i, w, mode='valid')

# Einlesen des Output aus VHDL
output = integer_read_array_4d("output", (1, 24, 24, 32))
output_vhdl1 = output[0, :, :, 0]
output_vhdl32 = output[0, :, :, 31]

# Vergleichen der Ergebnisse
print((output_py1 == output_vhdl1).all())
print((output_py32 == output_vhdl32).all())

# Plotten der Ergebnisse
fig, axs = plt.subplots(1, 4, figsize=(9, 3))
axs[0].imshow(output_vhdl1, cmap=plt.get_cmap('gray'))
axs[0].set_title('VHDL, Channel=0')
axs[1].imshow(output_py1, cmap=plt.get_cmap('gray'))
axs[1].set_title('Python, Channel=0')
axs[2].imshow(output_vhdl32, cmap=plt.get_cmap('gray'))
axs[2].set_title('VHDL, Channel=31')
axs[3].imshow(output_py32, cmap=plt.get_cmap('gray'))
axs[3].set_title('Python, Channel=31')
plt.savefig('convolution_integer.pdf')
plt.show()
