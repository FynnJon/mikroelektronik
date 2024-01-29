# Einlesen von txt
# HIER GUTE

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal


def integer_read_array_4d(fname, p_input_shape):
    p_integer_array_4d = np.zeros(p_input_shape)
    with open(fname + '.txt', 'r') as file:
        reader = list(csv.reader(file, delimiter=' '))
        ix = 0
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    #for i4 in range(p_input_shape[3]):
                        p_integer_array_4d[i1, i2, i3, :] = reader[ix]
                        ix = ix + 1
    return p_integer_array_4d


def integer_read_array_4db(fname, p_input_shape):
    p_integer_array_4d = np.zeros(p_input_shape)
    with open(fname + '.txt', 'r') as file:
        reader = list(csv.reader(file, delimiter=' '))
        ix = 0
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    #for i4 in range(p_input_shape[3]):
                        p_integer_array_4d[i1, i2, i3, :] = reader[ix]
                        ix = ix + 1
    return p_integer_array_4d


w = integer_read_array_4d("weights", (5, 5, 1, 1))
i = integer_read_array_4d("input", (1, 28, 28, 1))
w = w[:, :, 0, 0]
i = i[0, :, :, 0]
w = np.flip(w)
c = signal.convolve2d(i, w, mode='valid')
#plt.imshow(c, cmap=plt.get_cmap('gray'))
#plt.show()
#print(c)


a = integer_read_array_4db("output", (1, 24, 24, 2))
plt.imshow(a[0, :, :, 1], cmap=plt.get_cmap('gray'))
plt.show()
#print(a)

