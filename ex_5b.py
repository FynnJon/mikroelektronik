# Einlesen von txt

import numpy as np
import matplotlib.pyplot as plt
import csv


def integer_read_array_4d(fname, p_input_shape):
    p_integer_array_4d = np.zeros(p_input_shape)
    with open(fname + '.txt', 'r') as file:
        reader = list(csv.reader(file, delimiter=' '))
        print(reader)
        i = 0
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                for i3 in range(p_input_shape[2]):
                    #for i4 in range(p_input_shape[3]):
                        p_integer_array_4d[i1, i2, i3, :] = reader[i]
                        i = i + 1
    return p_integer_array_4d


a = integer_read_array_4d("output", (1, 24, 24, 1))
plt.imshow(a[0, :, :, 0], cmap=plt.get_cmap('gray'), vmin=-np.max(a[0, :, :, 0]), vmax=np.max(a[0, :, :, 0]))
plt.show()

