import numpy as np
from scipy import signal
import csv

#input_shape = (10, 10)
#input_n = np.random.randint(0, 255, input_shape)s
#print(input_n)
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1, 2], [3, 4]])
c = signal.convolve2d(a, b, 'valid')
print(a)
print(b)
print(c)



def integer_write_array_4d(fname, p_integer_vector_4d, p_input_shape):
    with open(fname + '.txt', 'w', newline='') as file:
        for i1 in range(p_input_shape[0]):
            for i2 in range(p_input_shape[1]):
                #for i3 in p_input_shape[2]:
                    #for i4 in p_input_shape[3]:
                        csv.writer(file, delimiter=' ').writerow(p_integer_vector_4d[i1, :])


#integer_write_array_4d("input", input_n, input_shape)
