import numpy as np
from scipy import signal

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [[1, 2], [3, 4]]
print(b)
b = np.flip(b)
print(b)
c = signal.convolve2d(a, b, mode='valid')
print(c)
