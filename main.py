import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


gbw = 100e3
w0 = 1000
w = np.linspace(1, 100e4, 10000000)
sys = signal.TransferFunction([2*np.pi*gbw], [(1+np.pi*gbw/w0), 0])
w, mag, phase = signal.bode(sys, w)

plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.show()
