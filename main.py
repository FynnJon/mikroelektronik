import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


gbw = 100e3
w0 = 1000
w = np.linspace(1e-3, 100e4, 10000000)
sys = signal.TransferFunction([(-0.478)**4], [1, -1.9, 1.37, 0.44, 0.05])
w, mag, phase = signal.bode(sys, w)

plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.show()
