import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


w0 = 100
Q=2
w = np.linspace(1e-3, 100e4, 10000000)

sys = signal.TransferFunction([-1, 0, -w0**2*Q], [1, w0/Q, w0**2])
w, mag, phase = signal.bode(sys, w)

fig, axs = plt.subplots(figsize=(8, 5))
axs.semilogx(w, mag, color='grey', label='TP 1. Ordnung')
ax = axs.twinx()
ax.semilogx(w, phase, color='grey', linestyle='dashed')

axs.grid()
axs.set_xlabel(r'$s$')
axs.set_ylabel('Betrag in dB')
ax.set_ylabel('Phase in °')
#axs.legend(loc='lower left')

plt.show()
