import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


w0 = 2.414 #oder -0.414
w = np.linspace(1e-3, 100e4, 10000000)

#TP 1. Ordnung
tp1 = signal.TransferFunction([w0], [1, w0])
w, mag1, phase1 = signal.bode(tp1, w)
#TP 4. Ordnung
#Dazu wurden Nenner und Zähler ausmultipliziert
tp4 = signal.TransferFunction([33.9585], [1, 9.656, 34.9644, 56.2693, 33.9585])
#mit 0.414
#tp4 = signal.TransferFunction([0.029], [1, 1.656, 1.028, 0.284, 0.029])
w, mag4, phase4 = signal.bode(tp4, w)

fig, axs = plt.subplots(figsize=(8, 5))
axs.semilogx(w, mag1, color='grey', label='TP 1. Ordnung')
axs.semilogx(w, mag4, color='blue', label='Reihenschaltung 4x TP 1. Ordnung')
ax = axs.twinx()
ax.semilogx(w, phase1, color='grey', linestyle='dashed')
ax.semilogx(w, phase4, color='blue', linestyle='dashed')

axs.grid()
axs.set_title('Bodediagramm Tiefpass 1. Ordnung und vierfache Reihenschaltung')
axs.set_xlabel(r'$s$')
axs.set_ylabel('Betrag in dB')
ax.set_ylabel('Phase in °')
axs.legend(loc='lower left')

plt.savefig('bodediagramm251.pdf')
plt.show()
