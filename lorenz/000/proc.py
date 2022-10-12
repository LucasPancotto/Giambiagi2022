"""Plot results"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.plot(out[0], out[1])
plt.plot(out[0], out[2])

plt.figure(2)
true = np.load('../chaotic.npy')
pred = np.load('pred.npy')
plt.plot(true[:, 0], true[:, 1])
plt.plot(true[:, 0], pred[:, 1])

plt.show()
