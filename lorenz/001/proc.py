"""Plot results"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.plot(out[0], out[1])
plt.plot(out[0], out[2])

plt.figure(2)
true = np.load('../chaotic.npy')
tt   = true[:, 0]
true = true[:, 1:4]
pred = np.load('pred.npy')
for ii in range(3):
    plt.figure(2+ii)
    plt.plot(tt, true[:, ii])
    plt.plot(tt, pred[:, ii])
plt.show()
