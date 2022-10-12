"""Plot results"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.semilogy(out[0], out[1], label='data')
plt.semilogy(out[0], out[2], label='phys')
bal = np.loadtxt('balance.dat', unpack=True)
plt.semilogy(bal[0], bal[1], label='alpha')
plt.legend()

true = np.load('../chaotic.npy')
tt   = true[:, 0]
true = true[:, 1:4]
pred = np.load('pred.npy')
for ii in range(3):
    plt.figure(2+ii)
    plt.plot(tt, true[:, ii])
    plt.plot(tt, pred[:, ii])

plt.figure(10)
inv = np.loadtxt('inverse.dat', unpack=True)
plt.plot(inv[0], inv[1], label=r'$\sigma$')
# plt.axhline(10, ls='--')
plt.plot(inv[0], inv[2], label=r'$\rho$')
# plt.axhline(28, ls='--')
plt.plot(inv[0], inv[3], label=r'$\beta$')
# plt.axhline(8/3, ls='--')
plt.legend()

# Orders of magnitude
# plt.figure(10)
# plt.plot(np.diff(true[:, 0])/np.diff(tt)[0])
# plt.plot(10.0*(true[:, 1]-true[:, 0]))
#
# plt.figure(20)
# plt.plot(np.diff(true[:, 1])/np.diff(tt)[0])
# # plt.plot(10.0*(true[:, 1]-true[:, 0]))
#
# plt.figure(30)
# plt.plot(np.diff(true[:, 2])/np.diff(tt)[0])

plt.show()
