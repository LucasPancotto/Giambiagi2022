"""Plot results"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.semilogy(out[0], out[1], label='data')
plt.semilogy(out[0], out[2], label='phys')
# bal = np.loadtxt('balance.dat', unpack=True)
# plt.semilogy(bal[0], bal[1], label='alpha')
plt.legend(fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)

true = np.load('../chaotic.npy')
tt   = true[:, 0]
true = true[:, 1:4]
pred = np.load('pred.npy')
lett = ['x', 'y', 'z']
for ii in range(3):
    plt.figure(2+ii)
    plt.plot(tt, true[:, ii], label='truth')
    plt.plot(tt, pred[:, ii], '--', label='prediction')
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel(f'${lett[ii]}$', fontsize=14)
    plt.legend(fontsize=14)

plt.figure(10)
inv = np.loadtxt('inverse.dat', unpack=True)
plt.plot(inv[0], 10*inv[1], label=r'$10\sigma$')
plt.axhline(10, ls='--', color='C0')
plt.plot(inv[0], 10*inv[2], label=r'$10\rho$')
plt.axhline(28, ls='--', color='C1')
plt.plot(inv[0], inv[3], label=r'$\beta$')
plt.axhline(8/3, ls='--', color='C2')
plt.legend(fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Coefficients', fontsize=14)

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
