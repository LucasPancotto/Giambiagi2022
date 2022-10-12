"""Plot results"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.semilogy(out[0], out[1], label='data')
plt.semilogy(out[0], out[2], label='phys')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)

# plt.show()

plt.figure(2)
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
# plt.close('all')
plt.show()

pred = np.load('pred_02000.npy')
lett = ['x', 'y', 'z']
for ii in range(3):
    plt.figure(2+ii)
    plt.plot(tt, true[:, ii], label='truth')
    plt.plot(tt, pred[:, ii], '--', label='prediction')
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel(f'${lett[ii]}$', fontsize=14)
    plt.legend(fontsize=14)

plt.figure(3)
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(true[:, 0], true[:, 1], true[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


plt.show()
