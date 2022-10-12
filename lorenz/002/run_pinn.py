# Lorenz PINN

import numpy as np
import tensorflow as tf
from   tensorflow import keras

from pinn import PhysicsInformedNN

# NN params
layers = [1]+[128]*8+[3]

# Load data
data = np.load('../chaotic.npy')

x_data = data[:, 0:1].astype(np.float32)
y_data = data[:, 1:4].astype(np.float32)

# Normalization layer
inorm    = [x_data.min(0), x_data.max(0)]
onorm    = [y_data.mean(0), y_data.std(0)]

# Batch size
mbsize   = int(np.shape(x_data)[0]/2)

# Piecewise constant learning rate
lr = 1e-5
# lr = 1e-6
# lr = 1e-7

PINN = PhysicsInformedNN(layers,
                         activation='siren',
                         optimizer=keras.optimizers.Adam(lr),
                         norm_in=inorm,
                         norm_out=onorm,
                         norm_out_type='z-score',
                         inverse=([{'type': 'const', 'value': 1.0},
                                   {'type': 'const', 'value': 1.0},
                                   {'type': 'const', 'value': 1.0}]),
                         restore=True)
PINN.optimizer.learning_rate.assign(lr)

# Validation
def valid_func(self, x_plot):
    def validation(ep):
        y_pred = self.model(x_plot)[0].numpy()
        np.save('pred', y_pred)
    return validation
PINN.validation = valid_func(PINN, x_data)

@tf.function
def lorenz_system(model, coords, params):
    '''Lorenz system, PINN version'''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        out  = model(coords)
        rp   = out[0]
        inv  = out[1:]
        x_p  = rp[:,0]
        y_p  = rp[:,1]
        z_p  = rp[:,2]

    # First derivatives
    x_t = tape.gradient(x_p, coords)[:,0]
    y_t = tape.gradient(y_p, coords)[:,0]
    z_t = tape.gradient(z_p, coords)[:,0]
    del tape

    # Coefficients are now new unknowns
    sigma = inv[0][:,0]
    rho   = inv[1][:,0]
    beta  = inv[2][:,0]

    # Equations to be enforced
    f1 = x_t - (sigma*(y_p - x_p))
    f2 = y_t - (x_p*(rho - z_p) - y_p)
    f3 = z_t - (x_p*y_p - beta*z_p)
        
    return [f1, f2, f3]

# Train
PINN.train(x_data, y_data,
           lorenz_system,
           epochs=100000,
           batch_size=mbsize,
           data_mask=[False, True, False],
           lambda_phys=1.0,
           alpha=0.1,
           valid_freq=10,
           print_freq=10,
           save_freq=10,
           )
