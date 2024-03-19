import numpy as np
from scipy.special import erf


def Kernel_Integration(dt, para):
 
    dt = dt.flatten()

    print(len(dt))
    
    distance = np.tile(dt[:, np.newaxis], (1, len(para['landmark']))) - np.tile(para['landmark'], (len(dt), 1))
    landmark = np.tile(para['landmark'], (len(dt), 1))

 
    if para['kernel'] == 'exp':
        G = 1 - np.exp(-para['w'] * (distance - landmark))
        G[G < 0] = 0

    elif para['kernel'] == 'gauss':
        G = 0.5 * (erf(distance / (np.sqrt(2) * para['w'])) + erf(landmark / (np.sqrt(2) * para['w'])))

    else:
        print('Error: please assign a kernel function!')
        G = None

    return G
