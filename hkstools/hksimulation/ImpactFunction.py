import numpy as np
from .Kernel import Kernel



def ImpactFunction(u, dt, para):
    A = np.reshape(para['A'][u, :, :], (para['A'].shape[1], para['A'].shape[2]))
    basis = Kernel(dt, para)

 
    if len(para['landmark']) == 1:
        basis_extended = np.ones_like(A) * basis
   
        phi = np.multiply(A, basis_extended)
    else:
        basis_extended = basis
        phi = np.dot(A.T, basis.T)

    return phi