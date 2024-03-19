import numpy as np
import random
from .precmat import PrecMat
from .base_prec import BaseMultiPrec
from scipy.stats import expon


class Expon(BaseMultiPrec):
    def __init__(self, scale, C, D):
        self.scale = scale  # scale = 1/E(x)
        self.C = C
        self.D = D
    def sample_prior(self):
        print('sample',self.scale)
        out = np.random.exponential(1/self.scale, size=(self.C, self.D, self.C))
        print(out)
        return out
    
    def sample_given_data(self, data):
        pass
# D M=kernel K D 
    def mean(self):
        out = np.full(shape=(self.C, self.D, self.C), fill_value = 1/self.scale)
        return out
    
    def lpdf(self, val):

        pdf_values = np.empty_like(val, dtype=float)

        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                for k in range(val.shape[2]):
                    pdf_values[i, j, k] = expon.pdf(val[i, j, k], scale=self.scale)  
        matrix_pdf = np.prod(pdf_values)

        return matrix_pdf