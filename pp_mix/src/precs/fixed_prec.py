import numpy as np
from .base_prec import BaseUnivPrec, BaseMultiPrec
from .precmat import PrecMat

class FixedUnivPrec(BaseUnivPrec):
    def __init__(self, sigma):
        self.sigma = sigma

    def sample_prior(self):
        return self.sigma

    def sample_given_data(self, data, curr, mean):
        return self.sigma

    def mean(self):
        return self.sigma

    def lpdf(self, val):
        return 0.0

class FixedPrec(BaseMultiPrec):
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma

    def sample_prior(self):
        out = self.sigma * np.identity(self.dim)
        return PrecMat(out)

    def sample_given_data(self, data, curr, mean):
        return self.sample_prior()

    def mean(self):
        out = self.sigma * np.identity(self.dim)
        return PrecMat(out)

    def lpdf(self, val):
        return 0.0

