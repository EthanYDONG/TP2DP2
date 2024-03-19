import numpy as np

class PrecMat:
    def __init__(self, prec):
        self.prec = np.array(prec)
        self.var = None

        self.univariate_val = None
        self.is_univariate = False
        self.compute_var = False

    def set_compute_var(self):
        self.compute_var = True

    def get_prec(self):
        return self.prec

    def get_var(self):
        if not self.compute_var:
            raise RuntimeError("Variance has not been computed!")
        return self.var

    def get_cho_factor(self):
        return self.cho_factor

    def get_cho_factor_eval(self):
        return self.cho_factor_eval

    def get_log_det(self):
        return self.log_det

    def __str__(self):
        return str(self.prec)

