class EigenMatrix:
    def __init__(self, data):
        self.rows = data.shape[0]
        self.cols = data.shape[0]
        self.data = data

class EigenVector:
    def __init__(self, data):
        self.size = len(data)
        self.data = data

    def extend_EigenVector(self, value):
        self.data.extend(value)
        self.size += len(value)

class PPState:
    def __init__(self):
        pass

class StraussState(PPState):

    def __init__(self):
        self.beta = 0
        self.gamma = 0
        self.R = 0
        self.birth_prob = []
        self.birth_arate = []

class NrepState(PPState):
    def __init__(self, u, p, tau):
        self.u = u
        self.p = p
        self.tau = tau

class MultivariateMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 1
        self.mtot = 3
        self.a_means = []
        self.na_means = []

        self.a_precs = []
        self.na_precs = []
        self.a_jumps = []
        self.na_jumps = []
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()


class UnivariateMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 2
        self.mtot = 3

        self.a_means = []
        self.na_means = 0
        self.a_precs = 0
        self.na_precs = 0
        self.a_jumps = 0
        self.na_jumps = 0
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()



class BernoulliMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 2
        self.mtot = 3

        self.a_probs = []
        self.na_probs = []

        self.a_jumps = EigenVector()
        self.na_jumps = EigenVector()
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()



