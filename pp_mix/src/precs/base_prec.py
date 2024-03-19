from abc import ABC, abstractmethod
import numpy as np

class BasePrec(ABC):
    @abstractmethod
    def __init__(self):
        pass

class BaseUnivPrec(BasePrec):
    @abstractmethod
    def sample_prior(self):
        pass
    
    @abstractmethod
    def sample_given_data(self, data, curr, mean):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def lpdf(self, val):
        pass

class BaseMultiPrec(BasePrec):
    @abstractmethod
    def sample_prior(self):
        pass
    
    @abstractmethod
    def sample_given_data(self, data, curr, mean):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def lpdf(self, val):
        pass
