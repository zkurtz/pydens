from abc import ABC, abstractmethod

class AbstractDensityModel(ABC):
    def __init__(self, verbose=False):
        self.verbose=verbose

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def density(self, X):
        pass

    @abstractmethod
    def rvs(self, n):
        pass

    def vp(self, string):
        if self.verbose:
            print(string)