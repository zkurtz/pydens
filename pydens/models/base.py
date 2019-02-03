from abc import ABC, abstractmethod

class AbstractDensity(ABC):
    def __init__(self, verbose=False):
        self.verbose=verbose
        # Aliases for density:
        self.pdf = self.density
        self.predict = self.density

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