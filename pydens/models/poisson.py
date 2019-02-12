import pdb
from scipy import stats

from . import base

class Poisson(base.AbstractDensity):
    '''
    This is a generalization of the Poisson distribution. It defines the density
    at non-integer values by first rounding the input
    '''
    def train(self, series):
        self.loc = round(series.min())
        self.mu = series.mean() - self.loc
        self.poisson = stats.poisson(mu=self.mu, loc=self.loc)
        self.negative_point_dens = 0.5*min(
            self.poisson.pmf(self.loc),
            self.poisson.pmf(round(series.max()))
        )

    def density(self, x):
        x = round(x)
        if x < self.loc:
            return self.negative_point_dens
        return self.poisson.pmf(x)

    def density_series(self, x):
        return [self.density(v) for v in x.values]

    def rvs(self, n):
        pdb.set_trace()
        return self.poisson.rvs(n)