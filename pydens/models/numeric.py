import pdb
from scipy import stats

from . import base
from .gaussians import GaussianMixture
from .poisson import Poisson
from .piecewise_niform import PiecewiseUniform

class NumericMixture(base.AbstractDensity):
    ''' Model a single numeric feature

    It's hard to approximate an *arbitrary* density parametrically. We'll
    do our best by fitting several different types of densities and then
    using something like Bayes factors to pick a sparse weighted average
    of these distributions
    '''
    model_classes = [Poisson, GaussianMixture]

    def train(self, series):
        '''
        :param series: pandas series of numeric values

        '''
        # # Handy for a uniform model
        # bounds = pd.concat([df.min(), df.max()], axis=1)
        # loc = bounds[0]
        # scale = (bounds[1] - bounds[0])

        # Each candidate model falls back to a trivial density model if the supplied data
        # is not compatible with the assumptions of the model (for example, non-integer values for
        # a poisson)
        self.models = [model() for model in self.model_classes]
        for m in self.models:
            m.train(series)

        loglikelihoods = [m.density_series(series) for m in self.models]
        #
        # poisson = Poisson()
        # poisson.train(series)
        #
        # gm = GaussianMixture()
        # gm.train(series)

        # Compute the log likelihood of each model on its training data
        pdb.set_trace()

    def density(self, x):
        print("not yet implemented; see self.density_series()")
        pdb.set_trace()

    def density_series(self, x):
        pdb.set_trace()

    def rvs(self, n):
        pdb.set_trace()