import math
import numpy as np
import pdb
from scipy import stats
from sklearn import mixture

from . import base

# TODO: CREATE HEURISTICS TO EFFICIENTLY DO MODEL SELECTION (POISSON VS GAUSSIAN, ETC) AND THEN FIT THE SELECTED MODEL

class GaussianMixture(base.AbstractDensity):
    ''' Model a single continuous feature '''
    def __init__(self, n_gaussians=None):
        self.n_gaussians = n_gaussians

    def train(self, series):
        '''
        :param series: pandas series of integers

        TODO - deal with terribly slow fitted. Sub-sample the data?
        '''
        # # Handy for a uniform model
        # bounds = pd.concat([df.min(), df.max()], axis=1)
        # loc = bounds[0]
        # scale = (bounds[1] - bounds[0])

        if self.n_gaussians is None:
            n_unique = series.nunique()
            n = math.ceil(np.log(n_unique)/2)
            self.n_gaussians = n

        gmm = mixture.GaussianMixture(n_components=self.n_gaussians)
        try:
            gmm.fit(series.values.reshape(-1, 1))
        except:
            pdb.set_trace()
        mu = gmm.means_
        var = gmm.covariances_
        self.gaussians = [stats.norm(mu[k], var[k]) for k in range(self.n_gaussians)]
        self.weights = gmm.weights_

    def density(self):
        pdb.set_trace()

    def density_series(self, x):
        pdb.set_trace()

    def rvs(self, n):
        pdb.set_trace()