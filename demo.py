import numpy as np
import pandas as pd

from pydens.cade import Cade
from pydens.models import JointDensity
from pydens.classifiers.lightgbm import Lgbm
from pydens import simulators
from pydens import wrappers

# Define a problem by simulating some data from a known distribution
np.random.seed(0)
sz = simulators.bivariate.Zena()
df = sz.rvs(1000)

# Use Cade to estimate the density of the data. Cade works by
# first fitting an initial naive joint density model and subsequently
# improving the initial density estimates with a classifier that
# tries to distinguish between the real data versus fake data sampled
# from the initial density model
cade = Cade(
    initial_density=JointDensity(verbose=1),
    classifier=Lgbm(verbose=1),
    verbose=True
)
cade.train(df, diagnostics=True)

# Apply fastKDE (pip install fastkde)
fkde = wrappers.FastKDE()
fkde.train(df)

# Validate the estimators on new data
new_df = sz.rvs(1000)
densities = pd.DataFrame({
    'CADE': cade.density(new_df),
    'fastKDE': fkde.density(new_df),
    'true_gen': sz.density(new_df) # <- The true generative density:
})

class Evaluation(object):
    metrics = ['correlation_with_truth', 'mean_likelihood']
    def __init__(self, estimators, truth=None):
        for est in estimators:
            assert isinstance(estimators[est], pd.Series)
        self.estimators = estimators
        self.truth = truth
        if truth is not None:
            assert isinstance(truth, pd.Series)

    def correlation_with_truth(self, pred):
        ''' Maximize me '''
        return pred.corr(self.truth, method='spearman')

    def mean_likelihood(self, pred):
        ''' Maximize me '''
        return pred.mean()

    def evaluate_estimator(self, name):
        est = self.estimators[name]
        return [getattr(self, m)(est) for m in self.metrics]

    def evaluate(self):
        estimators = self.estimators.keys()
        return pd.DataFrame({
            e: self.evaluate_estimator(e) for e in estimators
        }, index=self.metrics)


ev = Evaluation(
    estimators={e: densities[e] for e in ['CADE', 'fastKDE']},
    truth=densities['true_gen']
)
pd.set_option('display.precision', 3)
print(ev.evaluate())
