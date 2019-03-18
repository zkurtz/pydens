import numpy as np
import pandas as pd
import pdb

class Evaluation(object):
    metrics = ['correlation_with_truth', 'mean_likelihood']
    def __init__(self, estimators, truth=None):
        for est in estimators:
            assert isinstance(estimators[est], np.ndarray)
        self.estimators = estimators
        self.truth = truth
        if truth is not None:
            assert isinstance(truth, np.ndarray)

    def correlation_with_truth(self, pred):
        ''' Maximize me '''
        if self.truth is None:
            raise Exception("correlation_with_truth can't be computed since you did not provide anything for `truth`")
        s_pred = pd.Series(pred)
        t_pred = pd.Series(self.truth)
        return s_pred.corr(t_pred, method='spearman')

    def mean_likelihood(self, pred):
        ''' Maximize me, subject to the assumption that the predictions are consistent
         with a probability function (i.e. integrating to 1) '''
        return np.mean(pred)

    def evaluate_estimator(self, name):
        est = self.estimators[name]
        return [getattr(self, m)(est) for m in self.metrics]

    def evaluate(self):
        estimators = self.estimators.keys()
        return pd.DataFrame({
            e: self.evaluate_estimator(e) for e in estimators
        }, index=self.metrics)
