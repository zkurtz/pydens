import numpy as np
import pandas as pd

class Evaluation(object):
    def __init__(self, estimators, truth=None):
        for est in estimators:
            assert isinstance(estimators[est], np.ndarray)
        self.estimators = estimators
        self.truth = truth
        if truth is not None:
            assert isinstance(truth, np.ndarray)
        self.metrics = {
            'rank-order correlation': self.rank_order_correlation,
            'pearson correlation': self.pearson_correlation,
            'mean density': self.mean_density
        }

    def rank_order_correlation(self, pred):
        ''' Spearman (i.e. rank-order) correlation of prediction against truth

        Maximize me
        '''
        if self.truth is None:
            raise Exception("rank-order correlation can't be computed since you did not provide anything for `truth`")
        s_pred = pd.Series(pred)
        t_pred = pd.Series(self.truth)
        return s_pred.corr(t_pred, method='spearman')

    def pearson_correlation(self, pred):
        ''' Spearman (i.e. rank-order) correlation of prediction against truth

        Maximize me
        '''
        if self.truth is None:
            raise Exception("pearson correlation can't be computed since you did not provide anything for `truth`")
        s_pred = pd.Series(pred)
        t_pred = pd.Series(self.truth)
        return s_pred.corr(t_pred)

    def mean_density(self, pred):
        ''' The mean density; this is exp(-deviance)

         Maximize me IF your estimator does not cheat by
         producing a density model that integrates to something greater than one
         '''
        return np.mean(pred)

    def evaluate_estimator(self, name):
        est = self.estimators[name]
        return [fun(est) for name, fun in self.metrics.items()]

    def evaluate(self):
        estimators = self.estimators.keys()
        return pd.DataFrame({
            e: self.evaluate_estimator(e) for e in estimators
        }, index=self.metrics)
