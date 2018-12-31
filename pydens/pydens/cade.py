import numpy as np
import pandas as pd
import pdb

from .data import CadeData
from .classifiers.lightgbm import Lgbm
from .classifiers.base import AbstractLearner

class Cade(object):
    ''' Classifier-adjusted density estimation

    Based on https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf

    :param sim_size:
    '''

    modal_simulation_size = 10000

    def __init__(self,
            initial_density=None,
            classifier=Lgbm(),
            sim_size='auto'):
        self.initial_density = initial_density
        assert isinstance(classifier, AbstractLearner)
        self.classifier = classifier
        self.sim_size = sim_size

    def compute_simulation_size(self, df):
        ''' Determine the number of synthetic data samples to simulate

        If self.sim_size is 'auto', sets the simulation size as the geometric mean
        between the

        If self.sim_size is a positive number less than 100, simulation size is
        round(self.sim_size)*df.shape[0]

        Finally, if self.sim_size >= 100, simulation size is round(self.sim_size)
        '''
        n_real = df.shape[0]
        if isinstance(self.sim_size, str):
            assert self.sim_size=='auto'
            sim_n = np.sqrt(n_real*self.modal_simulation_size)
        elif self.sim_size < 100:
            assert self.sim_size > 0
            sim_n = round(self.sim_size*n_real)
            if sim_n < 10:
                raise Exception("Simulation size is very small. Consider using a larger value of sim_size")
        else:
            sim_n = round(self.sim_size)
        self.sim_rate = sim_n / df.shape[0]
        return int(sim_n)

    def train(self, df):
        ''' Model the density of the data

        :param df: (pandas DataFrame)
        '''
        assert isinstance(df, pd.DataFrame)
        # Train a generative density model
        self.initial_density.train(df)
        # Simulate fake data from the model and join it with the real data
        sim_n = self.compute_simulation_size(df)
        partially_synthetic_data = CadeData(
            X=pd.concat([df, self.initial_density.rvs(sim_n)]),
            y=np.concatenate([np.ones(df.shape[0]), np.zeros(sim_n)])
        )
        # Train a classifier to distinguish real from fake
        self.classifier.train(partially_synthetic_data)

    def density(self, X):
        ''' Predict the density at new points
        '''
        assert isinstance(X, pd.DataFrame)
        # Initial density estimate
        synthetic_dens = self.initial_density.density(X)
        # Classifier adjustment factor
        p_real = self.classifier.predict(X)
        odds_real = p_real/(1 - p_real)
        classifier_adjustment=self.sim_rate*odds_real
        # All together as equation 2.1 in
        #   https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf
        return synthetic_dens*classifier_adjustment
