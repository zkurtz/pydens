import numpy as np
import pandas as pd

from . import base

class SeriesTable(object):
    def __init__(self, series, compute_empirical_p=False):
        self.name = series.name
        self.df = pd.DataFrame(series.value_counts(sort=False))
        self.df.rename(columns={series.name: 'n_obs'}, inplace=True)
        if compute_empirical_p:
            self.df['empirical_p'] = self.df.n_obs/len(series)

class Multinomial(base.AbstractDensity):
    ''' Model a single categorical feature '''

    def train(self, series):
        '''
        :param series: pandas series of integers
        '''
        st = SeriesTable(series)
        self.name = st.name
        self.df = st.df
        reg_counts = self.df.n_obs.values + 1
        self.df['density'] = reg_counts/reg_counts.sum()
        # Assign a tiny prob for never-before observed values of this multinomial:
        self.out_of_sample_dens = 1/(2*self.df.shape[0])

    def density(self, x):
        ''' Compute the density for an individual value '''
        try:
            return self.df.density[x]
        except:
            # assert x not in self.df.index.values
            return self.out_of_sample_dens

    def density_series(self, x):
        '''
        Fast density computation for a list of values

        :param x: (pandas.Series) Values at which to compute the density
        '''
        assert isinstance(x, pd.Series)
        df = pd.DataFrame({
            'levels': x,
            'idx': range(len(x))
        }).merge(
            pd.DataFrame({
                'levels': self.df.index.values,
                'density': self.df.density.values
            }), on='levels', how='left'
        ).sort_values('idx')
        return df.density.fillna(self.out_of_sample_dens).values

    def rvs(self, n=1):
        return np.random.choice(
            a=self.df.index.values,
            size=n,
            p=self.df.density.values,
            replace=True)