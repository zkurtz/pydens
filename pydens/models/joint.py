import numpy as np
import pandas as pd
import pdb

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

class JointDensity(base.AbstractDensity):
    def __init__(self):
        self.Categorical = Multinomial
        self.Continuous = Multinomial # TODO: eventually import a smoother model from .continuous

    def _fit_categorical(self, series):
        model = self.Categorical()
        model.train(series)
        return model

    def _fit_continuous(self, values):
        model = self.Continuous()
        model.train(values)
        return model

    def _fit_univarite(self, series):
        if series.name in self.categorical_features:
            return self._fit_categorical(series)
        else:
            return self._fit_continuous(series)

    def train(self, df, categorical_features=None):
        assert isinstance(df, pd.DataFrame)
        if categorical_features is None:
            self.categorical_features = []
        else:
            assert isinstance(categorical_features, list)
            self.categorical_features = categorical_features
        self.columns = df.columns
        self.univariates = {v: self._fit_univarite(df[v]) for v in self.columns}
        #self.univariates = {v: stats.uniform(loc[k], scale[k]) for k, v in enumerate(self.columns)}

    def density(self, df, log=False):
        assert isinstance(df, pd.DataFrame)
        assert all(df.columns==self.columns)
        df_log_univariate = pd.DataFrame({
            #v: np.log([self.univariates[v].pdf(x) for x in df[v].values])
            v: np.log(self.univariates[v].density_series(df[v]))
            for v in self.columns
        })
        log_dens = df_log_univariate.sum(axis=1)
        if log:
            return log_dens
        return np.exp(log_dens)

    def rvs(self, n):
        ''' Generate n samples from the fitted distribution '''
        if not hasattr(self, 'univariates'):
            raise Exception("Call `train` before you call `rvs`")
        samples = {v: self.univariates[v].rvs(n) for v in self.columns}
        return pd.DataFrame(samples)[self.columns]