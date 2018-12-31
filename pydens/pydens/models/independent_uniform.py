import pandas as pd
import pdb
from scipy import stats

from . import base

class IndependentUniform(base.AbstractDensityModel):
    def __init__(self):
        pass

    def train(self, df):
        assert isinstance(df, pd.DataFrame)
        bounds = pd.concat([df.min(), df.max()], axis=1)
        loc = bounds[0]
        scale = (bounds[1] - bounds[0])
        self.columns = df.columns
        self.univariates = {v: stats.uniform(loc[k], scale[k]) for k, v in enumerate(self.columns)}

    def density(self, df, log=False):
        assert isinstance(df, pd.DataFrame)
        assert all(df.columns==self.columns)
        df_univariate = pd.DataFrame({
            v: self.univariates[v].pdf(df[v].values)
            for v in self.columns
        })
        return df_univariate.prod(axis=1)

    def rvs(self, n):
        ''' Generate n samples from the fitted distribution '''
        if not hasattr(self, 'univariates'):
            raise Exception("Call `train` before you call `rvs`")
        samples = {v: self.univariates[v].rvs(n) for v in self.columns}
        return pd.DataFrame(samples)[self.columns]