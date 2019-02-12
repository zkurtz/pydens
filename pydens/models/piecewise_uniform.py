import pandas as pd
import pdb
from scipy import stats


from . import base

class Shmistogram(object):
    def __init__(self, x, max_bins=10):
        if not isinstance(x, pd.Series):
            assert isinstance(x, np.ndarray) or isinstance(x, list)
            x = pd.Series(x)
        self.x = x
        self.max_bins = max_bins
        self.sure_loner_share = 2 / float(max_bins)
        self.shmistogram = self.shmistify()

    # def group_on_loners(self, x):
    #     '''
    #     x could look like F,F,T,T,F, for example; then the output is
    #     0,0,1,1,1. Basically all become 1 except for every F that's not preceded by T
    #     :param x: boolean numpy array indicating loner (true) at each index
    #     '''
    #     llist = [None]*len(x)
    #     llist[0]=0
    #     for k in range(1, len(x)):
    #         if x[k]:
    #             llist[k] = llist[k-1] + 1
    #         elif x[k-1]:
    #             llist[k] = llist[k-1] + 1
    #         else:
    #             llist[k] = llist[k-1]
    #     return llist

    def collapse(self, df):
        pdb.set_trace()

    def shmistify(self):
        tbl = self.x.value_counts()
        df = pd.DataFrame({
            'value': tbl.index.values,
            'freq': tbl.values
            }).sort_values('value').reset_index(drop=True)
        df['share'] = df.freq / len(self.x)
        df['is_loner'] = df.share > self.sure_loner_share
        groups = {key: group.drop('is_loner', axis=1).reset_index(drop=True)
            for key, group in df.groupby('is_loner')}

        pdb.set_trace()

        #df['group'] = self.group_on_loners(df.share > self.sure_loner_share)
        # vdiff = df.value.diff()
        # midpoints = vdiff[1:].values/2
        # df['bin_min'] = [df.value[0]] + (midpoints + df.value[:-1]).tolist()
        # df['bin_max'] = (df.value[1:] - midpoints).tolist() + [df.value.values[-1]]
        # #r = df.value.max() - df.value.min()
        #
        #
        # pdb.set_trace()
        # sdf = pd.concat([
        #     self.collapse(df[df.group==k]) for k in range(df.group.max()+1)
        # ])
        # pdb.set_trace()

    def plot(self):
        pass

class PiecewiseUniform(base.AbstractDensity):
    ''' Adaptive-width histogram density estimator
    '''
    def __init__(self, alpha=None):
        self.alpha = alpha

    def train(self, series):
        '''
        :param series: pandas series of numeric values

        '''
        # # Handy for a uniform model
        # bounds = pd.concat([df.min(), df.max()], axis=1)
        # loc = bounds[0]
        # scale = (bounds[1] - bounds[0])
        # stats.uniform(loc[k], scale[k])
        shmistogram = Shmistogram(series)
        pdb.set_trace()

    def density(self, x):
        print("not yet implemented; see self.density_series()")
        pdb.set_trace()

    def density_series(self, x):
        pdb.set_trace()

    def rvs(self, n):
        pdb.set_trace()