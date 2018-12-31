import numpy as np
import pandas as pd

class CadeData(object):
    ''' A standardized data format for pydens.cade.Cade '''
    def __init__(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert len(y.shape)==1
        assert len(y)==X.shape[0]
        self.X = X
        self.y = y
