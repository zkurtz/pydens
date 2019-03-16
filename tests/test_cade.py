import pytest

@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
def test_cade_default():
    from pydens.cade import Cade
    x = Cade()
    assert x is not None

def test_cade():
    import numpy as np
    import pandas as pd

    from pydens.cade import Cade
    from pydens.models import JointDensity
    from pydens.classifiers.lightgbm import Lgbm
    from pydens import simulators

    # Compile a Cade class with all defaults
    np.random.seed(0)
    N = 100
    sz = simulators.bivariate.Zena()
    data = sz.rvs(100)
    cade = Cade(
        initial_density=JointDensity(),
        classifier=Lgbm(),
        sim_size = N
    )
    cade.train(data, diagnostics=True)
    diagnostics = cade.diagnostics

    assert all([x in diagnostics.keys() for x in ['val_df', 'auc']])
    df = diagnostics['val_df']
    assert df.shape[0] == 2*N
    auc = diagnostics['auc']
    assert isinstance(auc, np.float)
    assert auc >= 0
    assert auc <= 1

    dens = cade.density(data.iloc[:3])
    assert len(dens) == 3
    assert isinstance(dens, pd.Series)
    assert dens.dtype == 'float64'
