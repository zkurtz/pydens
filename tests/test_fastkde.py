from scipy import stats

def test_fastkde_runs():
    from fastkde import fastKDE
    # TODO: reactivate this after
    #   https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for
    # gauss = stats.norm(-2, 4)
    # data = gauss.rvs(size=100)
    # _ = fastKDE.pdf(data)
