from scipy import stats

def test_fastkde_runs():
    from fastkde import fastKDE
    gauss = stats.norm(-2, 4)
    data = gauss.rvs(size=100)
    _ = fastKDE.pdf(data)
