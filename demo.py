import numpy as np
import pandas as pd
import pdb

from pydens.cade import Cade
from pydens.models import IndependentUniform
from pydens.classifiers.lightgbm import Lgbm
from pydens import simulators

# Define a problem by simulating some data from a known distribution
np.random.seed(0)
sz = simulators.bivariate.Zena()
data = sz.rvs(1000)

# Apply Cade to estimate the density of the data. Cade works by
# first fitting an initial, naive model such as independent uniform
# distributions for each features, and subsequently improving the initial
# density estimates with a classifier that distinguishes between
# the real data and samples from the initial density
cade = Cade(
    initial_density=IndependentUniform(),
    classifier=Lgbm()
)
diag = cade.train(data, diagnostics=True)
vdf = diag['val_df'][diag['val_df'].truth==1]
vdf.pred.describe()
print("The classifier achieved an AUROC score of " + str(round(diag['auc'], 3)))

# Apply fastKDE (pip install fastkde)
from fastkde import fastKDE
pdb.set_trace()
myPDF, axes = fastKDE.pdf(data.iloc[:,0].values, data.iloc[:,1].values)

# Check the estimates against the generative density
val_df = pd.DataFrame({
    'cade_est': cade.density(data),
    'fast_kde_est': myPDF(data),
    # Since this is a simulation, we get to observe the generative density:
    'gen': sz.pdf(data.values)
})
cor = round(val_df.est.corr(val_df.gen, method='spearman'), 3)
print("Spearman correlation, estimated vs generative density: " + str(cor))

# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# import numpy as np
# pd.Series(stats.truncnorm.rvs(-2, 4, size=1000)).hist()
# plt.show()