import numpy as np
import pandas as pd
import pdb

from pydens.cade import Cade
from pydens.models import JointDensity
from pydens.classifiers.lightgbm import Lgbm
from pydens import simulators

# Define a problem by simulating some data from a known distribution
np.random.seed(0)
sz = simulators.bivariate.Zena()
data = sz.rvs(1000)

# Apply Cade to estimate the density of the data. Cade works by
# first fitting an initial naive joint density model and subsequently
# improving the initial density estimates with a classifier that
# tries to distinguish between the real data versus fake data sampled
# from the initial density model
cade = Cade(
    initial_density=JointDensity(verbose=1),
    classifier=Lgbm(verbose=1)
)
diagnostics = cade.train(data, diagnostics=True)
val_df = diagnostics['val_df']
real_df = val_df[val_df.truth==1]
real_df.pred.describe()
print("The classifier achieved AUROC = " + str(round(diagnostics['auc'], 3)))

# Apply fastKDE (pip install fastkde)
from fastkde import fastKDE
myPDF, axes = fastKDE.pdf(data.iloc[:,0].values, data.iloc[:,1].values)

# Check the estimates against the generative density
val_df = pd.DataFrame({
    'cade_est': cade.density(data),
    #'fast_kde_est': myPDF(data),
    # Since this is a simulation, we get to observe the generative density:
    'gen': sz.pdf(data.values)
})
cor = round(val_df.cade_est.corr(val_df.gen, method='spearman'), 3)
print("Spearman correlation, estimated vs generative density: " + str(cor))
pdb.set_trace()

# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
# import numpy as np
# pd.Series(stats.truncnorm.rvs(-2, 4, size=1000)).hist()
# plt.show()