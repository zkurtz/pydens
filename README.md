[![Build Status](https://travis-ci.com/zkurtz/pydens.svg?branch=master)](https://travis-ci.com/zkurtz/pydens)
# pydens, density estimation in python

A scalable and modular implementation of 
[classifier-adjusted density estimation](https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf)
with a unified interface to other density estimation packages.
LightGBM is the default regression backend, but it's easy to replace this.

Applications of density estimation include
- feature engineering, since the density at a point can be a useful 
feature in supervised learning
- anomaly detection, since "a point of low density" is more or less what "anomaly" means

Disclaimer: This is a young repo with many basic TODOs remaining:
- define performance benchmarks
- vastly expand code testing coverage
- wrap fastkde, possibly interpolating on its output using `scipy.interpolate.griddata`
- wrap scipy.stats.gaussian_kde
- wrap a basic Voronoi tessellation implementation

## Installation

Not yet on pypi or conda forge, but installation is still easy with pip:
```buildoutcfg
pip install --upgrade pip
pip install numpy
pip install Cython
pip install -r requirements.txt
pip install .
# pip install git+https://github.com/zkurtz/pydens.git#egg=pydens
```

## Related work

Kernel density estimation:
- \[the original??\] [fastkde](https://bitbucket.org/lbl-cascade/fastkde/src/master/)
- A 'nearest neighbors' [fastkde](https://github.com/mjenrungrot/fastKDE)

Tree-based density estimation:
- [Random forests](https://github.com/ksanjeevan/randomforest-density-python)
- [Isolation forests](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)
for density ranking

Other:
- [Intersection](https://medium.com/datadriveninvestor/generating-fake-data-density-estimation-and-generative-adversarial-networks-3606a37fa95)
of density estimation and generative adversarial networks

## License

See LICENSE.