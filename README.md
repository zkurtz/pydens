[![Build Status](https://travis-ci.com/zkurtz/pydens.svg?branch=master)](https://travis-ci.com/zkurtz/pydens)
# pydens, density estimation in python

**pydens** provides a unified interface to several density estimation packages, 
including an implementation of 
[classifier-adjusted density 
estimation](https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf).
Examples in the `/notebooks` directory include
- [Basic usage and testing](notebooks/demo.ipynb).
- [Identifying the common and the rare in Census data](notebooks/anomalous_demographics.ipynb)

Applications of density estimation include
- mode detection: Locating regions of high density is a first step to efficiently
allocate resources to address an epidemic, market a product, etc.
- feature engineering: The density at a point with respect to any subset of the dimensions of a feature
space can encode unique information. 
- anomaly detection: A "point of low density" is a common working definition of "anomaly",
 although it's not the only one. (In astrostatistics, for example,
 a density spike may draw attention as a possible galaxy.)

Disclaimer: 
- This is a young and relatively untested repo. See [the wishlist](#Wishlist).
- While CADE may be a contender for the present state-of-the-art, 
[there is a case to be made](https://github.com/Microsoft/LightGBM/issues/2056) 
that a more efficient approach should soon render CADE largely obsolete.

## Installation

Not yet on pypi or conda forge, but installation is still easy with pip:
```buildoutcfg
pip install git+https://github.com/zkurtz/pydens.git#egg=pydens
```

## License

MIT. See LICENSE.

## Related work

- A 'nearest neighbors' [fastkde](https://github.com/mjenrungrot/fastKDE)
- [Random forests](https://github.com/ksanjeevan/randomforest-density-python)
- [Isolation forests](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)
for density ranking
- [Intersection](https://medium.com/datadriveninvestor/generating-fake-data-density-estimation-and-generative-adversarial-networks-3606a37fa95)
of density estimation and generative adversarial networks

## Wishlist

Infrastructure:
- expand code testing coverage
- build type-checking methods to enforce consistent outputs
- define additional performance metrics
- define new simulations and real-data benchmarks

Tutorials, starting with
- understanding density estimation metrics
- how CADE works
- density estimation trees

Density estmation:
- Optimize CADE default settings
- Merge the best of the tree-based methods of LightGBM, 
[detpack](https://cran.r-project.org/web/packages/detpack/index.html),
and 
[astropy.stats.bayesian_blocks](http://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html)
