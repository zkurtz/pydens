# pydens, density estimation in python

A scalable and modular implementation of 
[classifier-adjusted density estimation](https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf)
with a unified interface to other density estimation packages.

Disclaimer: This is a young repo with many basic TODOs remaining:
- define performance benchmarks
- set up a code testing framework
- wrap fastkde and scipy.stats.gaussian_kde

Applications of density estimation include feature engineering (the density at a point can be a useful 
feature in supervised learning) and anomaly detection (points of low density can be interpreted as anomalies).

## Related work

Kernel density estimation:
- Fast Kernel Density Estimation, [fastkde](https://bitbucket.org/lbl-cascade/fastkde/src/master/)

Tree-based density estimation:
- [Random forests](https://github.com/ksanjeevan/randomforest-density-python)
- [Isolation forests](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)
for density ranking

Other:
- [Intersection](https://medium.com/datadriveninvestor/generating-fake-data-density-estimation-and-generative-adversarial-networks-3606a37fa95)
of density estimation and generative adversarial networks

## License

See LICENSE.