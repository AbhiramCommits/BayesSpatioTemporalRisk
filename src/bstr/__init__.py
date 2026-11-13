"""BayesSpatioTemporalRisk: Bayesian spatio-temporal point-process risk modelling.

Modules
-------
config    : frozen simulation/prior configuration objects and project paths
grid      : square-cell geometry and rook contiguity
simulate  : seeded ground-truth simulator for raw event and mileage logs
aggregate : PySpark/SQL rollup of raw logs to a cell x time-block exposure table
baselines : frequentist pooled / fixed-effects / negative-binomial Poisson GLMs
model     : PyMC hierarchical ICAR + AR(1) Poisson model with exposure offset
checks    : posterior predictive checks, LOO/WAIC comparison, prior sensitivity
readout   : posterior assurance decisions and the stakeholder memo
"""

__version__ = "0.1.0"
