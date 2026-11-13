"""Model construction, sampling sanity, and exposure-offset invariance."""

from __future__ import annotations

import numpy as np

from bstr.config import Priors
from bstr.model import build_model, fit_model


def test_full_model_builds(tiny_env):
    model = build_model(tiny_env.exposure, tiny_env.edges, tiny_env.cfg, variant="full")
    for name in ("alpha", "beta", "region_effect", "phi", "tau", "sigma_nugget", "y"):
        assert name in model.named_vars
    model_pooled = build_model(
        tiny_env.exposure, tiny_env.edges, tiny_env.cfg, variant="pooled"
    )
    assert "phi" not in model_pooled.named_vars
    assert "tau" not in model_pooled.named_vars


def test_sampling_produces_finite_log_likelihood(tiny_env):
    model = build_model(tiny_env.exposure, tiny_env.edges, tiny_env.cfg, variant="full")
    idata = fit_model(
        model,
        draws=100,
        tune=100,
        chains=1,
        target_accept=0.9,
        seed=7,
        include_posterior_predictive=False,
    )
    log_lik = idata.log_likelihood["y"].values
    assert np.isfinite(log_lik).all()
    assert log_lik.shape[-1] == len(tiny_env.exposure)


def test_doubling_exposure_and_counts_leaves_rate_unchanged(tiny_env):
    base = tiny_env.exposure.copy()
    base["exposure_miles"] = 5000.0
    base["event_count"] = 10
    doubled = base.copy()
    doubled["exposure_miles"] = 10000.0
    doubled["event_count"] = 20

    alpha = []
    for df in (base, doubled):
        model = build_model(
            df, tiny_env.edges, tiny_env.cfg, Priors(), variant="pooled"
        )
        idata = fit_model(
            model,
            draws=200,
            tune=200,
            chains=2,
            target_accept=0.9,
            seed=11,
            include_posterior_predictive=False,
        )
        alpha.append(float(idata.posterior["alpha"].mean()))
    assert abs(alpha[0] - alpha[1]) < 0.1
