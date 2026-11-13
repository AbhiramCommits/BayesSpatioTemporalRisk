"""Slow empirical-coverage test (excluded from CI with -m 'not slow')."""

from __future__ import annotations

import numpy as np
import pytest

from bstr.model import build_model, fit_model, posterior_rate_summary


@pytest.mark.slow
def test_empirical_coverage_on_small_grid(tiny_env):
    model = build_model(tiny_env.exposure, tiny_env.edges, tiny_env.cfg, variant="full")
    idata = fit_model(
        model,
        draws=400,
        tune=400,
        chains=2,
        target_accept=0.9,
        seed=123,
        include_posterior_predictive=False,
    )
    summary = posterior_rate_summary(idata, tiny_env.exposure)
    covered = (summary["ci_low"] <= summary["true_rate_per_1k_miles"]) & (
        summary["true_rate_per_1k_miles"] <= summary["ci_high"]
    )
    assert float(covered.mean()) >= 0.90
    assert np.isfinite(summary["posterior_mean"]).all()
