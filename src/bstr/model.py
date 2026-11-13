"""PyMC hierarchical spatio-temporal Poisson model with an exposure offset.

For cells c = 1..C and time blocks t = 1..T, with exposure e[c,t] in thousands of
miles, the full (M3) model is::

    y[c,t] ~ Poisson(e[c,t] * exp(eta[c,t]))
    eta = alpha + X[c] @ beta + region_effect[region(c)] + phi[c] + tau[t] + nugget[c,t]
    region_effect = sigma_region * region_raw,   region_raw ~ N(0, 1)
    phi = sigma_phi * phi_raw,                   phi_raw ~ ZeroSumNormal(1)
    p(phi) ∝ exp(-0.5 * sum_{(i,j) in E} (phi_i - phi_j)^2 / sigma_phi^2)   (ICAR)
    tau_raw[t] = rho * tau_raw[t-1] + eps[t],    eps ~ N(0, 1)              (AR(1))
    tau = sigma_tau * tau_raw
    nugget[c,t] = sigma_nugget * z[c,t],         z ~ N(0, 1)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .config import Priors, SimConfig

VARIANTS = ("pooled", "region", "time", "full")


def build_model(
    df: pd.DataFrame,
    adjacency_edges: tuple[np.ndarray, np.ndarray],
    cfg: SimConfig,
    priors: Priors | None = None,
    variant: str = "full",
) -> pm.Model:
    """Build the model on the rows in ``df`` (call with training rows only).

    Parameters
    ----------
    df
        Exposure-table rows with event_count, exposure_miles, covariates, region_id.
    adjacency_edges
        (node1, node2) rook-neighbour arrays for the ICAR prior.
    cfg
        Grid dimensions (cell/time/region indexes must be < the configured sizes).
    priors
        Overridable prior scales.
    variant
        "pooled", "region", "time", or "full".
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    priors = priors or Priors()

    n_cells = cfg.N_CELLS
    n_time = cfg.N_TIME_BLOCKS
    n_regions = cfg.N_REGIONS
    n_obs = len(df)

    cell_idx = df["cell_id"].to_numpy(dtype=int)
    time_idx = df["time_block"].to_numpy(dtype=int)
    region_idx = df["region_id"].to_numpy(dtype=int)
    X = df[["intersection_density", "pedestrian_density"]].to_numpy(dtype=float)
    exposure_1k = df["exposure_miles"].to_numpy(dtype=float) / 1000.0
    y = df["event_count"].to_numpy(dtype=int)

    coords = {
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
        "region": np.arange(n_regions),
        "obs": np.arange(n_obs),
    }
    edge1 = np.asarray(adjacency_edges[0], dtype=int)
    edge2 = np.asarray(adjacency_edges[1], dtype=int)

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X)
        exposure_data = pm.Data("exposure_1k", exposure_1k)

        alpha = pm.Normal("alpha", mu=priors.ALPHA_MU, sigma=priors.ALPHA_SIGMA)
        beta = pm.Normal("beta", mu=priors.BETA_MU, sigma=priors.BETA_SIGMA, shape=2)
        eta = alpha + pt.dot(X_data, beta)

        if variant in ("region", "time", "full"):
            sigma_region = pm.HalfNormal("sigma_region", sigma=priors.SIGMA_REGION)
            region_raw = pm.Normal("region_raw", 0.0, 1.0, shape=n_regions)
            region_effect = pm.Deterministic(
                "region_effect", sigma_region * region_raw, dims="region"
            )
            eta = eta + region_effect[region_idx]

        if variant in ("time", "full"):
            sigma_tau = pm.HalfNormal("sigma_tau", sigma=priors.SIGMA_TAU)
            rho_raw = pm.Beta("rho_raw", alpha=priors.RHO_A, beta=priors.RHO_B)
            rho = pm.Deterministic("rho", 2.0 * rho_raw - 1.0)
            tau_raw = pm.Normal("tau_raw", 0.0, 1.0, shape=n_time, dims="time")
            ar_quad = tau_raw[0] ** 2 * (1.0 - rho**2) + pt.sum(
                (tau_raw[1:] - rho * tau_raw[:-1]) ** 2
            )
            pm.Potential("ar1_prior", -0.5 * ar_quad)
            tau = pm.Deterministic("tau", sigma_tau * tau_raw, dims="time")
            eta = eta + tau[time_idx]

        if variant == "full":
            sigma_phi = pm.HalfNormal("sigma_phi", sigma=priors.SIGMA_PHI)
            phi_raw = pm.ZeroSumNormal("phi_raw", sigma=1.0, shape=n_cells, dims="cell")
            phi = pm.Deterministic("phi", sigma_phi * phi_raw, dims="cell")
            icar_quad = pt.sum((phi[edge1] - phi[edge2]) ** 2)
            pm.Potential("icar_prior", -0.5 * icar_quad / sigma_phi**2)
            eta = eta + phi[cell_idx]

            sigma_nugget = pm.HalfNormal("sigma_nugget", sigma=priors.SIGMA_NUGGET)
            nugget_raw = pm.Normal("nugget_raw", 0.0, 1.0, shape=n_obs, dims="obs")
            eta = eta + sigma_nugget * nugget_raw

        mu = exposure_data * pt.exp(eta)
        pm.Poisson("y", mu=mu, observed=y, dims="obs")

    return model


def fit_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 20260101,
    progressbar: bool = False,
) -> Any:
    """Sample the posterior (with log-likelihood) and the posterior predictive."""
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": True},
        )
        sample_seconds = time.time() - t0
        pm.sample_posterior_predictive(
            idata,
            model=model,
            extend_inferencedata=True,
            random_seed=seed + 1,
            progressbar=progressbar,
        )
    idata.attrs["sampling_seconds"] = float(sample_seconds)
    idata.attrs["fit_seconds"] = float(time.time() - t0)
    idata.attrs["draws"] = int(draws)
    idata.attrs["tune"] = int(tune)
    idata.attrs["chains"] = int(chains)
    idata.attrs["target_accept"] = float(target_accept)
    return idata


def _flat(da: Any) -> np.ndarray:
    """Flatten (chain, draw, ...) into (sample, ...)."""
    values = da.values
    return values.reshape((-1, *values.shape[2:]))


def posterior_rate_draws(idata: Any, df: pd.DataFrame) -> np.ndarray:
    """Structural events-per-1k-miles draws for every row in ``df``.

    Returns an array of shape (posterior sample, row). The unstructured nugget is
    not included: it is an observation-level overdispersion term, not part of the
    ground-truth rate surface. Posterior predictive checks use the
    ``posterior_predictive`` group, which does include it.
    """
    post = idata.posterior
    alpha = _flat(post["alpha"])
    beta = _flat(post["beta"])  # (sample, 2)
    X = df[["intersection_density", "pedestrian_density"]].to_numpy(float)
    eta = alpha[:, None] + beta @ X.T

    if "region_effect" in post:
        region_idx = df["region_id"].to_numpy(int)
        eta += _flat(post["region_effect"])[:, region_idx]
    if "tau" in post:
        time_idx = df["time_block"].to_numpy(int)
        eta += _flat(post["tau"])[:, time_idx]
    if "phi" in post:
        cell_idx = df["cell_id"].to_numpy(int)
        eta += _flat(post["phi"])[:, cell_idx]

    return np.exp(eta)


def posterior_rate_summary(idata: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Posterior mean, sd, and 95% equal-tailed CrI of events per 1k miles."""
    rate = posterior_rate_draws(idata, df)
    out = df[["cell_id", "time_block"]].copy()
    out["posterior_mean"] = rate.mean(axis=0)
    out["posterior_sd"] = rate.std(axis=0, ddof=1)
    out["ci_low"] = np.percentile(rate, 2.5, axis=0)
    out["ci_high"] = np.percentile(rate, 97.5, axis=0)
    if "true_rate_per_1k_miles" in df:
        out["true_rate_per_1k_miles"] = df["true_rate_per_1k_miles"].to_numpy(float)
    return out


def convergence_diagnostics(idata: Any) -> dict:
    """Max R-hat and min bulk/tail ESS, reduced per variable (never via to_array)."""
    import arviz as az

    variables = list(idata.posterior.data_vars)
    rhat = az.rhat(idata, var_names=variables)
    ess_bulk = az.ess(idata, var_names=variables, method="bulk")
    ess_tail = az.ess(idata, var_names=variables, method="tail")
    return {
        "max_rhat": max(float(rhat[v].max()) for v in rhat.data_vars),
        "min_ess_bulk": min(float(ess_bulk[v].min()) for v in ess_bulk.data_vars),
        "min_ess_tail": min(float(ess_tail[v].min()) for v in ess_tail.data_vars),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def parameter_recovery(idata: Any, name: str, truth: float) -> dict:
    """Posterior mean and 95% equal-tailed CrI for one scalar parameter."""
    draws = _flat(idata.posterior[name])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "truth": float(truth),
        "posterior_mean": float(draws.mean()),
        "posterior_sd": float(draws.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "covered": bool(low <= truth <= high),
    }


def tau_summary(idata: Any, center: bool = True) -> np.ndarray:
    """Posterior draws of tau as (sample, time), optionally mean-centered over time."""
    tau = _flat(idata.posterior["tau"])
    if center:
        tau = tau - tau.mean(axis=1, keepdims=True)
    return tau


def phi_summary(idata: Any) -> np.ndarray:
    """Posterior draws of phi as (sample, cell)."""
    return _flat(idata.posterior["phi"])
