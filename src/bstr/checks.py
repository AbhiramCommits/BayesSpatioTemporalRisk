"""Posterior predictive checks, Bayesian model comparison, and prior sensitivity."""

from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

from .config import Priors, SimConfig
from .model import (
    _flat,
    build_model,
    fit_model,
    posterior_rate_draws,
    posterior_rate_summary,
)

PPC_NAMES = [
    "total_events",
    "prop_zero_cells",
    "p99_count",
    "variance_to_mean",
]


def _statistics(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Counts has shape (draw, observation); returns one value per draw."""
    return {
        "total_events": counts.sum(axis=1),
        "prop_zero_cells": (counts == 0).mean(axis=1),
        "p99_count": np.percentile(counts, 99, axis=1),
        "variance_to_mean": counts.var(axis=1) / counts.mean(axis=1),
    }


def ppc_test_statistics(idata) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Bayesian p-values P(T_rep >= T_obs) for four posterior predictive statistics."""
    pp = idata.posterior_predictive["y"].values
    pp = pp.reshape(-1, pp.shape[-1])
    y_obs = np.asarray(idata.observed_data["y"].values)
    rep = _statistics(pp)
    obs = _statistics(y_obs[None, :])
    rows = []
    for name in PPC_NAMES:
        p = float(np.mean(rep[name] >= obs[name][0]))
        rows.append(
            {
                "statistic": name,
                "observed": float(obs[name][0]),
                "replicated_mean": float(rep[name].mean()),
                "p_value": p,
                "p_value_two_sided": float(2.0 * min(p, 1.0 - p)),
            }
        )
    return pd.DataFrame(rows), rep


def plot_ppc_overlay(idata, path: Path) -> None:
    import matplotlib.pyplot as plt

    pp = idata.posterior_predictive["y"].values
    pp = pp.reshape(-1, pp.shape[-1])
    y_obs = np.asarray(idata.observed_data["y"].values)
    rng = np.random.default_rng(0)
    subset = pp[rng.choice(len(pp), size=min(300, len(pp)), replace=False)]
    upper = min(15, int(np.percentile(subset, 99.9)) + 1)
    bins = np.arange(0, upper + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(y_obs, bins=bins, density=True, color="black", alpha=0.45, label="observed")
    ax.hist(
        subset.ravel(),
        bins=bins,
        density=True,
        histtype="step",
        color="tab:blue",
        lw=2,
        label="replicated",
    )
    ax.set_xlabel("events per cell x time block")
    ax.set_ylabel("density")
    ax.set_title("Posterior predictive count distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ppc_stats(ppc_df: pd.DataFrame, rep: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (_, row) in zip(axes.ravel(), ppc_df.iterrows(), strict=True):
        values = rep[row["statistic"]]
        ax.hist(values, bins=40, color="tab:blue", alpha=0.7)
        ax.axvline(row["observed"], color="black", lw=2, label="observed")
        ax.set_title(f"{row['statistic']}  (p = {row['p_value']:.3f})")
        ax.set_ylabel("replicated draws")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Posterior predictive test statistics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def holdout_log_predictive_density(idata, df: pd.DataFrame, n_quadrature: int = 9) -> dict:
    """Total and mean held-out Poisson log predictive density.

    For models with a nugget, the nugget is integrated out with Gauss-Hermite
    quadrature; otherwise the structural rate is used directly. The predictive
    density is log(mean over posterior draws of p(y_new | theta)).
    """
    rate = posterior_rate_draws(idata, df)
    mu = rate * (df["exposure_miles"].to_numpy(float) / 1000.0)
    y = df["event_count"].to_numpy(float)
    n_draws = len(mu)

    if "sigma_nugget" in idata.posterior:
        sigma = _flat(idata.posterior["sigma_nugget"])
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
        weights = weights / weights.sum()
        log_pred = np.empty_like(mu)
        chunk = 250
        for start in range(0, n_draws, chunk):
            stop = min(start + chunk, n_draws)
            m = mu[start:stop]
            s = sigma[start:stop, None]
            terms = [
                np.log(w) + poisson.logpmf(y[None, :], m * np.exp(s * z))
                for z, w in zip(nodes, weights, strict=True)
            ]
            log_pred[start:stop] = logsumexp(np.stack(terms, axis=0), axis=0)
    else:
        log_pred = poisson.logpmf(y[None, :], mu)

    pointwise = logsumexp(log_pred, axis=0) - np.log(n_draws)
    return {
        "log_predictive_density": float(pointwise.sum()),
        "mean_log_predictive_density": float(pointwise.mean()),
        "n_holdout": int(len(df)),
    }


def frequentist_log_predictive_density(fit, df: pd.DataFrame) -> dict:
    """Held-out log predictive density for a statsmodels Poisson/NB baseline."""
    from .baselines import design_matrix

    X = design_matrix(df, fit.fixed_effects, fit.n_regions, fit.n_time_blocks)
    mu = np.asarray(
        fit.result.predict(X, offset=np.log(df["exposure_miles"].to_numpy(float) / 1000.0))
    )
    y = df["event_count"].to_numpy(float)
    if fit.name == "negative_binomial":
        alpha = float(fit.result.params["alpha"])
        size = 1.0 / alpha
        prob = size / (size + mu)
        lp = nbinom.logpmf(y, size, prob)
    else:
        lp = poisson.logpmf(y, mu)
    return {
        "log_predictive_density": float(np.sum(lp)),
        "mean_log_predictive_density": float(np.mean(lp)),
        "n_holdout": int(len(df)),
    }


def fit_bayesian_model_set(
    train: pd.DataFrame,
    edges,
    cfg: SimConfig,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    seed: int = 20260101,
) -> dict:
    """Fit M0 pooled, M1 +region, M2 +AR(1); M3 is supplied separately from Step 4."""
    fits = {}
    for i, variant in enumerate(("pooled", "region", "time")):
        name = {"pooled": "M0_pooled", "region": "M1_region", "time": "M2_time"}[variant]
        model = build_model(train, edges, cfg, Priors(), variant=variant)
        fits[name] = fit_model(
            model,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=seed + 100 * (i + 1),
            include_posterior_predictive=False,
        )
    return fits


def model_comparison_table(fits: dict) -> pd.DataFrame:
    """LOO + WAIC table with Pareto-k diagnostics, sorted by LOO."""
    import arviz as az

    loo_table = az.compare(fits, ic="loo")
    waic_table = az.compare(fits, ic="waic")
    pareto = {}
    for name, idata in fits.items():
        loo = az.loo(idata, pointwise=True)
        k = np.asarray(loo.pareto_k.values).ravel()
        pareto[name] = int(np.sum(k > 0.7))
    out = loo_table.copy()
    out["waic_elpd"] = waic_table["elpd_waic"]
    out["waic_se"] = waic_table["se"]
    out["waic_p"] = waic_table["p_waic"]
    out["waic_elpd_diff"] = waic_table["elpd_diff"]
    out["waic_dse"] = waic_table["dse"]
    out["pareto_k_gt_0.7"] = [pareto[m] for m in out.index]
    out.index.name = "model"
    return out


def _config_metrics(idata, df: pd.DataFrame) -> dict:
    rate = posterior_rate_draws(idata, df)
    w = df["exposure_miles"].to_numpy(float)
    fleet_draws = (rate * w).sum(axis=1) / w.sum()
    summary = posterior_rate_summary(idata, df)
    covered = (summary["ci_low"] <= summary["true_rate_per_1k_miles"]) & (
        summary["true_rate_per_1k_miles"] <= summary["ci_high"]
    )

    alpha = _flat(idata.posterior["alpha"])
    beta = _flat(idata.posterior["beta"])
    out = {
        "alpha_mean": float(alpha.mean()),
        "alpha_low": float(np.percentile(alpha, 2.5)),
        "alpha_high": float(np.percentile(alpha, 97.5)),
        "beta1_mean": float(beta[:, 0].mean()),
        "beta1_low": float(np.percentile(beta[:, 0], 2.5)),
        "beta1_high": float(np.percentile(beta[:, 0], 97.5)),
        "beta2_mean": float(beta[:, 1].mean()),
        "beta2_low": float(np.percentile(beta[:, 1], 2.5)),
        "beta2_high": float(np.percentile(beta[:, 1], 97.5)),
        "fleet_rate_per_1k": float(fleet_draws.mean()),
        "fleet_rate_low": float(np.percentile(fleet_draws, 2.5)),
        "fleet_rate_high": float(np.percentile(fleet_draws, 97.5)),
        "coverage_95": float(covered.mean()),
    }
    for name in ("sigma_region", "sigma_phi", "sigma_tau", "sigma_nugget", "rho"):
        if name in idata.posterior:
            out[f"{name}_mean"] = float(idata.posterior[name].mean())

    import arviz as az

    loo = az.loo(idata, pointwise=True)
    k = np.asarray(loo.pareto_k.values).ravel()
    out.update(
        {
            "elpd_loo": float(loo.elpd_loo),
            "elpd_loo_se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "pareto_k_gt_0.7": int(np.sum(k > 0.7)),
        }
    )
    return out


def prior_sweep(
    df: pd.DataFrame,
    edges,
    cfg: SimConfig,
    draws: int = 250,
    tune: int = 250,
    chains: int = 2,
    seed: int = 20260101,
    reference_idata=None,
) -> pd.DataFrame:
    """Refit the full model across sigma x beta prior scales.

    The sigma scale is applied jointly to the region, spatial, temporal, and nugget
    half-normal priors; beta ~ Normal(0, beta_scale). A reference row from the Step 4
    fit (all scales 1) is appended when ``reference_idata`` is supplied.
    """
    train = df[~df["holdout"]].reset_index(drop=True)
    rows = []
    for s_idx, sigma_scale in enumerate((0.5, 1.0, 2.0)):
        for b_idx, beta_scale in enumerate((0.5, 2.0)):
            priors = Priors(
                SIGMA_REGION=sigma_scale,
                SIGMA_PHI=sigma_scale,
                SIGMA_TAU=sigma_scale,
                SIGMA_NUGGET=sigma_scale,
                BETA_SIGMA=beta_scale,
            )
            model = build_model(train, edges, cfg, priors, variant="full")
            idata = fit_model(
                model,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                seed=seed + 17 * (s_idx * 2 + b_idx),
                include_posterior_predictive=False,
            )
            metrics = _config_metrics(idata, df)
            metrics.update(
                {
                    "config": f"sigma={sigma_scale:g},beta={beta_scale:g}",
                    "sigma_scale": sigma_scale,
                    "beta_scale": beta_scale,
                    "draws": draws,
                    "chains": chains,
                }
            )
            rows.append(metrics)
            del model, idata
            gc.collect()

    if reference_idata is not None:
        metrics = _config_metrics(reference_idata, df)
        metrics.update(
            {
                "config": "sigma=1,beta=1 (Step 4 fit)",
                "sigma_scale": 1.0,
                "beta_scale": 1.0,
                "draws": int(reference_idata.attrs.get("draws", 0)),
                "chains": int(reference_idata.attrs.get("chains", 0)),
            }
        )
        rows.append(metrics)

    columns = [
        "config",
        "sigma_scale",
        "beta_scale",
        "draws",
        "chains",
        "alpha_mean",
        "alpha_low",
        "alpha_high",
        "beta1_mean",
        "beta1_low",
        "beta1_high",
        "beta2_mean",
        "beta2_low",
        "beta2_high",
        "fleet_rate_per_1k",
        "fleet_rate_low",
        "fleet_rate_high",
        "coverage_95",
        "elpd_loo",
        "elpd_loo_se",
        "p_loo",
        "pareto_k_gt_0.7",
        "sigma_region_mean",
        "sigma_phi_mean",
        "sigma_tau_mean",
        "sigma_nugget_mean",
        "rho_mean",
    ]
    return pd.DataFrame(rows)[columns]


def plot_prior_sensitivity(
    sweep: pd.DataFrame, truth: dict, fleet_rate_truth: float, path: Path
) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("alpha", "alpha_mean", "alpha_low", "alpha_high", truth["alpha_true"], r"$\alpha$"),
        (
            "beta1",
            "beta1_mean",
            "beta1_low",
            "beta1_high",
            truth["beta_true"][0],
            r"$\beta_{intersection}$",
        ),
        (
            "beta2",
            "beta2_mean",
            "beta2_low",
            "beta2_high",
            truth["beta_true"][1],
            r"$\beta_{pedestrian}$",
        ),
        (
            "fleet_rate",
            "fleet_rate_per_1k",
            "fleet_rate_low",
            "fleet_rate_high",
            fleet_rate_truth,
            "fleet rate / 1k miles",
        ),
    ]
    ordered = sweep.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (_, mean_col, low_col, high_col, truth_value, label) in zip(
        axes.ravel(), panels, strict=True
    ):
        y = np.arange(len(ordered))
        mean = ordered[mean_col].to_numpy()
        low = ordered[low_col].to_numpy()
        high = ordered[high_col].to_numpy()
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([mean - low, high - mean]),
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            capsize=3,
        )
        ax.axvline(truth_value, color="black", ls="--", lw=1.5, label="truth")
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["config"], fontsize=8)
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Prior sensitivity of key posteriors (95% CrI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def priors_for_variant(base: Priors, **overrides) -> Priors:
    """Small helper used by tests/notebooks."""
    return replace(base, **overrides)
