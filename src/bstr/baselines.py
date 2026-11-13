"""Frequentist Poisson / negative-binomial baselines fitted with statsmodels.

All models use offset = log(exposure_miles / 1000), so the linear predictor is on
the events-per-1,000-miles scale. Predictions are compared with the simulated
ground truth on every cell x time block and on the shared holdout split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

COVARIATES = ["intersection_density", "pedestrian_density"]


@dataclass
class BaselineFit:
    name: str
    result: object
    fixed_effects: bool
    n_regions: int
    n_time_blocks: int


def design_matrix(
    df: pd.DataFrame,
    fixed_effects: bool = False,
    n_regions: int = 6,
    n_time_blocks: int = 24,
) -> pd.DataFrame:
    """Intercept + standardized covariates + optional unpooled region/time dummies."""
    X = pd.DataFrame(
        {
            "const": 1.0,
            "intersection_density": df["intersection_density"].to_numpy(float),
            "pedestrian_density": df["pedestrian_density"].to_numpy(float),
        },
        index=df.index,
    )
    if fixed_effects:
        regions = pd.Categorical(df["region_id"], categories=range(n_regions))
        blocks = pd.Categorical(df["time_block"], categories=range(n_time_blocks))
        reg = pd.get_dummies(regions, prefix="region", drop_first=True, dtype=float)
        tb = pd.get_dummies(blocks, prefix="tb", drop_first=True, dtype=float)
        X = pd.concat([X, reg.set_axis(df.index), tb.set_axis(df.index)], axis=1)
    return X


def _offset(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["exposure_miles"].to_numpy(float) / 1000.0)


def fit_pooled_poisson(df: pd.DataFrame) -> BaselineFit:
    """Naive fleet-wide Poisson rate: intercept + covariates only."""
    X = design_matrix(df, fixed_effects=False)
    res = sm.GLM(
        df["event_count"].to_numpy(float), X, family=sm.families.Poisson(), offset=_offset(df)
    ).fit()
    return BaselineFit("pooled_poisson", res, False, 6, 24)


def fit_fixed_effects_poisson(
    df: pd.DataFrame, n_regions: int = 6, n_time_blocks: int = 24
) -> BaselineFit:
    """Unpooled region and time-block fixed effects plus covariates."""
    X = design_matrix(df, True, n_regions, n_time_blocks)
    res = sm.GLM(
        df["event_count"].to_numpy(float), X, family=sm.families.Poisson(), offset=_offset(df)
    ).fit()
    return BaselineFit("fixed_effects_poisson", res, True, n_regions, n_time_blocks)


def fit_negative_binomial(
    df: pd.DataFrame, n_regions: int = 6, n_time_blocks: int = 24
) -> BaselineFit:
    """Same design as the fixed-effects GLM, with an estimated NB dispersion."""
    X = design_matrix(df, True, n_regions, n_time_blocks)
    res = sm.NegativeBinomial(
        df["event_count"].to_numpy(float), X, offset=_offset(df)
    ).fit(disp=0, maxiter=1000)
    return BaselineFit("negative_binomial", res, True, n_regions, n_time_blocks)


def predict_rate(fit: BaselineFit, df: pd.DataFrame) -> np.ndarray:
    """Predicted events per 1,000 miles (the exposure offset cancels)."""
    X = design_matrix(df, fit.fixed_effects, fit.n_regions, fit.n_time_blocks)
    mu = fit.result.predict(X, offset=_offset(df))
    return np.asarray(mu, float) / (df["exposure_miles"].to_numpy(float) / 1000.0)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def evaluate_baseline(
    fit: BaselineFit,
    df_all: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    beta_true: tuple[float, float],
) -> dict:
    """Metrics, Wald intervals, and out-of-sample deviance for one baseline."""
    res = fit.result
    rate_all = predict_rate(fit, df_all)
    rate_hold = predict_rate(fit, holdout)

    X_train = design_matrix(train, fit.fixed_effects, fit.n_regions, fit.n_time_blocks)
    X_hold = design_matrix(holdout, fit.fixed_effects, fit.n_regions, fit.n_time_blocks)
    mu_train = np.asarray(res.predict(X_train, offset=_offset(train)))
    mu_hold = np.asarray(res.predict(X_hold, offset=_offset(holdout)))
    y_train = train["event_count"].to_numpy(float)
    y_hold = holdout["event_count"].to_numpy(float)

    ci = res.conf_int()
    cov_ci = {}
    for i, name in enumerate(COVARIATES):
        low, high = float(ci.loc[name, 0]), float(ci.loc[name, 1])
        cov_ci[name] = {
            "estimate": float(res.params[name]),
            "ci_low": low,
            "ci_high": high,
            "true_beta": float(beta_true[i]),
            "covers_truth": bool(low <= beta_true[i] <= high),
        }

    truth = df_all["true_rate_per_1k_miles"].to_numpy(float)
    fleet_rate = float(
        np.sum(rate_all * df_all["exposure_miles"].to_numpy(float))
        / np.sum(df_all["exposure_miles"].to_numpy(float))
    )
    # Poisson-variance Pearson statistic, computed uniformly across GLM and NB fits.
    pearson = float(np.sum((y_train - mu_train) ** 2 / np.clip(mu_train, 1e-12, None)))
    df_resid = int(len(train) - len(res.params))
    return {
        "name": fit.name,
        "k": int(len(res.params)),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "log_likelihood": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "pearson_chi2": pearson,
        "pearson_chi2_per_df": float(pearson / df_resid),
        "rmse": float(np.sqrt(np.mean((rate_all - truth) ** 2))),
        "mae": float(np.mean(np.abs(rate_all - truth))),
        "rmse_holdout": float(
            np.sqrt(np.mean((rate_hold - holdout["true_rate_per_1k_miles"].to_numpy(float)) ** 2))
        ),
        "holdout_poisson_deviance": poisson_deviance(y_hold, mu_hold),
        "train_poisson_deviance": poisson_deviance(y_train, mu_train),
        "fleet_rate_per_1k_miles": fleet_rate,
        "covariates": cov_ci,
    }


def run_baselines(
    train: pd.DataFrame,
    df_all: pd.DataFrame,
    holdout: pd.DataFrame,
    beta_true: tuple[float, float],
    artifacts_dir: Path,
) -> dict:
    fits = [
        fit_pooled_poisson(train),
        fit_fixed_effects_poisson(train),
        fit_negative_binomial(train),
    ]
    metrics = {
        fit.name: evaluate_baseline(fit, df_all, train, holdout, beta_true) for fit in fits
    }
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "baseline_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "model": name,
                "k": m["k"],
                "loglik": round(m["log_likelihood"], 1),
                "AIC": round(m["aic"], 1),
                "BIC": round(m["bic"], 1),
                "chi2/df": round(m["pearson_chi2_per_df"], 2),
                "RMSE": round(m["rmse"], 4),
                "MAE": round(m["mae"], 4),
                "holdout dev": round(m["holdout_poisson_deviance"], 1),
            }
        )
    return pd.DataFrame(rows)
