"""Step 5: PPC, LOO/WAIC comparison, held-out predictive density, prior sensitivity."""

from __future__ import annotations

import json
import time

import arviz as az
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from bstr.baselines import fit_fixed_effects_poisson, fit_negative_binomial, fit_pooled_poisson
from bstr.checks import (
    fit_bayesian_model_set,
    frequentist_log_predictive_density,
    holdout_log_predictive_density,
    model_comparison_table,
    plot_ppc_overlay,
    plot_ppc_stats,
    plot_prior_sensitivity,
    ppc_test_statistics,
    prior_sweep,
    save_json,
)
from bstr.config import ARTIFACTS_DIR, FIGURES_DIR, SimConfig
from bstr.data import (
    attach_truth,
    load_exposure_table,
    load_ground_truth,
    load_holdout_mask,
    merge_holdout,
)
from bstr.grid import adjacency_edge_list, build_grid, rook_adjacency


def main() -> None:
    t0 = time.time()
    cfg = SimConfig()
    df = attach_truth(load_exposure_table(), load_ground_truth())
    df = merge_holdout(df, load_holdout_mask())
    train = df[~df["holdout"]].reset_index(drop=True)
    holdout = df[df["holdout"]].reset_index(drop=True)
    edges = adjacency_edge_list(rook_adjacency(build_grid(cfg)))

    with open(ARTIFACTS_DIR / "ground_truth_params.json") as fh:
        params = json.load(fh)

    m3 = az.from_netcdf(ARTIFACTS_DIR / "idata_bayes.nc")

    # (a) posterior predictive checks
    ppc_df, rep = ppc_test_statistics(m3)
    plot_ppc_overlay(m3, FIGURES_DIR / "ppc_overlay.png")
    plot_ppc_stats(ppc_df, rep, FIGURES_DIR / "ppc_stats.png")
    save_json(
        {"statistics": ppc_df.to_dict(orient="records")},
        ARTIFACTS_DIR / "ppc_metrics.json",
    )
    print("(a) posterior predictive checks")
    print(ppc_df.to_string(index=False))

    # (b) Bayesian model comparison
    print("\n(b) fitting M0-M2 for LOO/WAIC comparison ...")
    fits = fit_bayesian_model_set(train, edges, cfg)
    fits["M3_full"] = m3
    comparison = model_comparison_table(fits)
    comparison.to_csv(ARTIFACTS_DIR / "model_comparison.csv")
    print(
        comparison[
            ["rank", "elpd_loo", "se", "p_loo", "elpd_diff", "dse", "weight", "waic_elpd"]
        ].to_string()
    )
    print("Pareto-k > 0.7 counts:")
    print(comparison["pareto_k_gt_0.7"].to_string())

    # Common out-of-sample metric: held-out log predictive density
    holdout_rows = []
    for name, idata in fits.items():
        metric = holdout_log_predictive_density(idata, holdout)
        holdout_rows.append({"model": name, **metric})
    for fit in (
        fit_pooled_poisson(train),
        fit_fixed_effects_poisson(train),
        fit_negative_binomial(train),
    ):
        metric = frequentist_log_predictive_density(fit, holdout)
        holdout_rows.append({"model": fit.name, **metric})
    holdout_table = pd.DataFrame(holdout_rows).sort_values(
        "mean_log_predictive_density", ascending=False
    )
    holdout_table.to_csv(ARTIFACTS_DIR / "holdout_comparison.csv", index=False)
    print("\nheld-out log predictive density (higher is better):")
    print(holdout_table.to_string(index=False))

    # (c) prior sensitivity sweep
    print("\n(c) prior sensitivity sweep ...")
    sweep = prior_sweep(df, edges, cfg, reference_idata=m3)
    sweep.to_csv(ARTIFACTS_DIR / "prior_sensitivity.csv", index=False)
    truth_rate = float(
        (df["true_rate_per_1k_miles"] * df["exposure_miles"]).sum()
        / df["exposure_miles"].sum()
    )
    plot_prior_sensitivity(sweep, params, truth_rate, FIGURES_DIR / "prior_sensitivity.png")
    shift = float(sweep["fleet_rate_per_1k"].max() - sweep["fleet_rate_per_1k"].min())
    summary = {
        "max_abs_shift_fleet_rate_per_1k": shift,
        "min_fleet_rate_per_1k": float(sweep["fleet_rate_per_1k"].min()),
        "max_fleet_rate_per_1k": float(sweep["fleet_rate_per_1k"].max()),
        "n_configs": int(len(sweep)),
        "truth_fleet_rate_per_1k": truth_rate,
    }
    save_json(summary, ARTIFACTS_DIR / "prior_sensitivity_summary.json")
    print(sweep[["config", "fleet_rate_per_1k", "coverage_95", "elpd_loo"]].to_string(index=False))
    print(f"max absolute shift in fleet-wide rate: {shift:.4f} events / 1k miles")
    print(f"\ntotal step wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
