"""Step 4: fit the full Bayesian hierarchical ICAR + AR(1) Poisson model."""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bstr.config import ARTIFACTS_DIR, FIGURES_DIR, Priors, SimConfig
from bstr.data import (
    attach_truth,
    load_exposure_table,
    load_ground_truth,
    load_holdout_mask,
    merge_holdout,
)
from bstr.grid import adjacency_edge_list, build_grid, rook_adjacency
from bstr.model import (
    build_model,
    convergence_diagnostics,
    fit_model,
    parameter_recovery,
    phi_summary,
    posterior_rate_summary,
    tau_summary,
)

DRAWS = 1500
TUNE = 1000
CHAINS = 4
TARGET_ACCEPT = 0.95
SEED = 20260101


def plot_spatial_recovery(truth, idata, cfg, path) -> None:
    grid = build_grid(cfg)
    phi_draws = phi_summary(idata)
    post = phi_draws.mean(axis=0)
    truth_cell = truth.drop_duplicates("cell_id").sort_values("cell_id").reset_index(drop=True)
    plot_df = grid.copy()
    plot_df["phi_true"] = truth_cell["phi_true"].to_numpy()
    plot_df["phi_post"] = post

    vmax = float(max(np.abs(plot_df["phi_true"]).max(), np.abs(plot_df["phi_post"]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col, title in zip(
        axes,
        ["phi_true", "phi_post"],
        ["True spatial effect $\\phi$", "Posterior mean $\\phi$"],
        strict=True,
    ):
        plot_df.plot(
            column=col,
            ax=ax,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            edgecolor="0.75",
            linewidth=0.15,
            legend=True,
            legend_kwds={"shrink": 0.75, "label": "log-rate effect"},
        )
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal")
    fig.suptitle("Spatial recovery: 12 x 12 ODD grid")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_recovery(truth, idata, path) -> None:
    tau_draws = tau_summary(idata, center=True)
    mean = tau_draws.mean(axis=0)
    low, high = np.percentile(tau_draws, [2.5, 97.5], axis=0)
    tau_true = truth.drop_duplicates("time_block").sort_values("time_block")["tau_true"].to_numpy()
    tau_true = tau_true - tau_true.mean()
    hours = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(hours, low, high, color="tab:blue", alpha=0.22, label="95% CrI")
    ax.plot(hours, mean, color="tab:blue", lw=2, label="posterior mean")
    ax.plot(hours, tau_true, color="black", ls="--", lw=2, label="ground truth")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hour of day (time block)")
    ax.set_ylabel("centered log-rate effect $\\tau_t$")
    ax.set_title("Temporal recovery of the AR(1) effect")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    cfg = SimConfig()
    priors = Priors()

    df = attach_truth(load_exposure_table(), load_ground_truth())
    df = merge_holdout(df, load_holdout_mask())
    train = df[~df["holdout"]].reset_index(drop=True)

    with open(ARTIFACTS_DIR / "ground_truth_params.json") as fh:
        params = json.load(fh)

    edges = adjacency_edge_list(rook_adjacency(build_grid(cfg)))
    model = build_model(train, edges, cfg, priors, variant="full")
    idata = fit_model(
        model,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT,
        seed=SEED,
    )
    idata.to_netcdf(ARTIFACTS_DIR / "idata_bayes.nc")

    rates = posterior_rate_summary(idata, df)
    rates.to_parquet(ARTIFACTS_DIR / "posterior_rates.parquet", index=False)

    diag = convergence_diagnostics(idata)
    covered = (rates["ci_low"] <= rates["true_rate_per_1k_miles"]) & (
        rates["true_rate_per_1k_miles"] <= rates["ci_high"]
    )
    err = rates["posterior_mean"] - rates["true_rate_per_1k_miles"]

    phi_post = phi_summary(idata).mean(axis=0)
    truth_cell = df.drop_duplicates("cell_id").sort_values("cell_id")
    tau_draws = tau_summary(idata, center=True)
    tau_post = tau_draws.mean(axis=0)
    tau_true = df.drop_duplicates("time_block").sort_values("time_block")["tau_true"].to_numpy()
    tau_true = tau_true - tau_true.mean()

    recovery = {
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "target_accept": TARGET_ACCEPT,
        "n_train": int(len(train)),
        "n_cell_time_rows": int(len(df)),
        **diag,
        "wall_clock_sampling_seconds": float(idata.attrs["sampling_seconds"]),
        "wall_clock_fit_seconds": float(idata.attrs["fit_seconds"]),
        "parameters": {
            "alpha": parameter_recovery(idata, "alpha", params["alpha_true"]),
        },
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "empirical_coverage_95": float(covered.mean()),
        "n_rate_intervals": int(len(rates)),
        "corr_phi": float(np.corrcoef(phi_post, truth_cell["phi_true"])[0, 1]),
        "corr_tau_centered": float(np.corrcoef(tau_post, tau_true)[0, 1]),
        "sigma_posterior_means": {
            name: float(idata.posterior[name].mean())
            for name in ("sigma_region", "sigma_phi", "sigma_tau", "sigma_nugget", "rho")
        },
    }
    # beta is a vector; parameter_recovery above flattens it, so fix the two entries.
    beta_draws = idata.posterior["beta"].values.reshape(-1, 2)
    for i, key in enumerate(["beta_intersection_density", "beta_pedestrian_density"]):
        low, high = np.percentile(beta_draws[:, i], [2.5, 97.5])
        recovery["parameters"][key] = {
            "truth": float(params["beta_true"][i]),
            "posterior_mean": float(beta_draws[:, i].mean()),
            "posterior_sd": float(beta_draws[:, i].std(ddof=1)),
            "ci_low": float(low),
            "ci_high": float(high),
            "covered": bool(low <= params["beta_true"][i] <= high),
        }

    with open(ARTIFACTS_DIR / "recovery_metrics.json", "w") as fh:
        json.dump(recovery, fh, indent=2)

    plot_spatial_recovery(df, idata, cfg, FIGURES_DIR / "spatial_recovery.png")
    plot_temporal_recovery(df, idata, FIGURES_DIR / "temporal_recovery.png")

    print(f"fit: {CHAINS} chains x {DRAWS} draws (tune {TUNE}), target_accept {TARGET_ACCEPT}")
    print(
        f"diagnostics: max R-hat {diag['max_rhat']:.4f} | min bulk ESS "
        f"{diag['min_ess_bulk']:.0f} | min tail ESS {diag['min_ess_tail']:.0f} | "
        f"divergences {diag['divergences']}"
    )
    print(f"sampling wall clock: {recovery['wall_clock_sampling_seconds']:.1f}s")
    print("\nparameter recovery (truth vs posterior):")
    for name, r in recovery["parameters"].items():
        print(
            f"  {name:26s} truth {r['truth']:+.3f}  post {r['posterior_mean']:+.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  covered={r['covered']}"
        )
    print(
        f"\nrate accuracy: RMSE {recovery['rmse']:.4f} | MAE {recovery['mae']:.4f} "
        "(baselines in artifacts/baseline_metrics.json)"
    )
    print(
        f"empirical 95% CrI coverage: {recovery['empirical_coverage_95']:.4f} "
        f"over {recovery['n_rate_intervals']} cell x time-block rates"
    )
    print(
        f"correlation: phi {recovery['corr_phi']:.3f} | tau(centered) "
        f"{recovery['corr_tau_centered']:.3f}"
    )
    print(f"total step wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
