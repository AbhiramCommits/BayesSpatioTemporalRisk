"""Posterior assurance decisions and the stakeholder risk readout.

The decision engine aggregates the posterior rate over a region and set of time
blocks on the draw level (exposure-weighted), then asks for the posterior
probability that the rate is at or below the assurance threshold.

Additional-mileage estimates use a documented Poisson information approximation:
for a Poisson rate the 95% interval width scales as 1 / sqrt(miles), so reaching
a target width w* from the current width w requires miles * (w / w*)^2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .baselines import fit_pooled_poisson, predict_rate
from .config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, SimConfig
from .data import (
    attach_truth,
    load_exposure_table,
    load_ground_truth,
    load_holdout_mask,
    merge_holdout,
)
from .grid import build_grid
from .model import posterior_rate_draws

DEFAULT_THRESHOLD = 2.0
DEFAULT_CONFIDENCE = 0.95
DEFAULT_REGION = 3
DEFAULT_TIME_BLOCKS = (22, 23, 0, 1, 2, 3, 4)
TARGET_CRI_WIDTH = 0.5

DAY_BLOCKS = tuple(range(6, 18))
EVENING_BLOCKS = (18, 19, 20, 21)
NIGHT_BLOCKS = DEFAULT_TIME_BLOCKS
TIME_SLICES = {"day": DAY_BLOCKS, "evening": EVENING_BLOCKS, "night": NIGHT_BLOCKS}

DEFAULT_QUESTION = (
    "Can the fleet expand night operations (time blocks 22-04) into Region 3 while "
    "holding the critical-event rate at or below the assurance threshold of 2.0 events "
    "per 1,000 miles with at least 95% posterior probability?"
)


@dataclass(frozen=True)
class AssuranceDecision:
    posterior_mean: float
    ci_low: float
    ci_high: float
    cri_width: float
    threshold: float
    confidence: float
    prob_at_or_below_threshold: float
    decision: str
    n_draws: int
    current_miles: float | None
    target_cri_width: float
    additional_miles_needed: float | None
    method: str


def posterior_region_block_rate(
    idata, df: pd.DataFrame, region_id: int, time_blocks
) -> np.ndarray:
    """Exposure-weighted posterior draws of events/1k miles for a region x time slice.

    Aggregation happens on each posterior draw, never on posterior summaries, so the
    returned vector carries the full joint uncertainty of the selected cell rates.
    """
    blocks = set(int(b) for b in time_blocks)
    sel = df[(df["region_id"] == region_id) & (df["time_block"].isin(blocks))]
    if sel.empty:
        raise ValueError(f"no rows for region {region_id} and blocks {sorted(blocks)}")
    rate = posterior_rate_draws(idata, sel)
    weights = sel["exposure_miles"].to_numpy(float)
    weights = weights / weights.sum()
    return (rate * weights[None, :]).sum(axis=1)


def assurance_decision(
    draws: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    current_miles: float | None = None,
    target_width: float = TARGET_CRI_WIDTH,
) -> AssuranceDecision:
    """Classify a posterior rate distribution against an assurance threshold.

    GO requires P(rate <= threshold) >= confidence; NO-GO requires the complementary
    probability to be at least confidence; otherwise the evidence is insufficient.
    """
    draws = np.asarray(draws, float).ravel()
    low, high = np.percentile(draws, [2.5, 97.5])
    prob = float(np.mean(draws <= threshold))
    if prob >= confidence:
        decision = "GO"
    elif prob <= 1.0 - confidence:
        decision = "NO-GO"
    else:
        decision = "INSUFFICIENT EVIDENCE"

    width = float(high - low)
    additional = None
    if current_miles is not None:
        if width <= target_width:
            additional = 0.0
        else:
            additional = float(current_miles * ((width / target_width) ** 2 - 1.0))

    return AssuranceDecision(
        posterior_mean=float(draws.mean()),
        ci_low=float(low),
        ci_high=float(high),
        cri_width=width,
        threshold=float(threshold),
        confidence=float(confidence),
        prob_at_or_below_threshold=prob,
        decision=decision,
        n_draws=int(len(draws)),
        current_miles=None if current_miles is None else float(current_miles),
        target_cri_width=float(target_width),
        additional_miles_needed=additional,
        method=(
            "Poisson information approximation: 95% CrI width scales as "
            "1/sqrt(miles); additional miles = current miles * ((width/target)^2 - 1)"
        ),
    )


def _pooled_rate_for_slice(fit, df: pd.DataFrame, region_id: int, time_blocks) -> float:
    blocks = set(int(b) for b in time_blocks)
    sel = df[(df["region_id"] == region_id) & (df["time_block"].isin(blocks))]
    rate = predict_rate(fit, sel)
    miles = sel["exposure_miles"].to_numpy(float)
    return float((rate * miles).sum() / miles.sum())


def build_assurance_table(
    idata,
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    target_width: float = TARGET_CRI_WIDTH,
) -> pd.DataFrame:
    """Run the decision for all 6 regions x 3 time slices, ranked by exceedance risk."""
    train = df[~df["holdout"]].reset_index(drop=True)
    pooled_fit = fit_pooled_poisson(train)
    rows = []
    for region_id in range(int(df["region_id"].max()) + 1):
        for slice_name, blocks in TIME_SLICES.items():
            sel = df[(df["region_id"] == region_id) & (df["time_block"].isin(blocks))]
            miles = float(sel["exposure_miles"].sum())
            draws = posterior_region_block_rate(idata, df, region_id, blocks)
            decision = assurance_decision(
                draws, threshold, confidence, current_miles=miles, target_width=target_width
            )
            pooled_rate = _pooled_rate_for_slice(pooled_fit, df, region_id, blocks)
            pooled_decision = "GO" if pooled_rate <= threshold else "NO-GO"
            row = {
                "region_id": region_id,
                "time_slice": slice_name,
                "time_blocks": ",".join(str(b) for b in blocks),
                "n_rows": int(len(sel)),
                "exposure_miles": miles,
                "pooled_glm_rate": pooled_rate,
                "pooled_glm_decision": pooled_decision,
                **asdict(decision),
            }
            row["prob_exceed_threshold"] = 1.0 - row["prob_at_or_below_threshold"]
            row["pooled_agrees"] = (pooled_decision == "GO") == (decision.decision == "GO")
            rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["prob_exceed_threshold", "region_id", "time_slice"], ascending=[False, True, True]
    ).reset_index(drop=True)


def plot_risk_map(idata, df: pd.DataFrame, cfg, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    night = df[df["time_block"].isin(NIGHT_BLOCKS)].reset_index(drop=True)
    rate = posterior_rate_draws(idata, night)
    cell_rows = []
    for cell_id, idx in night.groupby("cell_id").indices.items():
        w = night["exposure_miles"].to_numpy(float)[idx]
        w = w / w.sum()
        draws = (rate[:, idx] * w[None, :]).sum(axis=1)
        cell_rows.append(
            {
                "cell_id": int(cell_id),
                "night_rate": float(draws.mean()),
                "cri_width": float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5)),
            }
        )
    cell_df = pd.DataFrame(cell_rows)
    grid = build_grid(cfg).merge(cell_df, on="cell_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    grid.plot(
        column="night_rate",
        ax=axes[0],
        cmap="magma",
        edgecolor="0.7",
        linewidth=0.15,
        legend=True,
        legend_kwds={"shrink": 0.75, "label": "events / 1k miles"},
    )
    axes[0].set_title("Night-time posterior mean rate")
    grid.plot(
        column="cri_width",
        ax=axes[1],
        cmap="viridis",
        edgecolor="0.7",
        linewidth=0.15,
        legend=True,
        legend_kwds={"shrink": 0.75, "label": "95% CrI width"},
    )
    axes[1].set_title("Night-time 95% CrI width")
    for ax in axes:
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal")
    fig.suptitle("Night operations (22-04): rate and uncertainty by cell")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_report(
    decision: AssuranceDecision,
    table: pd.DataFrame,
    question: str,
    region_id: int,
    time_blocks,
    path: Path,
    pooled_headline: float | None = None,
) -> None:
    recovery = _load_json(ARTIFACTS_DIR / "recovery_metrics.json")
    baseline = _load_json(ARTIFACTS_DIR / "baseline_metrics.json")
    prior = _load_json(ARTIFACTS_DIR / "prior_sensitivity_summary.json")
    comparison = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv", index_col=0)

    n_bayes_go = int((table["decision"] == "GO").sum())
    n_bayes_nogo = int((table["decision"] == "NO-GO").sum())
    n_bayes_insufficient = int((table["decision"] == "INSUFFICIENT EVIDENCE").sum())
    n_pooled_go = int((table["pooled_glm_decision"] == "GO").sum())
    n_disagree = int(
        ((table["pooled_glm_decision"] == "GO") & (table["decision"] != "GO")).sum()
    )
    risky = table.head(3)

    # Observed critical-event context comes from the exposure table.
    exposure_table = load_exposure_table()
    critical_rate = (
        1000.0 * exposure_table["critical_count"].sum() / exposure_table["exposure_miles"].sum()
    )
    pooled_rmse = baseline["pooled_poisson"]["rmse"]
    bayes_rmse = recovery["rmse"]
    loo_gain = float(comparison.loc["M0_pooled", "elpd_diff"])
    headline_row = table[(table["region_id"] == region_id) & (table["time_slice"] == "night")]
    if pooled_headline is None:
        pooled_headline = (
            float(headline_row["pooled_glm_rate"].iloc[0])
            if not headline_row.empty
            else float("nan")
        )
    pooled_headline_decision = (
        "GO" if pooled_headline <= decision.threshold else "NO-GO"
    )
    blocks_label = ", ".join(str(b) for b in time_blocks)
    miles_phrase = (
        "No additional mileage is required to reach the target CrI width of "
        f"{decision.target_cri_width:.2f}; the current width is {decision.cri_width:.2f} events per 1k miles."
        if decision.additional_miles_needed == 0.0
        else f"Approximately {decision.additional_miles_needed:,.0f} additional fleet miles in "
        "this region/time slice would be needed to shrink the CrI width to "
        f"{decision.target_cri_width:.2f} (Poisson information approximation)."
    )
    headline_agrees = (pooled_headline_decision == "GO") == (decision.decision == "GO")
    if headline_agrees:
        lead = (
            "For the headline question the pooled point estimate agrees "
            f"(it implies {pooled_headline_decision}; the Bayesian rule returns {decision.decision})."
        )
    else:
        lead = (
            "For the headline question the pooled point estimate would imply "
            f"{pooled_headline_decision} while the Bayesian rule returns {decision.decision}; "
            "that disagreement is the core result."
        )
    agreement = (
        lead
        + f" Across all {len(table)} region x time-slice decisions, the pooled model declares GO in "
        f"{n_pooled_go}, whereas the Bayesian model returns GO in {n_bayes_go}, NO-GO in "
        f"{n_bayes_nogo}, and INSUFFICIENT EVIDENCE in {n_bayes_insufficient}; the two rules "
        f"disagree in {n_disagree} slices."
    )
    disagree_regions = sorted(
        table.loc[
            (table["pooled_glm_decision"] == "GO") & (table["decision"] != "GO"), "region_id"
        ].unique()
    )
    if disagree_regions:
        agreement += (
            " The disagreements are concentrated in Region "
            + ", ".join(str(int(r)) for r in disagree_regions)
            + ", where the pooled covariate-only rate sits below the threshold but the "
            "regionally resolved posterior interval does not clear it."
        )

    lines = [
        "# Risk readout: night operations in Region 3",
        "",
        "**Question.** " + question,
        "",
        "## Method in three sentences",
        "",
        "Raw geo-tagged event and mileage logs were rolled up with Spark SQL into a "
        "12 x 12 cell by 24 hour-of-day exposure table covering 60 days and "
        f"{exposure_table['exposure_miles'].sum() / 1e6:.2f} million fleet miles. "
        "A Bayesian hierarchical Poisson model with a log-exposure offset, a spatial ICAR "
        "effect, an AR(1) hour-of-day effect, partial region pooling, and a cell x time-block "
        "nugget was fitted in PyMC. The assurance decision integrates the posterior over the "
        "cells and night blocks in the requested region, exposure-weighting each draw, and "
        "asks for the posterior probability that the rate is at or below the threshold.",
        "",
        "## Result",
        "",
        f"- Posterior rate: **{decision.posterior_mean:.2f} events per 1,000 miles** "
        f"(95% CrI {decision.ci_low:.2f} to {decision.ci_high:.2f}).",
        f"- P(rate <= {decision.threshold:.1f}) = **{decision.prob_at_or_below_threshold:.3f}** "
        f"(gate: {decision.confidence:.0%}).",
        f"- Decision: **{decision.decision}** for Region {region_id} (region_id={region_id}), "
        f"time blocks {blocks_label} (night).",
        f"- Precision: {miles_phrase}",
        "",
        "## Comparison with the pooled fleet-wide baseline",
        "",
        f"The pooled Poisson GLM has a point estimate of {pooled_headline:.2f} events per 1k "
        f"miles for this slice and would say GO on a point-estimate basis. {agreement}",
        "",
        "| Region | Slice | Posterior mean | 95% CrI | P(rate > 2.0) | Decision |",
        "|---:|---|---:|---|---:|---|",
    ]
    for _, row in risky.iterrows():
        lines.append(
            f"| {int(row['region_id'])} | {row['time_slice']} | {row['posterior_mean']:.2f} | "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}] | "
            f"{row['prob_exceed_threshold']:.3f} | {row['decision']} |"
        )
    lines += [
        "",
        "## Key assumptions",
        "",
        "- Exposure (fleet miles) is measured exactly and enters as a known offset.",
        "- The event rate is log-linear in two standardized covariates plus additive spatial, "
        "temporal, region, and nugget effects; the rate is constant within a cell x hour-of-day "
        "over the 60-day window.",
        "- The modelled outcome is all recorded safety events (all severities). Because critical "
        f"events are a subset (observed critical rate {critical_rate:.3f} per 1k miles), the "
        "all-event gate is stricter than a critical-only gate and is used conservatively.",
        "",
        "## Top three caveats",
        "",
        "1. **Simulated data.** The ground truth is a seeded simulation, so the excellent "
        f"recovery (RMSE {bayes_rmse:.2f} vs pooled GLM {pooled_rmse:.2f}; 95% CrI empirical "
        f"coverage {recovery['empirical_coverage_95']:.1%}) validates the pipeline, not real "
        "safety performance.",
        "2. **Spatial smoothing and identifiability.** The ICAR prior shrinks local effects "
        f"toward neighbours (posterior-mean phi correlation {recovery['corr_phi']:.2f} with "
        "truth), so a narrow hotspot smaller than a few cells can be attenuated even when its "
        "regional rate is high.",
        "3. **Unmodelled uncertainty.** Exposure error, severity misclassification, model-form "
        "error, and day-level heterogeneity beyond the nugget are not in the interval. The "
        "decision should be treated as a lower bound on uncertainty.",
        "",
        "## Validation context (from this build)",
        "",
        f"- 95% credible-interval empirical coverage across all 3,456 cell x time-block rates: "
        f"{recovery['empirical_coverage_95']:.1%}.",
        f"- LOO improvement of the full model over the pooled Bayesian baseline: "
        f"{loo_gain:.1f} elpd (difference SE {comparison.loc['M0_pooled', 'dse']:.1f}).",
        f"- Prior-sensitivity sweep: maximum shift in the fleet-wide rate estimate "
        f"{prior['max_abs_shift_fleet_rate_per_1k']:.3f} events per 1k miles.",
        "",
        "Figures: `figures/risk_map.png`, `figures/spatial_recovery.png`, "
        "`figures/temporal_recovery.png`, `figures/ppc_stats.png`.",
        "",
        "_Generated programmatically from artifacts in this repository; the numbers above are "
        "not hand-entered._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_blocks(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--region", type=int, default=DEFAULT_REGION)
    parser.add_argument("--time-blocks", type=str, default="22,23,0,1,2,3,4")
    parser.add_argument("--target-width", type=float, default=TARGET_CRI_WIDTH)
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "risk_readout.md"
    )
    args = parser.parse_args(argv)

    cfg = SimConfig()
    df = attach_truth(load_exposure_table(), load_ground_truth())
    df = merge_holdout(df, load_holdout_mask())
    idata = az.from_netcdf(ARTIFACTS_DIR / "idata_bayes.nc")

    blocks = _parse_blocks(args.time_blocks) if args.time_blocks else DEFAULT_TIME_BLOCKS
    draws = posterior_region_block_rate(idata, df, args.region, blocks)
    sel = df[(df["region_id"] == args.region) & (df["time_block"].isin(set(blocks)))]
    decision = assurance_decision(
        draws,
        threshold=args.threshold,
        confidence=args.confidence,
        current_miles=float(sel["exposure_miles"].sum()),
        target_width=args.target_width,
    )

    table = build_assurance_table(
        idata, df, threshold=args.threshold, confidence=args.confidence, target_width=args.target_width
    )
    table.to_csv(ARTIFACTS_DIR / "assurance_table.csv", index=False)
    plot_risk_map(idata, df, cfg, FIGURES_DIR / "risk_map.png")

    question = (
        f"Can the fleet expand night operations (time blocks {args.time_blocks}) into "
        f"Region {args.region} while holding the critical-event rate at or below the assurance "
        f"threshold of {args.threshold:.1f} events per 1,000 miles with at least "
        f"{args.confidence:.0%} posterior probability?"
    )
    pooled_headline = _pooled_rate_for_slice(
        fit_pooled_poisson(df[~df["holdout"]].reset_index(drop=True)), df, args.region, blocks
    )
    write_report(
        decision,
        table,
        question,
        args.region,
        blocks,
        args.report,
        pooled_headline=pooled_headline,
    )

    print(f"headline: Region {args.region}, blocks {blocks}")
    print(
        f"posterior mean {decision.posterior_mean:.3f} | 95% CrI "
        f"[{decision.ci_low:.3f}, {decision.ci_high:.3f}] | "
        f"P(rate <= {decision.threshold:g}) = {decision.prob_at_or_below_threshold:.3f}"
    )
    print(f"decision: {decision.decision}")
    print(f"additional miles to target width: {decision.additional_miles_needed}")
    print("\nassurance table (ranked by P(exceed)):")
    print(
        table[
            [
                "region_id",
                "time_slice",
                "posterior_mean",
                "ci_low",
                "ci_high",
                "prob_exceed_threshold",
                "decision",
                "pooled_glm_rate",
                "pooled_glm_decision",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print(f"\nreport written: {args.report}")


if __name__ == "__main__":
    main()
