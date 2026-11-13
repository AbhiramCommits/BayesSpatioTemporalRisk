"""Step 3: pooled, fixed-effects, and negative-binomial Poisson GLM baselines."""

from __future__ import annotations

import json

from bstr.baselines import comparison_frame, run_baselines
from bstr.config import ARTIFACTS_DIR, DATA_DIR, SimConfig
from bstr.data import (
    attach_truth,
    create_holdout_mask,
    load_exposure_table,
    load_ground_truth,
    merge_holdout,
)


def main() -> None:
    cfg = SimConfig()
    df = load_exposure_table()
    truth = load_ground_truth()
    df = attach_truth(df, truth)
    mask = create_holdout_mask(df, cfg, DATA_DIR)
    df = merge_holdout(df, mask)
    train = df[~df["holdout"]].copy()
    holdout = df[df["holdout"]].copy()

    with open(ARTIFACTS_DIR / "ground_truth_params.json") as fh:
        params = json.load(fh)

    metrics = run_baselines(train, df, holdout, tuple(params["beta_true"]), ARTIFACTS_DIR)
    print(f"train rows {len(train):,} / holdout rows {len(holdout):,}")
    print(comparison_frame(metrics).to_string(index=False))
    print("\ncovariate Wald 95% CIs vs truth:")
    for name, m in metrics.items():
        for cov, c in m["covariates"].items():
            print(
                f"  {name:24s} {cov:22s} {c['estimate']:+.3f} "
                f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
                f"truth {c['true_beta']:+.2f} covered={c['covers_truth']}"
            )


if __name__ == "__main__":
    main()
