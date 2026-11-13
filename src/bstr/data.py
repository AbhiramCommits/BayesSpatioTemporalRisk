"""Shared loaders for the exposure table, ground truth, and the common holdout split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, HOLDOUT_FRACTION, HOLDOUT_SEED_OFFSET, SimConfig


def load_exposure_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_parquet(Path(data_dir) / "exposure_table.parquet")
    return df.sort_values(["cell_id", "time_block"]).reset_index(drop=True)


def load_ground_truth(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "ground_truth.parquet")


def attach_truth(df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join the evaluation-only truth columns (never used as model inputs)."""
    cols = [
        "cell_id",
        "time_block",
        "true_log_rate",
        "true_rate_per_1k_miles",
        "phi_true",
        "tau_true",
        "region_true",
    ]
    return df.merge(truth[cols], on=["cell_id", "time_block"], how="left", validate="one_to_one")


def create_holdout_mask(
    df: pd.DataFrame, cfg: SimConfig, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Draw and persist THE holdout split reused by every model in the repository."""
    rng = np.random.default_rng(cfg.SEED + HOLDOUT_SEED_OFFSET)
    n = len(df)
    n_hold = int(round(HOLDOUT_FRACTION * n))
    idx = rng.permutation(n)[:n_hold]
    mask = df[["cell_id", "time_block"]].copy()
    mask["holdout"] = False
    mask.loc[mask.index[idx], "holdout"] = True
    mask = mask.sort_values(["cell_id", "time_block"]).reset_index(drop=True)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    mask.to_parquet(data_dir / "holdout_mask.parquet", index=False)
    return mask


def load_holdout_mask(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "holdout_mask.parquet")


def merge_holdout(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(mask, on=["cell_id", "time_block"], how="left", validate="one_to_one")
    if out["holdout"].isna().any():
        raise ValueError("holdout mask does not cover every exposure-table row")
    out["holdout"] = out["holdout"].astype(bool)
    return out
