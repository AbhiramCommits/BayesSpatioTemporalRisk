"""Seeded ground-truth simulator for raw event and mileage logs.

The data-generating process is a log-linear inhomogeneous Poisson point process
with a log-exposure offset::

    y[c, t, d] ~ Poisson( exposure_miles[c, t, d] / 1000
                          * exp(alpha + region[c] + phi[c] + tau[t] + X[c] @ beta) )

Raw logs are intentionally row-level and geo-tagged: the Spark layer, not the
simulator, performs the spatial join and rollup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, DATA_DIR, SimConfig, seed_for
from .grid import assign_regions, build_grid

SEVERITIES = np.array(["minor", "major", "critical"])
SEVERITY_PROBS = np.array([0.72, 0.23, 0.05])
EVENT_TYPES = np.array(
    ["pedestrian_conflict", "lane_departure", "hard_braking", "near_miss", "other"]
)
EVENT_TYPE_PROBS = np.array([0.22, 0.20, 0.30, 0.18, 0.10])


@dataclass
class Simulation:
    """All deterministic pieces of one simulated world."""

    grid: gpd.GeoDataFrame
    truth: pd.DataFrame
    exposure_day: pd.DataFrame  # cell_id, time_block, day_index, exposure_miles
    counts: np.ndarray  # (n_cells, n_time_blocks, n_days)
    raw_events: pd.DataFrame
    raw_mileage: pd.DataFrame
    params: dict


def _smooth_field(
    centroids: np.ndarray, sd: float, length_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean Gaussian field with an exponential covariance, via Cholesky."""
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    cov = sd**2 * np.exp(-dist / length_scale)
    cov += 1e-8 * np.eye(len(centroids))
    chol = np.linalg.cholesky(cov)
    field = chol @ rng.normal(size=len(centroids))
    return field - field.mean()


def _ar1_series(n: int, rho: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary zero-mean AR(1) series (not wrapped; block 23 does not touch block 0)."""
    innov_sd = sd * np.sqrt(1.0 - rho**2)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innov_sd)
    return out - out.mean()


def _exposure_factors(cfg: SimConfig, grid: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell, hour-of-day, and day multipliers for mean fleet miles (each mean 1)."""
    cy = (cfg.N_ROWS - 1) / 2.0
    cx = (cfg.N_COLS - 1) / 2.0
    radial = ((grid["row"].to_numpy() - cy) / (cfg.N_ROWS / 2.0)) ** 2 + (
        (grid["col"].to_numpy() - cx) / (cfg.N_COLS / 2.0)
    ) ** 2
    cell = 0.45 + 1.10 * np.exp(-radial / 2.0)
    cell = cell / cell.mean()

    hours = np.arange(cfg.N_TIME_BLOCKS)
    block = (
        0.55
        + 0.90 * np.exp(-(((hours - 8.0) ** 2) / 6.0))
        + 0.80 * np.exp(-(((hours - 17.5) ** 2) / 5.0))
        + 0.45 * np.exp(-(((hours - 22.0) ** 2) / 8.0))
    )
    block = block / block.mean()

    day = np.exp(np.random.default_rng(seed_for(cfg, 11)).normal(0.0, 0.15, cfg.N_DAYS))
    day = day / day.mean()
    return cell, block, day


@lru_cache(maxsize=4)
def simulate(cfg: SimConfig) -> Simulation:
    """Generate the complete simulated world once (cached per config)."""
    grid = build_grid(cfg)
    n_cells, n_t, n_days = cfg.N_CELLS, cfg.N_TIME_BLOCKS, cfg.N_DAYS
    centroids = grid[["centroid_x", "centroid_y"]].to_numpy()

    # --- Latent ground truth -------------------------------------------------
    phi_true = _smooth_field(
        centroids, cfg.SPATIAL_SD, cfg.SPATIAL_LENGTH_SCALE, np.random.default_rng(seed_for(cfg, 1))
    )
    tau_true = _ar1_series(
        n_t, cfg.TAU_RHO, cfg.TAU_SD, np.random.default_rng(seed_for(cfg, 2))
    )
    region_id = assign_regions(grid, cfg)
    region_true = np.random.default_rng(seed_for(cfg, 3)).normal(0.0, cfg.REGION_SD, cfg.N_REGIONS)

    rng_cov = np.random.default_rng(seed_for(cfg, 4))
    x1 = _smooth_field(centroids, 1.0, 2.0, rng_cov)
    x2 = _smooth_field(centroids, 1.0, 1.4, rng_cov)
    covariates = np.column_stack([(x1 - x1.mean()) / x1.std(), (x2 - x2.mean()) / x2.std()])
    beta_true = np.asarray(cfg.BETA_TRUE, dtype=float)

    # --- Raw mileage segments (actual exposure) ------------------------------
    cell_factor, block_factor, day_factor = _exposure_factors(cfg, grid)
    base = cfg.TOTAL_FLEET_MILES / (n_cells * n_t * n_days)
    expected = (
        base
        * cell_factor[:, None, None]
        * block_factor[None, :, None]
        * day_factor[None, None, :]
    )
    rng_miles = np.random.default_rng(seed_for(cfg, 5))
    n_seg = rng_miles.poisson(expected / cfg.MEAN_SEGMENT_MILES)
    c_g, t_g, d_g = np.meshgrid(
        np.arange(n_cells), np.arange(n_t), np.arange(n_days), indexing="ij"
    )
    seg_c = np.repeat(c_g.ravel(), n_seg.ravel())
    seg_t = np.repeat(t_g.ravel(), n_seg.ravel())
    seg_d = np.repeat(d_g.ravel(), n_seg.ravel())
    n_segments = len(seg_c)
    miles = rng_miles.gamma(2.0, cfg.MEAN_SEGMENT_MILES / 2.0, size=n_segments)
    exposure_3d = np.zeros((n_cells, n_t, n_days))
    np.add.at(exposure_3d, (seg_c, seg_t, seg_d), miles)

    seg_x = (grid["col"].to_numpy()[seg_c] + rng_miles.uniform(0.0, 1.0, n_segments)) * cfg.CELL_KM
    seg_y = (grid["row"].to_numpy()[seg_c] + rng_miles.uniform(0.0, 1.0, n_segments)) * cfg.CELL_KM
    seg_seconds = (
        seg_d.astype(float) * 86400.0
        + seg_t.astype(float) * 3600.0
        + rng_miles.uniform(0.0, 3600.0, n_segments)
    )
    seg_ts = np.datetime_as_string(
        np.datetime64("2026-01-01T00:00:00") + seg_seconds.astype("timedelta64[s]"), unit="s"
    )
    seg_vehicle = rng_miles.integers(1, cfg.FLEET_SIZE + 1, n_segments)
    raw_mileage = pd.DataFrame(
        {
            "segment_id": [f"seg_{i:07d}" for i in range(n_segments)],
            "vehicle_id": "veh_" + pd.Series(seg_vehicle).astype(str).str.zfill(4),
            "timestamp": seg_ts,
            "x": seg_x,
            "y": seg_y,
            "miles": miles,
        }
    )

    # --- Tune alpha so the *realized expected* rate is exactly the target ----
    w_1k = exposure_3d / 1000.0
    struct_ct = (
        region_true[region_id][:, None]
        + phi_true[:, None]
        + tau_true[None, :]
        + covariates @ beta_true[:, None]
    )
    alpha_true = float(
        np.log(cfg.TARGET_RATE_PER_1K)
        - np.log((w_1k * np.exp(struct_ct[:, :, None])).sum() / w_1k.sum())
    )
    log_rate_ct = alpha_true + struct_ct
    rate_ct = np.exp(log_rate_ct)
    counts = np.random.default_rng(seed_for(cfg, 6)).poisson(w_1k * rate_ct[:, :, None])

    # --- Expand counts to raw geo-tagged events ------------------------------
    ec, et, ed = np.nonzero(counts)
    event_counts = counts[ec, et, ed]
    ev_c = np.repeat(ec, event_counts)
    ev_t = np.repeat(et, event_counts)
    ev_d = np.repeat(ed, event_counts)
    n_events = len(ev_c)
    rng_ev = np.random.default_rng(seed_for(cfg, 7))
    ev_x = (grid["col"].to_numpy()[ev_c] + rng_ev.uniform(0.0, 1.0, n_events)) * cfg.CELL_KM
    ev_y = (grid["row"].to_numpy()[ev_c] + rng_ev.uniform(0.0, 1.0, n_events)) * cfg.CELL_KM
    ev_seconds = (
        ev_d.astype(float) * 86400.0
        + ev_t.astype(float) * 3600.0
        + rng_ev.uniform(0.0, 3600.0, n_events)
    )
    ev_ts = np.datetime_as_string(
        np.datetime64("2026-01-01T00:00:00") + ev_seconds.astype("timedelta64[s]"), unit="s"
    )
    raw_events = pd.DataFrame(
        {
            "event_id": [f"evt_{i:07d}" for i in range(n_events)],
            "timestamp": ev_ts,
            "x": ev_x,
            "y": ev_y,
            "vehicle_id": "veh_"
            + pd.Series(rng_ev.integers(1, cfg.FLEET_SIZE + 1, n_events)).astype(str).str.zfill(4),
            "severity": rng_ev.choice(SEVERITIES, size=n_events, p=SEVERITY_PROBS),
            "event_type": rng_ev.choice(EVENT_TYPES, size=n_events, p=EVENT_TYPE_PROBS),
        }
    )

    # --- Ground-truth table (one row per cell x time block) ------------------
    cc, tt = np.meshgrid(np.arange(n_cells), np.arange(n_t), indexing="ij")
    truth = pd.DataFrame(
        {
            "cell_id": cc.ravel(),
            "time_block": tt.ravel(),
            "true_log_rate": log_rate_ct.ravel(),
            "true_rate_per_1k_miles": rate_ct.ravel(),
            "phi_true": np.repeat(phi_true, n_t),
            "tau_true": np.tile(tau_true, n_cells),
            "region_id": np.repeat(region_id, n_t),
            "region_true": np.repeat(region_true[region_id], n_t),
            "intersection_density": np.repeat(covariates[:, 0], n_t),
            "pedestrian_density": np.repeat(covariates[:, 1], n_t),
        }
    )

    exposure_day = pd.DataFrame(
        {
            "cell_id": seg_c,
            "time_block": seg_t,
            "day_index": seg_d,
            "exposure_miles": miles,
        }
    )

    params = {
        "seed": cfg.SEED,
        "alpha_true": alpha_true,
        "beta_true": beta_true.tolist(),
        "beta_names": ["intersection_density", "pedestrian_density"],
        "spatial_sd": cfg.SPATIAL_SD,
        "spatial_length_scale_cells": cfg.SPATIAL_LENGTH_SCALE,
        "tau_rho": cfg.TAU_RHO,
        "tau_sd": cfg.TAU_SD,
        "region_sd": cfg.REGION_SD,
        "region_true": region_true.tolist(),
        "target_rate_per_1k_miles": cfg.TARGET_RATE_PER_1K,
        "n_rows": cfg.N_ROWS,
        "n_cols": cfg.N_COLS,
        "n_time_blocks": cfg.N_TIME_BLOCKS,
        "n_days": cfg.N_DAYS,
        "fleet_size": cfg.FLEET_SIZE,
        "mean_segment_miles": cfg.MEAN_SEGMENT_MILES,
        "total_fleet_miles_drawn": float(miles.sum()),
        "raw_event_rows": int(n_events),
        "raw_mileage_rows": int(n_segments),
        "total_events": int(counts.sum()),
        "realized_rate_per_1k_miles": float(counts.sum() / w_1k.sum()),
    }

    return Simulation(
        grid=grid,
        truth=truth,
        exposure_day=exposure_day,
        counts=counts,
        raw_events=raw_events,
        raw_mileage=raw_mileage,
        params=params,
    )


def simulate_raw_logs(cfg: SimConfig, out_dir: Path = DATA_DIR) -> dict:
    """Write data/raw_events.parquet and data/raw_mileage.parquet."""
    sim = simulate(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.raw_events.to_parquet(out_dir / "raw_events.parquet", index=False)
    sim.raw_mileage.to_parquet(out_dir / "raw_mileage.parquet", index=False)
    return {
        "event_rows": len(sim.raw_events),
        "mileage_rows": len(sim.raw_mileage),
        "total_miles": float(sim.raw_mileage["miles"].sum()),
        "total_events": int(len(sim.raw_events)),
        "realized_rate_per_1k_miles": float(
            len(sim.raw_events) / (sim.raw_mileage["miles"].sum() / 1000.0)
        ),
    }


def simulate_ground_truth(
    cfg: SimConfig, out_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR
) -> pd.DataFrame:
    """Write data/ground_truth.parquet and artifacts/ground_truth_params.json."""
    sim = simulate(cfg)
    out_dir = Path(out_dir)
    artifacts_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sim.truth.to_parquet(out_dir / "ground_truth.parquet", index=False)
    with open(artifacts_dir / "ground_truth_params.json", "w") as fh:
        json.dump(sim.params, fh, indent=2)
    return sim.truth


def ground_truth_grid(cfg: SimConfig) -> gpd.GeoDataFrame:
    """Convenience accessor for the grid geometry used by figures and tests."""
    return build_grid(cfg)
