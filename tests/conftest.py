"""Session-scoped tiny world: 4 x 4 grid, 6 time blocks, real Spark rollup."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bstr.aggregate import run_aggregation
from bstr.config import SimConfig
from bstr.data import attach_truth
from bstr.grid import adjacency_edge_list, build_grid, rook_adjacency
from bstr.simulate import simulate_ground_truth, simulate_raw_logs


@pytest.fixture(scope="session")
def tiny_env(tmp_path_factory):
    cfg = SimConfig(
        N_ROWS=4,
        N_COLS=4,
        N_TIME_BLOCKS=6,
        N_DAYS=8,
        N_REGIONS=4,
        TOTAL_FLEET_MILES=30_000.0,
        FLEET_SIZE=20,
        MEAN_SEGMENT_MILES=3.0,
    )
    base = tmp_path_factory.mktemp("bstr-tiny")
    data = base / "data"
    data.mkdir()
    simulate_raw_logs(cfg, data)
    simulate_ground_truth(cfg, data, base / "artifacts")
    result = run_aggregation(cfg, data, data, verbose=False)
    truth = pd.read_parquet(data / "ground_truth.parquet")
    exposure = attach_truth(result["exposure"], truth)
    edges = adjacency_edge_list(rook_adjacency(build_grid(cfg)))
    return SimpleNamespace(
        cfg=cfg,
        data_dir=data,
        exposure=exposure,
        region=result["region"],
        stats=result["stats"],
        truth=truth,
        edges=edges,
    )
