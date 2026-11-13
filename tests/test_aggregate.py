"""Spark/SQL rollup tests on the real tiny fixture."""

from __future__ import annotations

import pandas as pd

from bstr.aggregate import validate_exposure_table


def test_event_counts_are_conserved(tiny_env):
    stats = tiny_env.stats
    raw = pd.read_parquet(tiny_env.data_dir / "raw_events.parquet")
    assert stats["raw_event_rows"] == len(raw)
    assert stats["total_events"] == len(raw) - stats["dropped_events"]
    assert int(tiny_env.exposure["event_count"].sum()) == stats["total_events"]


def test_zero_event_cells_are_retained(tiny_env):
    assert (tiny_env.exposure["event_count"] == 0).any()
    expected = tiny_env.cfg.N_CELLS * tiny_env.cfg.N_TIME_BLOCKS
    assert len(tiny_env.exposure) == expected - tiny_env.stats["dropped_pairs"]


def test_validation_passes(tiny_env):
    audit = validate_exposure_table(
        tiny_env.exposure,
        raw_event_count=tiny_env.stats["raw_event_rows"],
        dropped_events=tiny_env.stats["dropped_events"],
    )
    assert audit["rows"] == len(tiny_env.exposure)
    assert (tiny_env.exposure["event_count"] >= tiny_env.exposure["critical_count"]).all()
