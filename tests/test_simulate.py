"""Seeded simulator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bstr import simulate as simulate_module
from bstr.config import SimConfig
from bstr.simulate import simulate


def test_seeded_simulation_is_reproducible(tiny_env):
    cfg = tiny_env.cfg
    simulate_module.simulate.cache_clear()
    first = simulate(cfg)
    simulate_module.simulate.cache_clear()
    second = simulate(cfg)

    pd.testing.assert_frame_equal(first.truth, second.truth)
    pd.testing.assert_frame_equal(first.raw_events, second.raw_events)
    pd.testing.assert_frame_equal(first.raw_mileage, second.raw_mileage)
    assert np.array_equal(first.counts, second.counts)


def test_realized_rate_matches_target():
    cfg = SimConfig()
    sim = simulate(cfg)
    realized = sim.params["realized_rate_per_1k_miles"]
    assert abs(realized - cfg.TARGET_RATE_PER_1K) / cfg.TARGET_RATE_PER_1K < 0.06
    assert sim.params["total_fleet_miles_drawn"] > 1.0e6


def test_exposure_strictly_positive(tiny_env):
    assert (tiny_env.exposure["exposure_miles"] > 0).all()
    assert tiny_env.exposure["exposure_miles"].sum() > 0
    assert (tiny_env.exposure["event_count"] >= 0).all()
