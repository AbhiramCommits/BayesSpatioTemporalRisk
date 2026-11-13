"""Assurance decision logic."""

from __future__ import annotations

import numpy as np

from bstr.readout import assurance_decision


def test_go_when_far_below_threshold():
    draws = np.full(4000, 0.5)
    decision = assurance_decision(draws, threshold=2.0, confidence=0.95, current_miles=10_000)
    assert decision.decision == "GO"
    assert decision.prob_at_or_below_threshold == 1.0
    assert decision.additional_miles_needed == 0.0


def test_no_go_when_far_above_threshold():
    draws = np.full(4000, 3.0)
    decision = assurance_decision(draws, threshold=2.0, confidence=0.95)
    assert decision.decision == "NO-GO"
    assert decision.prob_at_or_below_threshold == 0.0


def test_insufficient_evidence_when_interval_straddles_threshold():
    rng = np.random.default_rng(0)
    draws = rng.lognormal(mean=np.log(2.0), sigma=0.8, size=20_000)
    decision = assurance_decision(draws, threshold=2.0, confidence=0.95)
    assert decision.decision == "INSUFFICIENT EVIDENCE"
    assert 0.05 < decision.prob_at_or_below_threshold < 0.95


def test_additional_miles_uses_poisson_information_rule():
    rng = np.random.default_rng(1)
    draws = rng.lognormal(mean=np.log(1.0), sigma=0.35, size=20_000)
    decision = assurance_decision(
        draws, threshold=2.0, confidence=0.95, current_miles=10_000, target_width=0.25
    )
    assert decision.cri_width > 0.25
    expected = 10_000 * ((decision.cri_width / 0.25) ** 2 - 1.0)
    assert np.isclose(decision.additional_miles_needed, expected, rtol=1e-9)
