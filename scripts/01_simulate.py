"""Step 1: simulate raw event/mileage logs and the ground-truth rate surface."""

from __future__ import annotations

import time

from bstr.config import SimConfig
from bstr.simulate import simulate_ground_truth, simulate_raw_logs


def main() -> None:
    t0 = time.time()
    cfg = SimConfig()
    raw = simulate_raw_logs(cfg)
    truth = simulate_ground_truth(cfg)
    elapsed = time.time() - t0

    print(f"raw event rows      : {raw['event_rows']:,}")
    print(f"raw mileage rows    : {raw['mileage_rows']:,}")
    print(f"total fleet miles   : {raw['total_miles']:,.0f}")
    print(f"total events        : {raw['total_events']:,}")
    print(f"realized events/1k  : {raw['realized_rate_per_1k_miles']:.4f}")
    print(f"ground-truth rows   : {len(truth):,} (cell x time block)")
    print(f"simulation seconds  : {elapsed:.1f}")


if __name__ == "__main__":
    main()
