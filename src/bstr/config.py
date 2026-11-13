"""Project paths and frozen configuration objects.

Every random draw in the repository is seeded from :class:`SimConfig.SEED` (with a
documented integer offset per stage), so a clean `make all` reproduces the same
data, fits, and reported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SimConfig:
    """Ground-truth simulation configuration.

    The grid is N_ROWS x N_COLS unit cells; time blocks are hours of day; each
    cell x time block is observed over N_DAYS days and rolled up into one row.
    """

    N_ROWS: int = 12
    N_COLS: int = 12
    N_TIME_BLOCKS: int = 24
    N_DAYS: int = 60
    CELL_KM: float = 1.0
    SEED: int = 20260101

    # Ground-truth hyperparameters
    SPATIAL_SD: float = 0.6
    SPATIAL_LENGTH_SCALE: float = 2.5  # in cells
    TAU_RHO: float = 0.75
    TAU_SD: float = 0.4
    REGION_SD: float = 0.35
    BETA_TRUE: tuple[float, float] = (0.45, 0.30)
    TARGET_RATE_PER_1K: float = 1.5

    # Exposure / fleet scale
    TOTAL_FLEET_MILES: float = 1_200_000.0
    FLEET_SIZE: int = 250
    MEAN_SEGMENT_MILES: float = 3.0

    @property
    def N_CELLS(self) -> int:
        return self.N_ROWS * self.N_COLS

    @property
    def N_REGIONS(self) -> int:
        return 6


@dataclass(frozen=True)
class Priors:
    """Priors for the Bayesian model; every scale is overridable (Step 5 sweeps them)."""

    ALPHA_MU: float = 0.4054651081081644  # log(1.5)
    ALPHA_SIGMA: float = 1.0
    BETA_MU: float = 0.0
    BETA_SIGMA: float = 1.0
    SIGMA_REGION: float = 1.0
    SIGMA_PHI: float = 1.0
    SIGMA_TAU: float = 1.0
    SIGMA_NUGGET: float = 1.0
    RHO_A: float = 2.0
    RHO_B: float = 2.0


# The single holdout split reused by every model in the repository.
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED_OFFSET = 303


def seed_for(cfg: SimConfig, offset: int) -> int:
    """Deterministic child seed for one simulation stage."""
    return int(cfg.SEED + offset)
