from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datas" / "data.xlsx"
NEW_DATA_DIR = ROOT / "new_data"
OUTPUT_DIR = ROOT / "outputs"
NEW_OUTPUT_DIR = OUTPUT_DIR / "new_data"

SAMPLE_INTERVAL_S = 0.1
ENGINEERING_THRESHOLD = 85.4
REFERENCE_BACKGROUND = 60.0
BACKGROUND_DELTA = ENGINEERING_THRESHOLD - REFERENCE_BACKGROUND
CSV_HIGH_COUNT_QUANTILE = 0.99
POISSON_EFFECTIVE_WINDOW_S = 1.0
POISSON_CONFIDENCE_LEVELS = (0.95, 0.99)


@dataclass(frozen=True)
class BackgroundStrategy:
    name: str
    method: str
    initial_seconds: float = 5.0
    window_seconds: float = 10.0
    quantile: float = 0.2
    alpha: float = 0.03


BACKGROUND_STRATEGIES = [
    BackgroundStrategy(name="initial_5s", method="initial", initial_seconds=5.0),
    BackgroundStrategy(name="rolling_q20_10s", method="rolling_quantile", window_seconds=10.0, quantile=0.2),
    BackgroundStrategy(name="exponential_alpha_003", method="exponential", alpha=0.03),
]
