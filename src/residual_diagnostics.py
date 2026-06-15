from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import periodogram


@dataclass(frozen=True)
class ResidualDiagnostics:
    n_samples: int
    residual_mean: float
    residual_std: float
    residual_median: float
    residual_q95: float
    residual_q99: float
    max_positive_residual: float
    positive_area: float
    positive_sample_ratio: float
    positive_run_count: int
    longest_positive_run_samples: int
    longest_positive_run_duration_s: float
    dominant_period_s: float | None
    dominant_frequency_hz: float | None
    dominant_power: float | None


def compute_residual_series(
    df: pd.DataFrame,
    value_column: str,
    background_column: str,
) -> pd.DataFrame:
    """Return count-rate residuals after removing the estimated local background."""
    if value_column not in df.columns:
        raise ValueError(f"{value_column} column is required")
    if background_column not in df.columns:
        raise ValueError(f"{background_column} column is required")

    result = df.copy()
    residual = result[value_column].astype(float) - result[background_column].astype(float)
    result["residual_count_rate"] = residual
    result["positive_residual_count_rate"] = residual.clip(lower=0.0)
    return result


def _positive_runs(mask: pd.Series) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    values = mask.fillna(False).astype(bool).tolist()
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def estimate_dominant_period(
    residual: pd.Series,
    sample_interval_s: float,
) -> dict[str, float] | None:
    """Estimate the strongest non-zero residual period using a periodogram."""
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")

    values = residual.astype(float).dropna().to_numpy()
    if len(values) < 4:
        return None
    centered = values - float(np.mean(values))
    if np.allclose(centered, 0.0):
        return None

    frequencies, powers = periodogram(centered, fs=1.0 / sample_interval_s, scaling="spectrum")
    valid = frequencies > 0
    if not np.any(valid):
        return None

    valid_frequencies = frequencies[valid]
    valid_powers = powers[valid]
    peak_idx = int(np.argmax(valid_powers))
    dominant_frequency = float(valid_frequencies[peak_idx])
    if dominant_frequency <= 0:
        return None
    return {
        "dominant_frequency_hz": dominant_frequency,
        "dominant_period_s": float(1.0 / dominant_frequency),
        "dominant_power": float(valid_powers[peak_idx]),
    }


def compute_residual_diagnostics(
    df: pd.DataFrame,
    value_column: str,
    background_column: str,
    sample_interval_s: float,
    positive_threshold: float = 0.0,
) -> ResidualDiagnostics:
    """Summarize residual structure after subtracting the estimated background."""
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")

    residual_df = compute_residual_series(
        df,
        value_column=value_column,
        background_column=background_column,
    )
    residual = residual_df["residual_count_rate"].astype(float)
    positive = residual_df["positive_residual_count_rate"].astype(float)
    mask = residual >= positive_threshold
    if positive_threshold <= 0:
        mask = residual > 0
    runs = _positive_runs(mask)
    longest_run = max((end - start + 1 for start, end in runs), default=0)
    period = estimate_dominant_period(residual, sample_interval_s=sample_interval_s)

    return ResidualDiagnostics(
        n_samples=int(len(residual)),
        residual_mean=float(residual.mean()),
        residual_std=float(residual.std(ddof=1)) if len(residual) > 1 else 0.0,
        residual_median=float(residual.median()),
        residual_q95=float(residual.quantile(0.95)),
        residual_q99=float(residual.quantile(0.99)),
        max_positive_residual=float(positive.max()),
        positive_area=float(positive.sum() * sample_interval_s),
        positive_sample_ratio=float((positive > 0).mean()),
        positive_run_count=int(len(runs)),
        longest_positive_run_samples=int(longest_run),
        longest_positive_run_duration_s=float(longest_run * sample_interval_s),
        dominant_period_s=period["dominant_period_s"] if period else None,
        dominant_frequency_hz=period["dominant_frequency_hz"] if period else None,
        dominant_power=period["dominant_power"] if period else None,
    )
