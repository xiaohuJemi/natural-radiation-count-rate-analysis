from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))

    def slope(window_values: np.ndarray) -> float:
        y = window_values.astype(float)
        y_centered = y - y.mean()
        return float(np.sum(x_centered * y_centered) / denominator)

    return values.rolling(window, min_periods=window).apply(slope, raw=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling, differential, and normalized time-series features."""
    out = df.copy()
    count = out["count_rate"]
    out["ma5_count_rate"] = count.rolling(5, min_periods=1).mean()
    out["ma15_count_rate"] = count.rolling(15, min_periods=1).mean()
    out["ma30_count_rate"] = count.rolling(30, min_periods=30).mean()
    out["ma60_count_rate"] = count.rolling(60, min_periods=60).mean()
    out["std15_count_rate"] = count.rolling(15, min_periods=5).std(ddof=1)
    out["std30_count_rate"] = count.rolling(30, min_periods=10).std(ddof=1)
    out["diff_count_rate"] = count.diff().fillna(0.0)
    out["diff_ma30_count_rate"] = out["ma30_count_rate"].diff().fillna(0.0)
    out["slope15_count_rate"] = _rolling_slope(count, 15)
    out["slope30_count_rate"] = _rolling_slope(count, 30)

    mean = float(count.mean())
    std = float(count.std(ddof=1))
    out["count_rate_z"] = (count - mean) / std
    ranks = count.rank(method="average")
    out["count_rate_percentile"] = (ranks - 1) / (len(out) - 1)
    out["is_local_peak"] = (
        (count > count.shift(1)) & (count > count.shift(-1))
    ).fillna(False)
    out["is_local_trough"] = (
        (count < count.shift(1)) & (count < count.shift(-1))
    ).fillna(False)
    return out
