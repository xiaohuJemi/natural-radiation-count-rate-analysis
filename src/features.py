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


def build_caving_time_features(
    df: pd.DataFrame,
    engineering_threshold: float = 85.4,
) -> pd.DataFrame:
    """Add time-aware features for 0.1 s caving count-rate sequences."""
    out = df.copy()
    group_keys = ["source_file"] if "source_file" in out.columns else None

    def add_group_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        count = group["count_rate"]
        group["ma10_count_rate"] = count.rolling(10, min_periods=1).mean()
        group["ma30_count_rate"] = count.rolling(30, min_periods=1).mean()
        group["std10_count_rate"] = count.rolling(10, min_periods=2).std(ddof=1)
        group["std30_count_rate"] = count.rolling(30, min_periods=5).std(ddof=1)
        group["diff_count_rate"] = count.diff().fillna(0.0)
        group["slope10_count_rate"] = _rolling_slope(count, 10)
        group["above_85_4_raw"] = count >= engineering_threshold
        group["above_85_4_ma10"] = group["ma10_count_rate"] >= engineering_threshold
        group["above_85_4_ma30"] = group["ma30_count_rate"] >= engineering_threshold
        return group

    if group_keys:
        return out.groupby(group_keys, group_keys=False, sort=False).apply(add_group_features)
    return add_group_features(out)
