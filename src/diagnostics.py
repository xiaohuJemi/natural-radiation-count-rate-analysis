from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .models import (
    assign_threshold_states,
    compute_domain_thresholds,
    detect_changepoints,
    fit_isolation_forest,
    select_gmm_components,
)


def changepoint_max_segments_sensitivity(
    series: pd.Series,
    max_segments_values: list[int] | None = None,
    min_segment_size: int = 40,
) -> pd.DataFrame:
    max_segments_values = max_segments_values or [6, 8, 10, 12, 15]
    rows = []
    for max_segments in max_segments_values:
        result = detect_changepoints(series, min_segment_size=min_segment_size, max_segments=max_segments)
        rows.append(
            {
                "max_segments": max_segments,
                "min_segment_size": min_segment_size,
                "chosen_segments": result.chosen_segments,
                "at_max_segments": result.at_max_segments,
                "bic_converged": result.bic_converged,
                "min_bic": min(result.bic_by_segments.values()),
            }
        )
    return pd.DataFrame(rows)


def changepoint_min_size_sensitivity(
    series: pd.Series,
    min_segment_sizes: list[int] | None = None,
    max_segments: int = 15,
) -> pd.DataFrame:
    min_segment_sizes = min_segment_sizes or [30, 40, 60, 80]
    rows = []
    for min_size in min_segment_sizes:
        result = detect_changepoints(series, min_segment_size=min_size, max_segments=max_segments)
        rows.append(
            {
                "min_segment_size": min_size,
                "max_segments": max_segments,
                "chosen_segments": result.chosen_segments,
                "at_max_segments": result.at_max_segments,
                "bic_converged": result.bic_converged,
                "min_bic": min(result.bic_by_segments.values()),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    df: pd.DataFrame,
    baseline_windows: list[int] | None = None,
) -> pd.DataFrame:
    baseline_windows = baseline_windows or [100, 150, 200, 300]
    rows = []
    for window in baseline_windows:
        thresholds = compute_domain_thresholds(df, baseline_window=window)
        states = assign_threshold_states(df, thresholds)
        counts = states.value_counts()
        very_high_ratio = float(counts.get("very_high_count", 0) / len(states))
        baseline = df["ma30_count_rate"].dropna().iloc[: thresholds.baseline_window]
        shapiro_p = float(stats.shapiro(baseline).pvalue) if len(baseline) <= 5000 else float("nan")
        normaltest_p = float(stats.normaltest(baseline).pvalue) if len(baseline) >= 8 else float("nan")
        rows.append(
            {
                "baseline_window": thresholds.baseline_window,
                "baseline_mean": thresholds.baseline_mean,
                "baseline_std": thresholds.baseline_std,
                "warning_threshold": thresholds.warning_threshold,
                "high_threshold": thresholds.high_threshold,
                "very_high_threshold": thresholds.very_high_threshold,
                "very_high_count": int(counts.get("very_high_count", 0)),
                "very_high_ratio": very_high_ratio,
                "threshold_discriminative": very_high_ratio <= 0.30,
                "shapiro_p": shapiro_p,
                "normaltest_p": normaltest_p,
            }
        )
    return pd.DataFrame(rows)


def poisson_threshold_table(
    df: pd.DataFrame,
    baseline_windows: list[int] | None = None,
    z_values: list[float] | None = None,
) -> pd.DataFrame:
    baseline_windows = baseline_windows or [100, 150, 200, 300]
    z_values = z_values or [1.96, 3.0, 4.0]
    rows = []
    values = df["ma30_count_rate"].fillna(df["ma15_count_rate"])
    for window in baseline_windows:
        baseline = df["ma30_count_rate"].dropna().iloc[:window]
        lam = float(baseline.mean())
        for z in z_values:
            threshold = lam + z * np.sqrt(lam)
            rows.append(
                {
                    "baseline_window": len(baseline),
                    "z": z,
                    "poisson_threshold": float(threshold),
                    "n_above": int((values >= threshold).sum()),
                    "ratio_above": float((values >= threshold).mean()),
                }
            )
    return pd.DataFrame(rows)


def isolation_forest_sensitivity(
    df: pd.DataFrame,
    contaminations: list[float] | None = None,
) -> pd.DataFrame:
    contaminations = contaminations or [0.02, 0.04, 0.06, 0.08, 0.10]
    rows = []
    previous: set[int] | None = None
    for contamination in contaminations:
        flags = fit_isolation_forest(df, contamination=contamination)
        current = set(flags[flags].index.astype(int))
        overlap = len(current & previous) / len(current | previous) if previous is not None and current | previous else np.nan
        rows.append(
            {
                "contamination": contamination,
                "n_high_count_anomalies": len(current),
                "ratio": len(current) / len(df),
                "jaccard_vs_previous": float(overlap) if not np.isnan(overlap) else np.nan,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def write_diagnostics(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "changepoint_max_segments": diagnostics_dir / "changepoint_max_segments_sensitivity.csv",
        "changepoint_min_size": diagnostics_dir / "changepoint_min_size_sensitivity.csv",
        "threshold_sensitivity": diagnostics_dir / "threshold_sensitivity.csv",
        "poisson_thresholds": diagnostics_dir / "poisson_thresholds.csv",
        "gmm_model_selection": diagnostics_dir / "gmm_model_selection.csv",
        "isolation_forest_sensitivity": diagnostics_dir / "isolation_forest_sensitivity.csv",
    }

    changepoint_max_segments_sensitivity(df["ma30_count_rate"]).to_csv(
        outputs["changepoint_max_segments"], index=False, encoding="utf-8-sig"
    )
    changepoint_min_size_sensitivity(df["ma30_count_rate"]).to_csv(
        outputs["changepoint_min_size"], index=False, encoding="utf-8-sig"
    )
    threshold_sensitivity(df).to_csv(outputs["threshold_sensitivity"], index=False, encoding="utf-8-sig")
    poisson_threshold_table(df).to_csv(outputs["poisson_thresholds"], index=False, encoding="utf-8-sig")
    select_gmm_components(df).to_csv(outputs["gmm_model_selection"], index=False, encoding="utf-8-sig")
    isolation_forest_sensitivity(df).to_csv(
        outputs["isolation_forest_sensitivity"], index=False, encoding="utf-8-sig"
    )
    return outputs

