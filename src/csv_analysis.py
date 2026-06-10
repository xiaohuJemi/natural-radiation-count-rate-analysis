from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CSV_HIGH_COUNT_QUANTILE, NEW_DATA_DIR, NEW_OUTPUT_DIR


@dataclass(frozen=True)
class CsvAnalysisOutputs:
    normalized_csv: Path
    summary_csv: Path
    minute_trend_csv: Path
    second_components_csv: Path
    component_summary_csv: Path
    high_runs_csv: Path
    figures: list[Path]


def load_long_csv(path: str | Path) -> pd.DataFrame:
    """Load the long CSV as an independent count-rate time series."""
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}")
    out = raw.iloc[:, [0, 2]].copy()
    out.columns = ["timestamp", "count_rate"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["count_rate"] = pd.to_numeric(out["count_rate"], errors="coerce")
    out = out.dropna(subset=["timestamp", "count_rate"]).reset_index(drop=True)
    out["row_number"] = np.arange(1, len(out) + 1)
    out["elapsed_s_by_timestamp"] = (out["timestamp"] - out["timestamp"].min()).dt.total_seconds()
    out["minute"] = out["timestamp"].dt.floor("1min")
    return out[
        [
            "row_number",
            "timestamp",
            "elapsed_s_by_timestamp",
            "minute",
            "count_rate",
        ]
    ]


def _summarize_true_runs(df: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    rows = []
    start: int | None = None
    values = df[flag_column].fillna(False).to_numpy(dtype=bool)
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        if start is not None and ((not flag) or idx == len(values) - 1):
            end = idx - 1 if not flag else idx
            segment = df.iloc[start : end + 1]
            rows.append(
                {
                    "run_id": len(rows) + 1,
                    "flag_column": flag_column,
                    "start_row": int(segment["row_number"].iloc[0]),
                    "end_row": int(segment["row_number"].iloc[-1]),
                    "start_timestamp": segment["timestamp"].iloc[0],
                    "end_timestamp": segment["timestamp"].iloc[-1],
                    "n_samples": int(len(segment)),
                    "duration_s_by_timestamp": float(
                        (segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]).total_seconds()
                    ),
                    "max_count_rate": float(segment["count_rate"].max()),
                    "mean_count_rate": float(segment["count_rate"].mean()),
                }
            )
            start = None
    return pd.DataFrame(rows)


def build_second_level_components(
    df: pd.DataFrame,
    background_window_s: int = 300,
    background_quantile: float = 0.2,
    smooth_window_s: int = 5,
    contribution_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate slow background and relative count-rate contribution on a second grid."""
    second = (
        df.groupby("timestamp")
        .agg(
            elapsed_s=("elapsed_s_by_timestamp", "first"),
            n_samples=("count_rate", "size"),
            count_rate_mean=("count_rate", "mean"),
            count_rate_median=("count_rate", "median"),
            count_rate_min=("count_rate", "min"),
            count_rate_max=("count_rate", "max"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    second["count_rate_smooth"] = second["count_rate_mean"].rolling(
        smooth_window_s,
        min_periods=1,
        center=True,
    ).mean()
    second["background_estimate"] = second["count_rate_smooth"].rolling(
        background_window_s,
        min_periods=max(10, min(background_window_s, 30)),
        center=True,
    ).quantile(background_quantile)
    second["background_estimate"] = second["background_estimate"].bfill().ffill()
    second["radiation_component"] = second["count_rate_smooth"] - second["background_estimate"]
    second["positive_component"] = second["radiation_component"].clip(lower=0.0)
    component_threshold = float(second["positive_component"].quantile(contribution_quantile))
    second["component_high"] = second["positive_component"] >= component_threshold

    if len(second) >= 2 and float(second["elapsed_s"].max()) > float(second["elapsed_s"].min()):
        slope = float(np.polyfit(second["elapsed_s"], second["background_estimate"], deg=1)[0])
    else:
        slope = 0.0

    summary = pd.DataFrame(
        [
            {
                "background_window_s": int(background_window_s),
                "background_quantile": float(background_quantile),
                "smooth_window_s": int(smooth_window_s),
                "component_quantile": float(contribution_quantile),
                "component_high_threshold": component_threshold,
                "background_mean": float(second["background_estimate"].mean()),
                "background_min": float(second["background_estimate"].min()),
                "background_max": float(second["background_estimate"].max()),
                "background_range": float(second["background_estimate"].max() - second["background_estimate"].min()),
                "background_slope_per_s": slope,
                "component_mean": float(second["radiation_component"].mean()),
                "positive_component_mean": float(second["positive_component"].mean()),
                "positive_component_max": float(second["positive_component"].max()),
                "component_high_seconds": int(second["component_high"].sum()),
                "component_high_ratio": float(second["component_high"].mean()),
            }
        ]
    )
    high_runs = _summarize_true_runs(
        second.rename(columns={"count_rate_smooth": "count_rate", "component_high": "component_high_flag"}).assign(
            row_number=np.arange(1, len(second) + 1)
        ),
        "component_high_flag",
    )
    return second, summary, high_runs


def summarize_long_csv(
    df: pd.DataFrame,
    high_count_quantile: float = CSV_HIGH_COUNT_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create file-level, minute-level, and high-run summaries."""
    df = df.copy()
    per_timestamp = df.groupby("timestamp").size()
    count = df["count_rate"]
    high_count_threshold = float(count.quantile(high_count_quantile))
    df["above_high_count_quantile"] = count >= high_count_threshold
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "timestamp_min": df["timestamp"].min(),
                "timestamp_max": df["timestamp"].max(),
                "duration_s_by_timestamp": float(
                    (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
                ),
                "unique_timestamps": int(df["timestamp"].nunique()),
                "rows_per_timestamp_min": int(per_timestamp.min()),
                "rows_per_timestamp_max": int(per_timestamp.max()),
                "rows_per_timestamp_mean": float(per_timestamp.mean()),
                "count_rate_min": float(count.min()),
                "count_rate_q01": float(count.quantile(0.01)),
                "count_rate_q05": float(count.quantile(0.05)),
                "count_rate_q25": float(count.quantile(0.25)),
                "count_rate_median": float(count.median()),
                "count_rate_mean": float(count.mean()),
                "count_rate_q75": float(count.quantile(0.75)),
                "count_rate_q95": float(count.quantile(0.95)),
                "count_rate_q99": float(count.quantile(0.99)),
                "count_rate_max": float(count.max()),
                "count_rate_std": float(count.std(ddof=1)),
                "high_count_quantile": float(high_count_quantile),
                "high_count_threshold": high_count_threshold,
                "above_high_count_samples": int(df["above_high_count_quantile"].sum()),
                "above_high_count_ratio": float(df["above_high_count_quantile"].mean()),
            }
        ]
    )

    minute_trend = (
        df.groupby("minute")
        .agg(
            n_samples=("count_rate", "size"),
            count_rate_mean=("count_rate", "mean"),
            count_rate_median=("count_rate", "median"),
            count_rate_std=("count_rate", "std"),
            count_rate_min=("count_rate", "min"),
            count_rate_max=("count_rate", "max"),
            above_high_count_samples=("above_high_count_quantile", "sum"),
            above_high_count_ratio=("above_high_count_quantile", "mean"),
        )
        .reset_index()
    )
    minute_trend["elapsed_min"] = (
        minute_trend["minute"] - minute_trend["minute"].min()
    ).dt.total_seconds() / 60.0

    high_runs = _summarize_true_runs(df, "above_high_count_quantile")
    return summary, minute_trend, high_runs


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_csv_minute_trend(minute_trend: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(minute_trend["elapsed_min"], minute_trend["count_rate_mean"], color="#2563eb", lw=1.5, label="minute mean")
    ax1.fill_between(
        minute_trend["elapsed_min"],
        minute_trend["count_rate_min"],
        minute_trend["count_rate_max"],
        color="#93c5fd",
        alpha=0.25,
        label="minute min-max",
    )
    ax2.bar(
        minute_trend["elapsed_min"],
        minute_trend["above_high_count_samples"],
        width=0.8,
        color="#f97316",
        alpha=0.35,
        label="top-quantile samples",
    )
    ax1.set_title("CSV Long Series Minute-Level Trend")
    ax1.set_xlabel("Elapsed time (min)")
    ax1.set_ylabel("Count rate")
    ax2.set_ylabel("High-count samples")
    ax1.grid(True, alpha=0.25)
    lines = ax1.get_lines() + [ax1.collections[0], ax2.patches[0]]
    labels = ["minute mean", "minute min-max", "top-quantile samples"]
    ax1.legend(lines, labels, fontsize=8, loc="upper right")
    _save(fig, path)


def plot_csv_high_count_events(df: pd.DataFrame, high_count_threshold: float, path: Path) -> None:
    df = df.copy()
    df["above_high_count_quantile"] = df["count_rate"] >= high_count_threshold
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    sample = df.iloc[:: max(1, len(df) // 12000)].copy()
    ax.plot(sample["elapsed_s_by_timestamp"] / 60.0, sample["count_rate"], color="#374151", lw=0.6, alpha=0.65)
    high = df[df["above_high_count_quantile"]]
    ax.scatter(
        high["elapsed_s_by_timestamp"] / 60.0,
        high["count_rate"],
        color="#dc2626",
        s=8,
        alpha=0.8,
        label="top-quantile count-rate samples",
    )
    ax.axhline(high_count_threshold, color="#f97316", ls="--", lw=1.2, label="high-count quantile")
    ax.set_title("CSV Long Series High-Count Events")
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Count rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_csv_background_components(second: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    elapsed_min = second["elapsed_s"] / 60.0
    ax.plot(elapsed_min, second["count_rate_smooth"], color="#111827", lw=1.0, label="second-level smoothed count rate")
    ax.plot(elapsed_min, second["background_estimate"], color="#2563eb", lw=1.6, label="estimated background")
    high = second[second["component_high"]]
    ax.scatter(
        high["elapsed_s"] / 60.0,
        high["count_rate_smooth"],
        color="#dc2626",
        s=10,
        alpha=0.75,
        label="high relative contribution",
    )
    ax.set_title("CSV Background and Relative Radiation Contribution")
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Count rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_csv_component_series(second: pd.DataFrame, component_threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    elapsed_min = second["elapsed_s"] / 60.0
    ax.plot(elapsed_min, second["radiation_component"], color="#0f766e", lw=1.0, label="relative component")
    ax.fill_between(
        elapsed_min,
        0,
        second["positive_component"],
        color="#99f6e4",
        alpha=0.35,
        label="positive component",
    )
    ax.axhline(component_threshold, color="#f97316", ls="--", lw=1.2, label="component high quantile")
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("CSV Relative Radiation Component Over Time")
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Count rate above estimated background")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def run_csv_analysis(
    csv_path: str | Path | None = None,
    output_dir: str | Path = NEW_OUTPUT_DIR,
) -> CsvAnalysisOutputs:
    csv_path = Path(csv_path) if csv_path is not None else NEW_DATA_DIR / "2025-11-28.csv"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_long_csv(csv_path)
    summary, minute_trend, high_runs = summarize_long_csv(df)
    second_components, component_summary, component_high_runs = build_second_level_components(df)
    high_count_threshold = float(summary["high_count_threshold"].iloc[0])

    normalized_path = output_dir / "csv_long_series_normalized.csv"
    summary_path = output_dir / "csv_long_series_summary.csv"
    minute_path = output_dir / "csv_long_series_minute_trend.csv"
    second_components_path = output_dir / "csv_second_level_components.csv"
    component_summary_path = output_dir / "csv_component_summary.csv"
    high_runs_path = output_dir / "csv_long_series_high_runs.csv"
    component_high_runs_path = output_dir / "csv_component_high_runs.csv"
    trend_figure = output_dir / "fig_csv_minute_trend.png"
    events_figure = output_dir / "fig_csv_high_count_events.png"
    background_figure = output_dir / "fig_csv_background_components.png"
    component_figure = output_dir / "fig_csv_relative_component.png"

    df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    minute_trend.to_csv(minute_path, index=False, encoding="utf-8-sig")
    second_components.to_csv(second_components_path, index=False, encoding="utf-8-sig")
    component_summary.to_csv(component_summary_path, index=False, encoding="utf-8-sig")
    high_runs.to_csv(high_runs_path, index=False, encoding="utf-8-sig")
    component_high_runs.to_csv(component_high_runs_path, index=False, encoding="utf-8-sig")
    plot_csv_minute_trend(minute_trend, trend_figure)
    plot_csv_high_count_events(df, high_count_threshold, events_figure)
    component_threshold = float(component_summary["component_high_threshold"].iloc[0])
    plot_csv_background_components(second_components, background_figure)
    plot_csv_component_series(second_components, component_threshold, component_figure)

    return CsvAnalysisOutputs(
        normalized_csv=normalized_path,
        summary_csv=summary_path,
        minute_trend_csv=minute_path,
        second_components_csv=second_components_path,
        component_summary_csv=component_summary_path,
        high_runs_csv=high_runs_path,
        figures=[trend_figure, events_figure, background_figure, component_figure],
    )
