from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

from .models import Thresholds


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["sample_index"], df["count_rate"], color="#6b7280", lw=0.8, alpha=0.55, label="Raw count rate")
    ax.plot(df["sample_index"], df["ma30_count_rate"], color="#0f766e", lw=1.8, label="30-point moving average")
    ax.set_title("Natural Radiation Count-Rate Time Series")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Count rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_distribution(df: pd.DataFrame, path: Path) -> None:
    values = df["count_rate"].to_numpy(dtype=float)
    xs = np.linspace(values.min(), values.max(), 300)
    kde = gaussian_kde(values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(values, bins=35, density=True, color="#9ca3af", edgecolor="white", alpha=0.85)
    axes[0].plot(xs, kde(xs), color="#b91c1c", lw=2)
    axes[0].set_title("Count-Rate Distribution and KDE")
    axes[0].set_xlabel("Count rate")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.2)
    axes[1].boxplot(values, vert=True, patch_artist=True, boxprops={"facecolor": "#bfdbfe"})
    axes[1].set_title("Count-Rate Boxplot")
    axes[1].set_ylabel("Count rate")
    axes[1].grid(True, axis="y", alpha=0.2)
    _save(fig, path)


def plot_rolling_stats(df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["sample_index"], df["ma30_count_rate"], color="#2563eb", lw=1.8, label="30-point moving average")
    ax2.plot(df["sample_index"], df["std30_count_rate"], color="#f97316", lw=1.2, alpha=0.9, label="30-point rolling std")
    ax1.set_title("Rolling Mean and Rolling Standard Deviation")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("30-point moving average", color="#2563eb")
    ax2.set_ylabel("30-point rolling std", color="#f97316")
    ax1.grid(True, alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    _save(fig, path)


def plot_changepoints(df: pd.DataFrame, boundaries: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["sample_index"], df["ma30_count_rate"], color="#0f766e", lw=1.7)
    for boundary in boundaries:
        row = df.iloc[boundary]
        ax.axvline(row["sample_index"], color="#dc2626", lw=1.2, ls="--", alpha=0.8)
    ax.set_title("Exploratory Change-Point Boundaries")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("30-point moving average")
    ax.grid(True, alpha=0.25)
    _save(fig, path)


def plot_gmm_states(df: pd.DataFrame, path: Path) -> None:
    states = [s for s in sorted(df["gmm_state"].dropna().unique())]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["sample_index"], df["ma30_count_rate"], color="#374151", lw=0.9, alpha=0.45)
    for i, state in enumerate(states):
        part = df[df["gmm_state"] == state]
        ax.scatter(part["sample_index"], part["ma30_count_rate"], s=10, color=cmap(i), label=state, alpha=0.85)
    ax.set_title("Exploratory GMM Count-State Partition")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("30-point moving average")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(4, max(1, len(states))), fontsize=8)
    _save(fig, path)


def plot_method_comparison(df: pd.DataFrame, boundaries: list[int], thresholds: Thresholds, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(df["sample_index"], df["ma30_count_rate"], color="#111827", lw=1.2, label="30-point moving average")
    ax.axhline(thresholds.warning_threshold, color="#f59e0b", lw=1.1, ls="--", label="Exploratory warning threshold")
    ax.axhline(thresholds.high_threshold, color="#dc2626", lw=1.1, ls="--", label="Exploratory high threshold")
    ax.axhline(thresholds.very_high_threshold, color="#7f1d1d", lw=1.1, ls="--", label="Exploratory very-high threshold")
    anomalous = df[df["is_high_count_anomaly"]]
    ax.scatter(anomalous["sample_index"], anomalous["ma30_count_rate"], color="#9333ea", s=18, label="High-count anomalies")
    for boundary in boundaries:
        ax.axvline(df.iloc[boundary]["sample_index"], color="#64748b", lw=0.8, ls=":", alpha=0.85)
    ax.set_title("Thresholds, Change Points, and High-Count Anomalies")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("30-point moving average")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_mechanism_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.62, 0.22, 0.24, "Coal-dominant flow", "Low and stable counts"),
        (0.39, 0.62, 0.22, 0.24, "Progressive parting", "Step-like count increase"),
        (0.73, 0.62, 0.22, 0.24, "High-count material", "High or rising counts"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        rect = plt.Rectangle((x, y), w, h, fc="#f8fafc", ec="#334155", lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=10)
    for x1, x2 in [(0.27, 0.39), (0.61, 0.73)]:
        ax.annotate("", xy=(x2, 0.74), xytext=(x1, 0.74), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#475569"})
    xs = np.linspace(0.07, 0.93, 240)
    y = np.piecewise(
        xs,
        [xs < 0.32, (xs >= 0.32) & (xs < 0.65), xs >= 0.65],
        [
            lambda z: 0.25 + 0.015 * np.sin(z * 50),
            lambda z: 0.28 + 0.15 * np.floor((z - 0.32) / 0.09) / 4 + 0.01 * np.sin(z * 60),
            lambda z: 0.42 + 0.35 * (z - 0.65),
        ],
    )
    ax.plot(xs, y, color="#0f766e", lw=2.2)
    ax.text(0.5, 0.08, "Conceptual diagram only: not generated from the measured sequence", ha="center", fontsize=12)
    _save(fig, path)


def plot_changepoint_bic(result_bic: dict[int, float], path: Path) -> None:
    xs = sorted(result_bic)
    ys = [result_bic[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, ys, marker="o", color="#2563eb")
    ax.set_title("Corrected BIC by Segment Count")
    ax.set_xlabel("Number of segments")
    ax.set_ylabel("BIC")
    ax.grid(True, alpha=0.25)
    _save(fig, path)


def plot_diagnostic_lines(df: pd.DataFrame, x_col: str, y_cols: list[str], path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col in y_cols:
        ax.plot(df[x_col], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_baseline_qq(values: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    stats.probplot(values.dropna(), dist="norm", plot=ax)
    ax.set_title("Baseline QQ Plot")
    ax.grid(True, alpha=0.25)
    _save(fig, path)


def create_all_plots(df: pd.DataFrame, boundaries: list[int], thresholds: Thresholds, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    paths = [
        output_dir / "fig01_time_series.png",
        output_dir / "fig02_distribution.png",
        output_dir / "fig03_rolling_stats.png",
        output_dir / "fig04_changepoints.png",
        output_dir / "fig05_gmm_states.png",
        output_dir / "fig06_method_comparison.png",
        output_dir / "concept01_radiation_mechanism_diagram.png",
    ]
    plot_time_series(df, paths[0])
    plot_distribution(df, paths[1])
    plot_rolling_stats(df, paths[2])
    plot_changepoints(df, boundaries, paths[3])
    plot_gmm_states(df, paths[4])
    plot_method_comparison(df, boundaries, thresholds, paths[5])
    plot_mechanism_diagram(paths[6])
    return paths

