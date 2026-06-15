from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .residual_diagnostics import compute_residual_diagnostics, compute_residual_series
from .models import summarize_threshold_crossing


def build_residual_diagnostics(
    components: pd.DataFrame,
    sample_interval_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build condition-level residual tables after candidate-background removal."""
    residual_frames = []
    summary_rows = []
    for source_file, group in components.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        residual_group = compute_residual_series(
            group,
            value_column="ma10_count_rate",
            background_column="candidate_background_estimate",
        )
        if "candidate_poisson_99_threshold" in residual_group.columns:
            residual_group["residual_poisson_99_boundary"] = (
                residual_group["candidate_poisson_99_threshold"]
                - residual_group["candidate_background_estimate"]
            )
            residual_group["residual_above_poisson_99"] = (
                residual_group["residual_count_rate"] >= residual_group["residual_poisson_99_boundary"]
            )
            residual_group["residual_excess_over_poisson_99"] = (
                residual_group["residual_count_rate"] - residual_group["residual_poisson_99_boundary"]
            ).clip(lower=0.0)
        else:
            residual_group["residual_poisson_99_boundary"] = pd.NA
            residual_group["residual_above_poisson_99"] = False
            residual_group["residual_excess_over_poisson_99"] = 0.0

        diagnostics = compute_residual_diagnostics(
            residual_group,
            value_column="ma10_count_rate",
            background_column="candidate_background_estimate",
            sample_interval_s=sample_interval_s,
            positive_threshold=0.0,
        )
        poisson_crossing = summarize_threshold_crossing(
            residual_group,
            flag_column="residual_above_poisson_99",
            value_column="residual_count_rate",
            sample_interval_s=sample_interval_s,
        )
        summary_rows.append(
            {
                "source_file": source_file,
                "condition": group["condition"].iloc[0],
                "condition_label": group["condition_label"].iloc[0],
                "candidate_background": float(group["candidate_background_estimate"].iloc[0]),
                **asdict(diagnostics),
                "poisson99_residual_excess_area": float(
                    residual_group["residual_excess_over_poisson_99"].sum() * sample_interval_s
                ),
                "poisson99_residual_max_excess": float(
                    residual_group["residual_excess_over_poisson_99"].max()
                ),
                **{
                    f"poisson99_{key}": value
                    for key, value in asdict(poisson_crossing).items()
                },
            }
        )
        residual_frames.append(residual_group)
    return pd.concat(residual_frames, ignore_index=True), pd.DataFrame(summary_rows)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_diagnostics(
    residuals: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    if residuals.empty:
        return paths

    conditions = list(residuals["condition"].drop_duplicates())
    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 3.2 * len(conditions)), sharex=False)
    if len(conditions) == 1:
        axes = [axes]
    for ax, condition in zip(axes, conditions):
        group = residuals[residuals["condition"] == condition]
        ax.plot(group["time_s"], group["residual_count_rate"], color="#2f6f9f", linewidth=1.2, label="Residual")
        ax.fill_between(
            group["time_s"],
            0,
            group["positive_residual_count_rate"],
            color="#d95f02",
            alpha=0.20,
            label="Positive residual",
        )
        if "residual_poisson_99_boundary" in group.columns:
            ax.plot(
                group["time_s"],
                group["residual_poisson_99_boundary"],
                color="#1b9e77",
                linewidth=1.0,
                linestyle="--",
                label="Poisson 99% boundary",
            )
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(f"Background-Removed Residual - {condition}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Residual count rate")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    path = output_dir / "fig_residual_time_series_by_condition.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in conditions:
        group = residuals[residuals["condition"] == condition]
        ax.hist(
            group["residual_count_rate"],
            bins=32,
            alpha=0.45,
            density=True,
            label=condition,
        )
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Residual Distribution After Candidate Background Removal")
    ax.set_xlabel("Residual count rate")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "fig_residual_distribution_by_condition.png"
    _save(fig, path)
    paths.append(path)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ordered = summary.sort_values("positive_area", ascending=False)
        ax.bar(ordered["condition"], ordered["positive_area"], color="#4c78a8")
        ax.set_title("Positive Residual Area by Condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Positive residual area")
        ax.grid(True, axis="y", alpha=0.25)
        path = output_dir / "fig_residual_positive_area_summary.png"
        _save(fig, path)
        paths.append(path)
    return paths
