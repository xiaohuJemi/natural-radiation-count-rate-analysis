from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import BACKGROUND_STRATEGIES, ENGINEERING_THRESHOLD, NEW_DATA_DIR, SAMPLE_INTERVAL_S
from .data_loader import load_caving_condition_data
from .features import build_caving_time_features


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_background_strategy_overlays(strategy_components: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    colors = {
        "initial_5s": "#2563eb",
        "rolling_q20_10s": "#16a34a",
        "exponential_alpha_003": "#dc2626",
    }
    for _, group in strategy_components.groupby("source_file", sort=False):
        base = group[group["background_strategy"] == BACKGROUND_STRATEGIES[0].name]
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        ax.plot(base["time_s"], base["ma10_count_rate"], color="#111827", lw=1.2, label="ma10 count rate")
        ax.axhline(ENGINEERING_THRESHOLD, color="#f97316", ls="--", lw=1.2, label="fixed 85.4 cps")
        for strategy, part in group.groupby("background_strategy", sort=False):
            ax.plot(
                part["time_s"],
                part["background_estimate"],
                color=colors.get(strategy, "#64748b"),
                lw=1.2,
                label=f"background: {strategy}",
            )
        ax.set_title(f"Background Strategy Overlay - {base['condition'].iloc[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count rate (cps)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        path = output_dir / f"fig_background_overlay_{base['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_strategy_summary(strategy_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    conditions = list(strategy_summary["condition"].drop_duplicates())
    strategies = list(strategy_summary["background_strategy"].drop_duplicates())
    x = range(len(conditions))
    width = 0.24

    for metric, ylabel, file_name in [
        ("dynamic_longest_run_duration_s", "Longest dynamic crossing run (s)", "fig_dynamic_crossing_by_strategy.png"),
        (
            "radiation_component_max",
            "Max separated radiation contribution (cps)",
            "fig_radiation_component_by_strategy.png",
        ),
        ("background_range", "Estimated background range (cps)", "fig_background_range_by_strategy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for offset, strategy in enumerate(strategies):
            values = []
            for condition in conditions:
                row = strategy_summary[
                    (strategy_summary["condition"] == condition)
                    & (strategy_summary["background_strategy"] == strategy)
                ].iloc[0]
                values.append(row[metric])
            shifted = [value + (offset - 1) * width for value in x]
            ax.bar(shifted, values, width=width, label=strategy)
        ax.set_xticks(list(x))
        ax.set_xticklabels(conditions, rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / file_name
        _save(fig, path)
        paths.append(path)
    return paths


def plot_stage_sensitivity(
    min_size_sensitivity: pd.DataFrame,
    rule_sensitivity: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []

    conditions = list(min_size_sensitivity["condition"].drop_duplicates())
    min_sizes = list(min_size_sensitivity["min_segment_size"].drop_duplicates())
    x = range(len(conditions))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, min_size in enumerate(min_sizes):
        values = []
        for condition in conditions:
            row = min_size_sensitivity[
                (min_size_sensitivity["condition"] == condition)
                & (min_size_sensitivity["min_segment_size"] == min_size)
            ].iloc[0]
            values.append(row["significant_total_duration_s"])
        shifted = [value + (offset - 1) * width for value in x]
        ax.bar(shifted, values, width=width, label=f"{min_size} samples")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Significant high-count duration (s)")
    ax.set_title("Stage Sensitivity by Minimum Segment Size")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "fig_stage_min_size_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    profiles = list(rule_sensitivity["rule_profile"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, profile in enumerate(profiles):
        values = []
        for condition in conditions:
            row = rule_sensitivity[
                (rule_sensitivity["condition"] == condition)
                & (rule_sensitivity["rule_profile"] == profile)
            ].iloc[0]
            values.append(row["significant_total_duration_s"])
        shifted = [value + (offset - 1) * width for value in x]
        ax.bar(shifted, values, width=width, label=profile)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Significant high-count duration (s)")
    ax.set_title("Stage Label Sensitivity by Rule Profile")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "fig_stage_rule_sensitivity.png"
    _save(fig, path)
    paths.append(path)
    return paths


def plot_caving_stage_segments(
    features: pd.DataFrame,
    stages: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    colors = {
        "background_or_coal_dominant_stage": "#dbeafe",
        "minor_elevated_count_stage": "#fde68a",
        "significant_high_count_stage": "#fecaca",
        "return_or_closing_stage": "#dcfce7",
    }
    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        stage_group = stages[stages["source_file"] == source_file]
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        used_labels: set[str] = set()
        for _, stage in stage_group.iterrows():
            label = str(stage["stage_label"])
            legend_label = label if label not in used_labels else None
            used_labels.add(label)
            ax.axvspan(
                stage["start_time_s"],
                stage["end_time_s"],
                color=colors.get(label, "#e5e7eb"),
                alpha=0.45,
                label=legend_label,
            )
            if stage["stage_id"] > 1:
                ax.axvline(stage["start_time_s"], color="#64748b", lw=0.8, ls=":", alpha=0.9)
        background = float(stage_group["background_ma10"].iloc[0])
        ax.plot(group["time_s"], group["count_rate"], color="#94a3b8", lw=0.6, alpha=0.5, label="raw count rate")
        ax.plot(group["time_s"], group["ma10_count_rate"], color="#111827", lw=1.4, label="ma10 count rate")
        ax.axhline(background, color="#2563eb", lw=1.1, ls="-.", label="initial 5s background")
        ax.axhline(ENGINEERING_THRESHOLD, color="#dc2626", lw=1.1, ls="--", label="85.4 cps threshold")
        ax.set_title(f"Caving Stage Segmentation - {group['condition'].iloc[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count rate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        path = output_dir / f"fig_stage_segments_{group['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_initial_baseline_sensitivity(
    baseline_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    conditions = list(baseline_summary["condition"].drop_duplicates())
    windows = list(baseline_summary["baseline_window_s"].drop_duplicates())
    x = range(len(conditions))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, window_s in enumerate(windows):
        values = []
        for condition in conditions:
            row = baseline_summary[
                (baseline_summary["condition"] == condition)
                & (baseline_summary["baseline_window_s"] == window_s)
            ].iloc[0]
            values.append(row["background_ma10_median"])
        shifted = [value + (offset - 1) * width for value in x]
        ax.bar(shifted, values, width=width, label=f"{window_s:g}s")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Initial background estimate (ma10 cps)")
    ax.set_title("Initial Background Estimate by Window Length")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "fig_initial_background_window_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    five_s = baseline_summary[baseline_summary["baseline_window_s"] == 5.0]
    raw = load_caving_condition_data(NEW_DATA_DIR, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    ax.boxplot(
        [
            features[features["condition"] == condition]
            .reset_index(drop=True)
            .iloc[: int(5 / SAMPLE_INTERVAL_S)]["count_rate"]
            .to_numpy()
            for condition in conditions
        ],
        labels=conditions,
        patch_artist=True,
        boxprops={"facecolor": "#bfdbfe", "alpha": 0.75},
    )
    for idx, condition in enumerate(conditions, start=1):
        row = five_s[five_s["condition"] == condition].iloc[0]
        ax.text(idx, row["raw_max"] + 1.5, f"outliers={int(row['raw_iqr_outlier_count'])}", ha="center", fontsize=8)
    y_min = float(five_s["raw_min"].min()) - 2.0
    y_max = float(five_s["raw_max"].max()) + 8.0
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Raw count rate")
    ax.set_title("First 5s Raw Count-Rate Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    path = output_dir / "fig_initial_5s_baseline_boxplot.png"
    _save(fig, path)
    paths.append(path)
    return paths


def plot_threshold_sensitivity(
    threshold_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    conditions = list(threshold_summary["condition"].drop_duplicates())
    windows = list(threshold_summary["baseline_window_s"].drop_duplicates())
    deltas = list(threshold_summary["threshold_delta"].drop_duplicates())

    fig, axes = plt.subplots(1, len(deltas), figsize=(15, 4.8), sharey=True)
    if len(deltas) == 1:
        axes = [axes]
    x = range(len(conditions))
    width = 0.22
    for ax, delta in zip(axes, deltas):
        subset = threshold_summary[threshold_summary["threshold_delta"] == delta]
        for offset, window_s in enumerate(windows):
            values = []
            for condition in conditions:
                row = subset[
                    (subset["condition"] == condition)
                    & (subset["baseline_window_s"] == window_s)
                ].iloc[0]
                values.append(row["longest_run_duration_s"])
            shifted = [value + (offset - 1) * width for value in x]
            ax.bar(shifted, values, width=width, label=f"{window_s:g}s")
        ax.set_xticks(list(x))
        ax.set_xticklabels(conditions, rotation=20)
        ax.set_title(f"B + {delta:g}")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Longest threshold-crossing run (s)")
    axes[-1].legend(fontsize=8, title="Baseline window")
    fig.suptitle("Threshold Sensitivity by Initial Background Window")
    path = output_dir / "fig_threshold_sensitivity_by_baseline.png"
    _save(fig, path)
    paths.append(path)
    return paths


def plot_poisson_fluctuation(
    poisson_components: pd.DataFrame,
    poisson_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    for _, group in poisson_components.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        ax.plot(group["time_s"], group["ma10_count_rate"], color="#111827", lw=1.3, label="ma10 count rate")
        ax.plot(group["time_s"], group["background_estimate"], color="#2563eb", lw=1.0, label="initial 5s background")
        ax.plot(group["time_s"], group["poisson_upper_95"], color="#f59e0b", lw=1.0, ls="--", label="Poisson 95% upper")
        ax.plot(group["time_s"], group["poisson_upper_99"], color="#dc2626", lw=1.0, ls="--", label="Poisson 99% upper")
        high = group["above_poisson_99"].astype(bool)
        ax.fill_between(
            group["time_s"],
            group["ma10_count_rate"],
            group["poisson_upper_99"],
            where=high,
            color="#fecaca",
            alpha=0.55,
            label="above 99% fluctuation band",
        )
        ax.set_title(f"Poisson Fluctuation Band - {group['condition'].iloc[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count rate (cps)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        path = output_dir / f"fig_poisson_fluctuation_{group['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)

    conditions = list(poisson_summary["condition"].drop_duplicates())
    levels = list(poisson_summary["confidence_level"].drop_duplicates())
    x = range(len(conditions))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for offset, level in enumerate(levels):
        values = []
        for condition in conditions:
            row = poisson_summary[
                (poisson_summary["condition"] == condition)
                & (poisson_summary["confidence_level"] == level)
            ].iloc[0]
            values.append(row["longest_run_duration_s"])
        shifted = [value + (offset - 0.5) * width for value in x]
        ax.bar(shifted, values, width=width, label=f"{int(level * 100)}% upper")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Longest above-Poisson run (s)")
    ax.set_title("High-Count Runs Beyond Poisson Fluctuation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "fig_poisson_excess_summary.png"
    _save(fig, path)
    paths.append(path)
    return paths


def plot_candidate_baseline_windows(
    features: pd.DataFrame,
    windows: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        window_group = windows[windows["source_file"] == source_file]
        selected = window_group[window_group["selected_candidate"].astype(bool)]
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        ax.plot(group["time_s"], group["count_rate"], color="#94a3b8", lw=0.6, alpha=0.5, label="raw count rate")
        ax.plot(group["time_s"], group["ma10_count_rate"], color="#111827", lw=1.4, label="ma10 count rate")
        ax.axhline(
            float(window_group["low_mean_limit"].iloc[0]),
            color="#2563eb",
            lw=1.0,
            ls="--",
            label="low-count mean limit",
        )
        used_candidate_label = False
        for _, row in window_group[window_group["is_candidate_baseline_window"].astype(bool)].iterrows():
            ax.axvspan(
                row["start_time_s"],
                row["end_time_s"],
                color="#bbf7d0",
                alpha=0.18,
                label="candidate windows" if not used_candidate_label else None,
            )
            used_candidate_label = True
        for _, row in selected.iterrows():
            ax.axvspan(
                row["start_time_s"],
                row["end_time_s"],
                color="#22c55e",
                alpha=0.45,
                label="selected candidate" if _ == selected.index[0] else None,
            )
        ax.set_title(f"Candidate Local-Baseline Windows - {group['condition'].iloc[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count rate (cps)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        path = output_dir / f"fig_candidate_baseline_windows_{group['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_adaptive_threshold_methods(
    components: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    threshold_styles = [
        ("fixed_absolute_threshold", "fixed 85.4", "#64748b", "--"),
        ("static_initial_delta_threshold", "initial B + delta", "#f97316", "-."),
        ("candidate_stat_threshold", "candidate B + 3 sigma", "#7c3aed", ":"),
        ("candidate_poisson_99_threshold", "candidate Poisson 99%", "#dc2626", "--"),
        ("candidate_reference_delta_threshold", "candidate B + reference delta", "#16a34a", "-"),
    ]
    for _, group in components.groupby("source_file", sort=False):
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        ax.plot(group["time_s"], group["ma10_count_rate"], color="#111827", lw=1.35, label="ma10 count rate")
        ax.plot(
            group["time_s"],
            group["candidate_background_estimate"],
            color="#2563eb",
            lw=1.0,
            label="candidate local background",
        )
        for column, label, color, linestyle in threshold_styles:
            ax.plot(group["time_s"], group[column], color=color, lw=1.0, ls=linestyle, label=label)
        high = group["above_candidate_reference_delta"].astype(bool)
        ax.fill_between(
            group["time_s"],
            group["ma10_count_rate"],
            group["candidate_reference_delta_threshold"],
            where=high,
            color="#bbf7d0",
            alpha=0.35,
            label="above candidate B + reference delta",
        )
        ax.set_title(f"Adaptive Threshold Comparison - {group['condition'].iloc[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count rate (cps)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        path = output_dir / f"fig_adaptive_threshold_methods_{group['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)

    conditions = list(summary["condition"].drop_duplicates())
    methods = list(summary["method"].drop_duplicates())
    colors = ["#64748b", "#f97316", "#7c3aed", "#dc2626", "#16a34a"]
    x = list(range(len(conditions)))
    width = 0.15
    for metric, ylabel, file_name in [
        ("longest_run_duration_s", "Longest threshold-crossing run (s)", "fig_adaptive_longest_run_summary.png"),
        ("total_excess_area", "Total excess area (cps*s)", "fig_adaptive_excess_area_summary.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        for offset, method in enumerate(methods):
            values = []
            for condition in conditions:
                row = summary[(summary["condition"] == condition) & (summary["method"] == method)].iloc[0]
                values.append(row[metric])
            shifted = [value + (offset - 2) * width for value in x]
            ax.bar(shifted, values, width=width, label=method, color=colors[offset % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        path = output_dir / file_name
        _save(fig, path)
        paths.append(path)
    return paths


def plot_high_count_events(
    event_table: pd.DataFrame,
    event_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths = []
    focus_methods = ["candidate_poisson_99", "candidate_reference_delta"]
    focus_summary = event_summary[event_summary["method"].isin(focus_methods)]
    conditions = list(focus_summary["condition"].drop_duplicates())
    x = list(range(len(conditions)))
    width = 0.32

    for metric, ylabel, file_name in [
        ("event_count", "Event count", "fig_high_count_event_count_summary.png"),
        ("total_excess_area", "Total event excess area (cps*s)", "fig_high_count_event_area_summary.png"),
        ("max_event_duration_s", "Max event duration (s)", "fig_high_count_event_duration_summary.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        for offset, method in enumerate(focus_methods):
            values = []
            for condition in conditions:
                row = focus_summary[
                    (focus_summary["condition"] == condition) & (focus_summary["method"] == method)
                ].iloc[0]
                values.append(row[metric])
            shifted = [value + (offset - 0.5) * width for value in x]
            ax.bar(shifted, values, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        path = output_dir / file_name
        _save(fig, path)
        paths.append(path)

    if event_table.empty:
        return paths

    focus_events = event_table[event_table["method"].isin(focus_methods)]
    colors = {"candidate_poisson_99": "#dc2626", "candidate_reference_delta": "#16a34a"}
    for source_file, group in focus_events.groupby("source_file", sort=False):
        fig, ax = plt.subplots(figsize=(11.0, 3.8))
        y_positions = {method: idx for idx, method in enumerate(focus_methods)}
        for _, event in group.iterrows():
            y = y_positions[event["method"]]
            ax.barh(
                y,
                event["duration_s"],
                left=event["start_time_s"],
                height=0.32,
                color=colors.get(event["method"], "#64748b"),
                alpha=0.75,
            )
            ax.plot(event["peak_time_s"], y, marker="o", ms=4, color="#111827")
        ax.set_yticks([y_positions[method] for method in focus_methods])
        ax.set_yticklabels(focus_methods)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"High-Count Event Timeline - {group['condition'].iloc[0]}")
        ax.grid(True, axis="x", alpha=0.25)
        path = output_dir / f"fig_high_count_event_timeline_{group['condition'].iloc[0]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


