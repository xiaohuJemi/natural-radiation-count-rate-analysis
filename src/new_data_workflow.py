from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import (
    BACKGROUND_DELTA,
    BACKGROUND_STRATEGIES,
    ENGINEERING_THRESHOLD,
    NEW_DATA_DIR,
    NEW_OUTPUT_DIR,
    POISSON_CONFIDENCE_LEVELS,
    POISSON_EFFECTIVE_WINDOW_S,
    SAMPLE_INTERVAL_S,
    BackgroundStrategy,
)
from .data_loader import load_caving_condition_data
from .features import build_caving_time_features
from .models import (
    CandidateWindowConfig,
    add_dynamic_threshold,
    add_poisson_confidence_bands,
    detect_changepoints,
    detect_candidate_baseline_windows,
    estimate_baseline_from_candidate_windows,
    extract_threshold_events,
    separate_components,
    summarize_background_drift,
    summarize_threshold_crossing,
)
from .new_data_plots import (
    plot_adaptive_threshold_methods,
    plot_background_strategy_overlays,
    plot_candidate_baseline_windows,
    plot_caving_stage_segments,
    plot_high_count_events,
    plot_initial_baseline_sensitivity,
    plot_poisson_fluctuation,
    plot_stage_sensitivity,
    plot_strategy_summary,
    plot_threshold_sensitivity,
)
from .residual_workflow import build_residual_diagnostics, plot_residual_diagnostics


DEFAULT_STAGE_RULES = {
    "significant_high_ratio": 0.10,
    "minor_mean_delta": 12.0,
    "minor_max_delta": 18.0,
    "return_mean_delta": 10.0,
}

STAGE_RULE_PROFILES = {
    "relaxed": {
        "significant_high_ratio": 0.05,
        "minor_mean_delta": 10.0,
        "minor_max_delta": 15.0,
        "return_mean_delta": 12.0,
    },
    "default": DEFAULT_STAGE_RULES,
    "conservative": {
        "significant_high_ratio": 0.20,
        "minor_mean_delta": 15.0,
        "minor_max_delta": 22.0,
        "return_mean_delta": 8.0,
    },
}


def _process_group(group: pd.DataFrame) -> pd.DataFrame:
    group = separate_components(
        group,
        value_column="ma10_count_rate",
        method="rolling_quantile",
        sample_interval_s=SAMPLE_INTERVAL_S,
        window_seconds=10.0,
        quantile=0.2,
    )
    return add_dynamic_threshold(
        group,
        delta=BACKGROUND_DELTA,
        sample_interval_s=SAMPLE_INTERVAL_S,
        value_column="ma10_count_rate",
    )


def _process_group_with_strategy(group: pd.DataFrame, strategy: BackgroundStrategy) -> pd.DataFrame:
    group = separate_components(
        group,
        value_column="ma10_count_rate",
        method=strategy.method,
        initial_seconds=strategy.initial_seconds,
        sample_interval_s=SAMPLE_INTERVAL_S,
        window_seconds=strategy.window_seconds,
        quantile=strategy.quantile,
        alpha=strategy.alpha,
    )
    group = add_dynamic_threshold(
        group,
        delta=BACKGROUND_DELTA,
        sample_interval_s=SAMPLE_INTERVAL_S,
        value_column="ma10_count_rate",
    )
    group["background_strategy"] = strategy.name
    return group


def _summarize_group(group: pd.DataFrame) -> dict[str, object]:
    fixed = summarize_threshold_crossing(
        group,
        flag_column="above_85_4_ma10",
        value_column="ma10_count_rate",
        sample_interval_s=SAMPLE_INTERVAL_S,
    )
    dynamic = summarize_threshold_crossing(
        group,
        flag_column="above_dynamic_threshold",
        value_column="ma10_count_rate",
        sample_interval_s=SAMPLE_INTERVAL_S,
    )
    drift = summarize_background_drift(group, sample_interval_s=SAMPLE_INTERVAL_S)
    return {
        "source_file": group["source_file"].iloc[0],
        "condition": group["condition"].iloc[0],
        "condition_label": group["condition_label"].iloc[0],
        "n_samples": int(len(group)),
        "duration_s": float(len(group) * SAMPLE_INTERVAL_S),
        "count_rate_mean": float(group["count_rate"].mean()),
        "count_rate_max": float(group["count_rate"].max()),
        "ma10_max": float(group["ma10_count_rate"].max()),
        "radiation_component_max": float(group["radiation_component"].max()),
        "radiation_component_mean": float(group["radiation_component"].mean()),
        **{f"fixed_{key}": value for key, value in asdict(fixed).items()},
        **{f"dynamic_{key}": value for key, value in asdict(dynamic).items()},
        **drift,
    }


def _stage_label(
    segment: pd.DataFrame,
    background_ma10: float,
    significant_seen: bool,
    rules: dict[str, float] | None = None,
) -> str:
    rules = rules or DEFAULT_STAGE_RULES
    mean_delta = float(segment["ma10_count_rate"].mean() - background_ma10)
    max_delta = float(segment["ma10_count_rate"].max() - background_ma10)
    high_ratio = float((segment["ma10_count_rate"] >= ENGINEERING_THRESHOLD).mean())
    if high_ratio >= rules["significant_high_ratio"] or float(segment["ma10_count_rate"].max()) >= ENGINEERING_THRESHOLD:
        return "significant_high_count_stage"
    if mean_delta >= rules["minor_mean_delta"] or max_delta >= rules["minor_max_delta"]:
        return "minor_elevated_count_stage"
    if significant_seen and mean_delta <= rules["return_mean_delta"]:
        return "return_or_closing_stage"
    return "background_or_coal_dominant_stage"


def analyze_caving_stages(
    data_dir: str | Path = NEW_DATA_DIR,
    min_segment_size: int = 20,
    max_segments: int = 8,
    stage_rules: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment the three caving-condition curves and assign cautious stage labels."""
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    stage_rows = []
    changepoint_rows = []
    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        background_ma10 = float(group["ma10_count_rate"].iloc[: int(5 / SAMPLE_INTERVAL_S)].median())
        result = detect_changepoints(
            group["ma10_count_rate"].reset_index(drop=True),
            min_segment_size=min_segment_size,
            max_segments=max_segments,
        )
        cuts = [0] + result.boundaries + [len(group)]
        significant_seen = False
        boundary_times = [float(group["time_s"].iloc[idx]) for idx in result.boundaries]
        changepoint_rows.append(
            {
                "source_file": source_file,
                "condition": group["condition"].iloc[0],
                "condition_label": group["condition_label"].iloc[0],
                "chosen_segments": result.chosen_segments,
                "min_segment_size": result.min_segment_size,
                "max_segments": result.max_segments,
                "at_max_segments": result.at_max_segments,
                "bic_converged": result.bic_converged,
                "boundary_indices": ";".join(str(idx) for idx in result.boundaries),
                "boundary_times_s": ";".join(f"{time:.1f}" for time in boundary_times),
            }
        )
        for stage_id, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
            segment = group.iloc[start:end]
            label = _stage_label(segment, background_ma10, significant_seen=significant_seen, rules=stage_rules)
            if label == "significant_high_count_stage":
                significant_seen = True
            high_ratio = float((segment["ma10_count_rate"] >= ENGINEERING_THRESHOLD).mean())
            stage_rows.append(
                {
                    "source_file": source_file,
                    "condition": segment["condition"].iloc[0],
                    "condition_label": segment["condition_label"].iloc[0],
                    "stage_id": stage_id,
                    "stage_label": label,
                    "start_index": int(start),
                    "end_index_exclusive": int(end),
                    "start_time_s": float(segment["time_s"].iloc[0]),
                    "end_time_s": float(segment["time_s"].iloc[-1]),
                    "duration_s": float(len(segment) * SAMPLE_INTERVAL_S),
                    "n_samples": int(len(segment)),
                    "background_ma10": background_ma10,
                    "mean_count_rate": float(segment["count_rate"].mean()),
                    "mean_ma10_count_rate": float(segment["ma10_count_rate"].mean()),
                    "max_ma10_count_rate": float(segment["ma10_count_rate"].max()),
                    "mean_delta_from_background": float(segment["ma10_count_rate"].mean() - background_ma10),
                    "max_delta_from_background": float(segment["ma10_count_rate"].max() - background_ma10),
                    "threshold_crossing_ratio": high_ratio,
                    "threshold_crossing_samples": int((segment["ma10_count_rate"] >= ENGINEERING_THRESHOLD).sum()),
                    "changepoint_chosen_segments": result.chosen_segments,
                    "changepoint_at_max_segments": result.at_max_segments,
                    "changepoint_bic_converged": result.bic_converged,
                }
            )
    return pd.DataFrame(stage_rows), pd.DataFrame(changepoint_rows)


def _summarize_stage_sensitivity(stages: pd.DataFrame, context: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for (source_file, condition), group in stages.groupby(["source_file", "condition"], sort=False):
        significant = group[group["stage_label"] == "significant_high_count_stage"]
        rows.append(
            {
                **context,
                "source_file": source_file,
                "condition": condition,
                "condition_label": group["condition_label"].iloc[0],
                "n_stages": int(len(group)),
                "significant_stage_count": int(len(significant)),
                "significant_total_duration_s": float(significant["duration_s"].sum()) if not significant.empty else 0.0,
                "first_significant_start_time_s": (
                    float(significant["start_time_s"].iloc[0]) if not significant.empty else None
                ),
                "last_significant_end_time_s": float(significant["end_time_s"].iloc[-1]) if not significant.empty else None,
                "stage_label_sequence": ";".join(str(label) for label in group["stage_label"]),
            }
        )
    return rows


def analyze_stage_min_size_sensitivity(
    data_dir: str | Path = NEW_DATA_DIR,
    min_segment_sizes: tuple[int, ...] = (10, 20, 30),
    max_segments: int = 8,
) -> pd.DataFrame:
    """Test whether caving-stage interpretation changes with minimum segment size."""
    rows = []
    for min_size in min_segment_sizes:
        stages, changepoints = analyze_caving_stages(
            data_dir=data_dir,
            min_segment_size=min_size,
            max_segments=max_segments,
        )
        stage_rows = _summarize_stage_sensitivity(stages, {"min_segment_size": int(min_size)})
        for row in stage_rows:
            cp = changepoints[
                (changepoints["source_file"] == row["source_file"])
                & (changepoints["condition"] == row["condition"])
            ].iloc[0]
            row.update(
                {
                    "chosen_segments": int(cp["chosen_segments"]),
                    "at_max_segments": bool(cp["at_max_segments"]),
                    "bic_converged": bool(cp["bic_converged"]),
                    "boundary_times_s": cp["boundary_times_s"],
                }
            )
        rows.extend(stage_rows)
    return pd.DataFrame(rows)


def analyze_stage_rule_sensitivity(
    data_dir: str | Path = NEW_DATA_DIR,
    rule_profiles: dict[str, dict[str, float]] | None = None,
    min_segment_size: int = 20,
    max_segments: int = 8,
) -> pd.DataFrame:
    """Test whether stage labels change under relaxed/default/conservative rule thresholds."""
    profiles = rule_profiles or STAGE_RULE_PROFILES
    rows = []
    for profile_name, rules in profiles.items():
        stages, _ = analyze_caving_stages(
            data_dir=data_dir,
            min_segment_size=min_segment_size,
            max_segments=max_segments,
            stage_rules=rules,
        )
        rows.extend(
            _summarize_stage_sensitivity(
                stages,
                {
                    "rule_profile": profile_name,
                    "significant_high_ratio": rules["significant_high_ratio"],
                    "minor_mean_delta": rules["minor_mean_delta"],
                    "minor_max_delta": rules["minor_max_delta"],
                    "return_mean_delta": rules["return_mean_delta"],
                },
            )
        )
    return pd.DataFrame(rows)


def analyze_initial_baseline_sensitivity(
    data_dir: str | Path = NEW_DATA_DIR,
    baseline_windows_s: tuple[float, ...] = (3.0, 5.0, 10.0),
    threshold_deltas: tuple[float, ...] = (20.0, BACKGROUND_DELTA, 30.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify initial-background uncertainty and threshold sensitivity."""
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    baseline_rows = []
    threshold_rows = []

    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        for window_s in baseline_windows_s:
            n_initial = max(1, int(round(window_s / SAMPLE_INTERVAL_S)))
            segment = group.iloc[:n_initial]
            q25 = float(segment["count_rate"].quantile(0.25))
            q75 = float(segment["count_rate"].quantile(0.75))
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            outliers = (segment["count_rate"] < lower) | (segment["count_rate"] > upper)
            background_ma10 = float(segment["ma10_count_rate"].median())
            baseline_rows.append(
                {
                    "source_file": source_file,
                    "condition": group["condition"].iloc[0],
                    "condition_label": group["condition_label"].iloc[0],
                    "baseline_window_s": float(window_s),
                    "n_samples": int(len(segment)),
                    "raw_mean": float(segment["count_rate"].mean()),
                    "raw_median": float(segment["count_rate"].median()),
                    "raw_std": float(segment["count_rate"].std(ddof=1)),
                    "raw_q25": q25,
                    "raw_q75": q75,
                    "raw_iqr": iqr,
                    "raw_min": float(segment["count_rate"].min()),
                    "raw_max": float(segment["count_rate"].max()),
                    "raw_iqr_outlier_count": int(outliers.sum()),
                    "background_ma10_median": background_ma10,
                }
            )
            for delta in threshold_deltas:
                delta_value = round(float(delta), 4)
                threshold = background_ma10 + delta_value
                working = group.copy()
                working["above_sensitivity_threshold"] = working["ma10_count_rate"] >= threshold
                crossing = summarize_threshold_crossing(
                    working,
                    flag_column="above_sensitivity_threshold",
                    value_column="ma10_count_rate",
                    sample_interval_s=SAMPLE_INTERVAL_S,
                )
                threshold_rows.append(
                    {
                        "source_file": source_file,
                        "condition": group["condition"].iloc[0],
                        "condition_label": group["condition_label"].iloc[0],
                        "baseline_window_s": float(window_s),
                        "background_ma10_median": background_ma10,
                        "threshold_delta": delta_value,
                        "threshold_value": threshold,
                        **asdict(crossing),
                    }
                )

    return pd.DataFrame(baseline_rows), pd.DataFrame(threshold_rows)


def analyze_poisson_fluctuation(
    data_dir: str | Path = NEW_DATA_DIR,
    effective_window_s: float = POISSON_EFFECTIVE_WINDOW_S,
    confidence_levels: tuple[float, ...] = POISSON_CONFIDENCE_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate high count-rate contribution from expected Poisson background fluctuation."""
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    processed = features.groupby("source_file", group_keys=False, sort=False).apply(
        lambda group: add_poisson_confidence_bands(
            separate_components(
                group,
                value_column="ma10_count_rate",
                method="initial",
                initial_seconds=5.0,
                sample_interval_s=SAMPLE_INTERVAL_S,
            ),
            value_column="ma10_count_rate",
            effective_window_s=effective_window_s,
            confidence_levels=confidence_levels,
        )
    )
    summary_rows = []
    for _, group in processed.groupby("source_file", sort=False):
        for level in confidence_levels:
            suffix = f"{int(round(level * 100))}"
            flag_column = f"above_poisson_{suffix}"
            excess_column = f"poisson_excess_{suffix}"
            upper_column = f"poisson_upper_{suffix}"
            crossing = summarize_threshold_crossing(
                group,
                flag_column=flag_column,
                value_column="ma10_count_rate",
                sample_interval_s=SAMPLE_INTERVAL_S,
            )
            positive_excess = group.loc[group[excess_column] > 0, excess_column]
            summary_rows.append(
                {
                    "source_file": group["source_file"].iloc[0],
                    "condition": group["condition"].iloc[0],
                    "condition_label": group["condition_label"].iloc[0],
                    "confidence_level": float(level),
                    "effective_window_s": float(effective_window_s),
                    "background_mean": float(group["background_estimate"].mean()),
                    "poisson_sigma_rate_mean": float(group["poisson_sigma_rate"].mean()),
                    "poisson_upper_mean": float(group[upper_column].mean()),
                    "max_ma10_count_rate": float(group["ma10_count_rate"].max()),
                    "max_excess_over_poisson": float(group[excess_column].max()),
                    "mean_positive_excess_over_poisson": (
                        float(positive_excess.mean()) if not positive_excess.empty else 0.0
                    ),
                    "min_poisson_upper_tail_p": float(group["poisson_upper_tail_p"].min()),
                    **asdict(crossing),
                }
            )
    return processed.reset_index(drop=True), pd.DataFrame(summary_rows)


def analyze_candidate_baseline_windows(
    data_dir: str | Path = NEW_DATA_DIR,
    config: CandidateWindowConfig | None = None,
) -> pd.DataFrame:
    """Identify stable low-count candidate windows using only count-rate time series."""
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    rows = []
    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        windows = detect_candidate_baseline_windows(
            group,
            value_column="ma10_count_rate",
            time_column="time_s",
            sample_interval_s=SAMPLE_INTERVAL_S,
            config=config,
        )
        if windows.empty:
            continue
        windows.insert(0, "source_file", source_file)
        windows.insert(1, "condition", group["condition"].iloc[0])
        windows.insert(2, "condition_label", group["condition_label"].iloc[0])
        rows.append(windows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def analyze_adaptive_threshold_methods(
    data_dir: str | Path = NEW_DATA_DIR,
    stat_k: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed, static-background, and candidate-background thresholds."""
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    candidate_windows = analyze_candidate_baseline_windows(data_dir=data_dir)
    component_frames = []
    summary_rows = []
    method_columns = {
        "fixed_absolute_85_4": "fixed_absolute_threshold",
        "static_initial_delta": "static_initial_delta_threshold",
        "candidate_stat_3sigma": "candidate_stat_threshold",
        "candidate_poisson_99": "candidate_poisson_99_threshold",
        "candidate_reference_delta": "candidate_reference_delta_threshold",
    }

    for source_file, group in features.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        window_group = candidate_windows[candidate_windows["source_file"] == source_file]
        estimate = estimate_baseline_from_candidate_windows(
            group,
            window_group,
            value_column="ma10_count_rate",
            sample_interval_s=SAMPLE_INTERVAL_S,
        )
        n_initial = max(1, int(round(5.0 / SAMPLE_INTERVAL_S)))
        initial_background = float(group["ma10_count_rate"].iloc[:n_initial].median())

        working = group.copy()
        working["candidate_background_estimate"] = estimate.background
        working["candidate_background_sigma"] = estimate.sigma
        working["candidate_background_source"] = estimate.source
        working["candidate_radiation_component"] = working["ma10_count_rate"] - estimate.background
        working["fixed_absolute_threshold"] = ENGINEERING_THRESHOLD
        working["static_initial_delta_threshold"] = initial_background + BACKGROUND_DELTA
        working["candidate_stat_threshold"] = estimate.background + stat_k * estimate.sigma
        working["candidate_reference_delta_threshold"] = estimate.background + BACKGROUND_DELTA
        working = add_poisson_confidence_bands(
            working,
            background_column="candidate_background_estimate",
            value_column="ma10_count_rate",
            effective_window_s=POISSON_EFFECTIVE_WINDOW_S,
            confidence_levels=(0.99,),
        )
        working["candidate_poisson_99_threshold"] = working["poisson_upper_99"]

        for method, threshold_column in method_columns.items():
            flag_column = f"above_{method}"
            excess_column = f"excess_{method}"
            working[flag_column] = working["ma10_count_rate"] >= working[threshold_column]
            working[excess_column] = (working["ma10_count_rate"] - working[threshold_column]).clip(lower=0.0)
            crossing = summarize_threshold_crossing(
                working,
                flag_column=flag_column,
                value_column="ma10_count_rate",
                sample_interval_s=SAMPLE_INTERVAL_S,
            )
            positive_excess = working.loc[working[excess_column] > 0, excess_column]
            summary_rows.append(
                {
                    "source_file": source_file,
                    "condition": group["condition"].iloc[0],
                    "condition_label": group["condition_label"].iloc[0],
                    "method": method,
                    "threshold_column": threshold_column,
                    "candidate_background": estimate.background,
                    "candidate_sigma": estimate.sigma,
                    "candidate_window_count": estimate.n_windows,
                    "candidate_window_samples": estimate.n_samples,
                    "candidate_background_source": estimate.source,
                    "initial_background": initial_background,
                    "threshold_mean": float(working[threshold_column].mean()),
                    "threshold_min": float(working[threshold_column].min()),
                    "threshold_max": float(working[threshold_column].max()),
                    "total_excess_area": float(working[excess_column].sum() * SAMPLE_INTERVAL_S),
                    "max_excess": float(working[excess_column].max()),
                    "mean_positive_excess": float(positive_excess.mean()) if not positive_excess.empty else 0.0,
                    **asdict(crossing),
                }
            )
        component_frames.append(working)
    return pd.concat(component_frames, ignore_index=True), pd.DataFrame(summary_rows)


def analyze_high_count_events(
    data_dir: str | Path = NEW_DATA_DIR,
    min_duration_s: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract event-level features from adaptive threshold crossings."""
    components, _ = analyze_adaptive_threshold_methods(data_dir=data_dir)
    methods = {
        "fixed_absolute_85_4": "fixed_absolute_threshold",
        "static_initial_delta": "static_initial_delta_threshold",
        "candidate_stat_3sigma": "candidate_stat_threshold",
        "candidate_poisson_99": "candidate_poisson_99_threshold",
        "candidate_reference_delta": "candidate_reference_delta_threshold",
    }
    event_frames = []
    for source_file, group in components.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)
        for method, threshold_column in methods.items():
            flag_column = f"above_{method}"
            excess_column = f"excess_{method}"
            events = extract_threshold_events(
                group,
                flag_column=flag_column,
                value_column="ma10_count_rate",
                threshold_column=threshold_column,
                excess_column=excess_column,
                baseline_column="candidate_background_estimate",
                sample_interval_s=SAMPLE_INTERVAL_S,
                min_duration_s=min_duration_s,
            )
            if events.empty:
                continue
            events.insert(0, "source_file", source_file)
            events.insert(1, "condition", group["condition"].iloc[0])
            events.insert(2, "condition_label", group["condition_label"].iloc[0])
            events.insert(3, "method", method)
            events.insert(4, "threshold_column", threshold_column)
            event_frames.append(events)

    if event_frames:
        event_table = pd.concat(event_frames, ignore_index=True)
    else:
        event_table = pd.DataFrame(
            columns=[
                "source_file",
                "condition",
                "condition_label",
                "method",
                "threshold_column",
                "event_id",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "peak_value",
                "max_excess",
                "excess_area",
            ]
        )

    summary_rows = []
    conditions = components[["source_file", "condition", "condition_label"]].drop_duplicates()
    for _, condition_row in conditions.iterrows():
        for method in methods:
            subset = event_table[
                (event_table["source_file"] == condition_row["source_file"])
                & (event_table["method"] == method)
            ]
            summary_rows.append(
                {
                    "source_file": condition_row["source_file"],
                    "condition": condition_row["condition"],
                    "condition_label": condition_row["condition_label"],
                    "method": method,
                    "event_count": int(len(subset)),
                    "total_event_duration_s": float(subset["duration_s"].sum()) if not subset.empty else 0.0,
                    "max_event_duration_s": float(subset["duration_s"].max()) if not subset.empty else 0.0,
                    "total_excess_area": float(subset["excess_area"].sum()) if not subset.empty else 0.0,
                    "max_peak_value": float(subset["peak_value"].max()) if not subset.empty else 0.0,
                    "max_relative_peak_from_background": (
                        float(subset["relative_peak_from_background"].max()) if not subset.empty else 0.0
                    ),
                    "first_event_start_time_s": float(subset["start_time_s"].min()) if not subset.empty else None,
                    "last_event_end_time_s": float(subset["end_time_s"].max()) if not subset.empty else None,
                }
            )
    return event_table, pd.DataFrame(summary_rows)


def analyze_residual_diagnostics(
    data_dir: str | Path = NEW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose residual count-rate structure after candidate-background removal."""
    components, _ = analyze_adaptive_threshold_methods(data_dir=data_dir)
    return build_residual_diagnostics(components, sample_interval_s=SAMPLE_INTERVAL_S)


def analyze_new_data(
    data_dir: str | Path = NEW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    processed = features.groupby("source_file", group_keys=False, sort=False).apply(_process_group)
    summary = pd.DataFrame([_summarize_group(group) for _, group in processed.groupby("source_file", sort=False)])
    return processed, summary


def analyze_background_strategies(
    data_dir: str | Path = NEW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_caving_condition_data(data_dir, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    processed_frames = []
    summary_rows = []
    for strategy in BACKGROUND_STRATEGIES:
        strategy_processed = features.groupby("source_file", group_keys=False, sort=False).apply(
            lambda group: _process_group_with_strategy(group, strategy)
        )
        processed_frames.append(strategy_processed)
        for _, group in strategy_processed.groupby("source_file", sort=False):
            row = _summarize_group(group)
            row["background_strategy"] = strategy.name
            summary_rows.append(row)
    return pd.concat(processed_frames, ignore_index=True), pd.DataFrame(summary_rows)


def run_new_data_analysis(output_dir: str | Path = NEW_OUTPUT_DIR) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed, summary = analyze_new_data()
    strategy_components, strategy_summary = analyze_background_strategies()
    raw = load_caving_condition_data(NEW_DATA_DIR, sample_interval_s=SAMPLE_INTERVAL_S)
    features = build_caving_time_features(raw, engineering_threshold=ENGINEERING_THRESHOLD)
    stage_segments, changepoint_summary = analyze_caving_stages()
    min_size_sensitivity = analyze_stage_min_size_sensitivity()
    rule_sensitivity = analyze_stage_rule_sensitivity()
    baseline_summary, threshold_sensitivity = analyze_initial_baseline_sensitivity()
    poisson_components, poisson_summary = analyze_poisson_fluctuation()
    candidate_windows = analyze_candidate_baseline_windows()
    adaptive_components, adaptive_summary = analyze_adaptive_threshold_methods()
    high_count_events, high_count_event_summary = analyze_high_count_events()
    residuals, residual_summary = analyze_residual_diagnostics()
    paths = [
        output_dir / "caving_time_series_components.csv",
        output_dir / "caving_condition_summary.csv",
        output_dir / "background_strategy_components.csv",
        output_dir / "background_strategy_summary.csv",
        output_dir / "caving_stage_segments.csv",
        output_dir / "caving_changepoint_summary.csv",
        output_dir / "caving_stage_min_size_sensitivity.csv",
        output_dir / "caving_stage_rule_sensitivity.csv",
        output_dir / "caving_initial_baseline_summary.csv",
        output_dir / "caving_threshold_sensitivity.csv",
        output_dir / "caving_poisson_components.csv",
        output_dir / "caving_poisson_summary.csv",
        output_dir / "candidate_baseline_windows.csv",
        output_dir / "adaptive_threshold_components.csv",
        output_dir / "adaptive_threshold_summary.csv",
        output_dir / "high_count_events.csv",
        output_dir / "high_count_event_summary.csv",
        output_dir / "residual_diagnostics_components.csv",
        output_dir / "residual_diagnostics_summary.csv",
    ]
    processed.to_csv(paths[0], index=False, encoding="utf-8-sig")
    summary.to_csv(paths[1], index=False, encoding="utf-8-sig")
    strategy_components.to_csv(paths[2], index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(paths[3], index=False, encoding="utf-8-sig")
    stage_segments.to_csv(paths[4], index=False, encoding="utf-8-sig")
    changepoint_summary.to_csv(paths[5], index=False, encoding="utf-8-sig")
    min_size_sensitivity.to_csv(paths[6], index=False, encoding="utf-8-sig")
    rule_sensitivity.to_csv(paths[7], index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(paths[8], index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(paths[9], index=False, encoding="utf-8-sig")
    poisson_components.to_csv(paths[10], index=False, encoding="utf-8-sig")
    poisson_summary.to_csv(paths[11], index=False, encoding="utf-8-sig")
    candidate_windows.to_csv(paths[12], index=False, encoding="utf-8-sig")
    adaptive_components.to_csv(paths[13], index=False, encoding="utf-8-sig")
    adaptive_summary.to_csv(paths[14], index=False, encoding="utf-8-sig")
    high_count_events.to_csv(paths[15], index=False, encoding="utf-8-sig")
    high_count_event_summary.to_csv(paths[16], index=False, encoding="utf-8-sig")
    residuals.to_csv(paths[17], index=False, encoding="utf-8-sig")
    residual_summary.to_csv(paths[18], index=False, encoding="utf-8-sig")
    paths.extend(plot_background_strategy_overlays(strategy_components, output_dir))
    paths.extend(plot_strategy_summary(strategy_summary, output_dir))
    paths.extend(plot_caving_stage_segments(features, stage_segments, output_dir))
    paths.extend(plot_stage_sensitivity(min_size_sensitivity, rule_sensitivity, output_dir))
    paths.extend(plot_initial_baseline_sensitivity(baseline_summary, output_dir))
    paths.extend(plot_threshold_sensitivity(threshold_sensitivity, output_dir))
    paths.extend(plot_poisson_fluctuation(poisson_components, poisson_summary, output_dir))
    paths.extend(plot_candidate_baseline_windows(features, candidate_windows, output_dir))
    paths.extend(plot_adaptive_threshold_methods(adaptive_components, adaptive_summary, output_dir))
    paths.extend(plot_high_count_events(high_count_events, high_count_event_summary, output_dir))
    paths.extend(plot_residual_diagnostics(residuals, residual_summary, output_dir))
    return paths
