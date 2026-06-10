from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    return paths
