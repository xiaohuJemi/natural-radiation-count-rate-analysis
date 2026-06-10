from pathlib import Path

import pandas as pd

from src.data_loader import load_caving_condition_data, parse_caving_condition
from src.features import build_caving_time_features
from src.models import add_dynamic_threshold, separate_components, summarize_threshold_crossing
from src.new_data_workflow import (
    analyze_background_strategies,
    analyze_caving_stages,
    analyze_initial_baseline_sensitivity,
    analyze_poisson_fluctuation,
    analyze_stage_min_size_sensitivity,
    analyze_stage_rule_sensitivity,
)


ROOT = Path(__file__).resolve().parents[1]
NEW_DATA_DIR = ROOT / "new_data"


def test_parse_caving_condition_ignores_support_number():
    condition, label = parse_caving_condition("normal_caving.xlsx")
    assert condition == "normal_caving"
    assert label == "正常放煤"


def test_load_caving_condition_data_time_axis_and_labels():
    df = load_caving_condition_data(NEW_DATA_DIR, sample_interval_s=0.1)
    assert len(df) == 1508
    assert set(df["condition"]) == {"normal_caving", "minor_caving", "excessive_caving"}
    first_group = df[df["source_file"] == "normal_caving.xlsx"]
    assert first_group["sample_index"].iloc[0] == 0
    assert first_group["time_s"].iloc[1] == 0.1
    assert first_group["time_s"].iloc[-1] == 52.5


def test_caving_features_and_component_separation():
    df = pd.DataFrame(
        {
            "time_s": [i * 0.1 for i in range(8)],
            "count_rate": [60, 61, 59, 60, 90, 92, 91, 60],
        }
    )
    features = build_caving_time_features(df, engineering_threshold=85.4)
    separated = separate_components(
        features,
        value_column="ma10_count_rate",
        method="initial",
        initial_seconds=0.4,
        sample_interval_s=0.1,
    )
    separated = add_dynamic_threshold(separated, delta=10.0, value_column="ma10_count_rate")
    summary = summarize_threshold_crossing(
        separated,
        flag_column="above_dynamic_threshold",
        value_column="ma10_count_rate",
        sample_interval_s=0.1,
    )
    assert "background_estimate" in separated.columns
    assert separated["radiation_component"].max() > 0
    assert summary.n_crossing_samples > 0


def test_background_strategy_comparison_outputs_all_conditions_and_methods():
    components, summary = analyze_background_strategies()
    assert len(summary) == 9
    assert set(summary["condition"]) == {"normal_caving", "minor_caving", "excessive_caving"}
    assert set(summary["background_strategy"]) == {
        "initial_5s",
        "rolling_q20_10s",
        "exponential_alpha_003",
    }
    assert len(components) == 1508 * 3
    assert "background_estimate" in components.columns
    assert "dynamic_longest_run_duration_s" in summary.columns


def test_caving_stage_segmentation_outputs_stage_labels():
    stages, changepoints = analyze_caving_stages()
    assert set(stages["condition"]) == {"normal_caving", "minor_caving", "excessive_caving"}
    assert len(changepoints) == 3
    assert "significant_high_count_stage" in set(stages["stage_label"])
    assert stages["duration_s"].min() > 0
    assert "boundary_times_s" in changepoints.columns


def test_initial_baseline_sensitivity_outputs_windows_and_deltas():
    baseline, threshold = analyze_initial_baseline_sensitivity()
    assert len(baseline) == 9
    assert len(threshold) == 27
    assert set(baseline["baseline_window_s"]) == {3.0, 5.0, 10.0}
    assert set(threshold["threshold_delta"]) == {20.0, 25.4, 30.0}
    assert baseline["raw_iqr_outlier_count"].min() >= 0
    assert threshold["longest_run_duration_s"].min() >= 0


def test_stage_sensitivity_outputs_min_size_and_rule_profiles():
    min_size = analyze_stage_min_size_sensitivity()
    rules = analyze_stage_rule_sensitivity()
    assert set(min_size["min_segment_size"]) == {10, 20, 30}
    assert set(rules["rule_profile"]) == {"relaxed", "default", "conservative"}
    assert set(min_size["condition"]) == {"normal_caving", "minor_caving", "excessive_caving"}
    assert set(rules["condition"]) == {"normal_caving", "minor_caving", "excessive_caving"}
    assert min_size["significant_total_duration_s"].min() >= 0
    assert rules["significant_stage_count"].min() >= 0


def test_poisson_fluctuation_outputs_confidence_bands_and_summary():
    components, summary = analyze_poisson_fluctuation()
    assert len(components) == 1508
    assert len(summary) == 6
    assert set(summary["confidence_level"]) == {0.95, 0.99}
    assert "poisson_upper_95" in components.columns
    assert "poisson_upper_99" in components.columns
    assert "above_poisson_99" in components.columns
    assert components["poisson_upper_tail_p"].between(0, 1).all()
    assert summary["longest_run_duration_s"].min() >= 0
    assert summary["max_excess_over_poisson"].max() > 0
