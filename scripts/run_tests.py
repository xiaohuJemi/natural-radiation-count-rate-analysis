from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_data_loader import (
    test_load_radiation_data_shape_and_columns,
    test_validate_workbook_ma30_values,
)
from tests.test_csv_analysis import (
    test_build_second_level_components_separates_background_and_component,
    test_summarize_long_csv_reports_high_quantile_runs,
)
from tests.test_features import test_build_features_expected_missing_values
from tests.test_models import (
    test_candidate_baseline_windows_select_stable_low_count_segment,
    test_changepoint_result_reports_truncation_status,
    test_estimate_baseline_from_candidate_windows_uses_selected_windows,
    test_extract_threshold_events_groups_continuous_crossings,
    test_gmm_model_selection_outputs_candidate_range,
    test_threshold_sensitivity_flags_poor_discrimination,
)
from tests.test_new_caving_data import (
    test_adaptive_threshold_methods_compare_all_conditions_and_methods,
    test_background_strategy_comparison_outputs_all_conditions_and_methods,
    test_candidate_baseline_windows_outputs_candidates_for_each_condition,
    test_caving_features_and_component_separation,
    test_caving_stage_segmentation_outputs_stage_labels,
    test_high_count_events_extract_event_level_metrics,
    test_initial_baseline_sensitivity_outputs_windows_and_deltas,
    test_load_caving_condition_data_time_axis_and_labels,
    test_parse_caving_condition_ignores_support_number,
    test_poisson_fluctuation_outputs_confidence_bands_and_summary,
    test_stage_sensitivity_outputs_min_size_and_rule_profiles,
)


def main() -> None:
    tests = [
        test_load_radiation_data_shape_and_columns,
        test_validate_workbook_ma30_values,
        test_summarize_long_csv_reports_high_quantile_runs,
        test_build_second_level_components_separates_background_and_component,
        test_build_features_expected_missing_values,
        test_changepoint_result_reports_truncation_status,
        test_gmm_model_selection_outputs_candidate_range,
        test_threshold_sensitivity_flags_poor_discrimination,
        test_candidate_baseline_windows_select_stable_low_count_segment,
        test_estimate_baseline_from_candidate_windows_uses_selected_windows,
        test_extract_threshold_events_groups_continuous_crossings,
        test_parse_caving_condition_ignores_support_number,
        test_load_caving_condition_data_time_axis_and_labels,
        test_caving_features_and_component_separation,
        test_background_strategy_comparison_outputs_all_conditions_and_methods,
        test_caving_stage_segmentation_outputs_stage_labels,
        test_initial_baseline_sensitivity_outputs_windows_and_deltas,
        test_stage_sensitivity_outputs_min_size_and_rule_profiles,
        test_poisson_fluctuation_outputs_confidence_bands_and_summary,
        test_candidate_baseline_windows_outputs_candidates_for_each_condition,
        test_adaptive_threshold_methods_compare_all_conditions_and_methods,
        test_high_count_events_extract_event_level_metrics,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
