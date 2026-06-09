from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_data_loader import (
    test_load_radiation_data_shape_and_columns,
    test_validate_workbook_ma30_values,
)
from tests.test_features import test_build_features_expected_missing_values
from tests.test_models import (
    test_changepoint_result_reports_truncation_status,
    test_gmm_model_selection_outputs_candidate_range,
    test_threshold_sensitivity_flags_poor_discrimination,
)


def main() -> None:
    tests = [
        test_load_radiation_data_shape_and_columns,
        test_validate_workbook_ma30_values,
        test_build_features_expected_missing_values,
        test_changepoint_result_reports_truncation_status,
        test_gmm_model_selection_outputs_candidate_range,
        test_threshold_sensitivity_flags_poor_discrimination,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
