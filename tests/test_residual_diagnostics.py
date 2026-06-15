import numpy as np
import pandas as pd

from src.residual_diagnostics import (
    compute_residual_diagnostics,
    compute_residual_series,
    estimate_dominant_period,
)


def test_compute_residual_series_subtracts_background_and_positive_component():
    df = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2, 0.3],
            "ma10_count_rate": [60.0, 62.0, 58.0, 70.0],
            "candidate_background_estimate": [60.0, 60.0, 60.0, 60.0],
        }
    )

    residuals = compute_residual_series(
        df,
        value_column="ma10_count_rate",
        background_column="candidate_background_estimate",
    )

    assert residuals["residual_count_rate"].tolist() == [0.0, 2.0, -2.0, 10.0]
    assert residuals["positive_residual_count_rate"].tolist() == [0.0, 2.0, 0.0, 10.0]


def test_compute_residual_diagnostics_reports_positive_runs_and_energy():
    df = pd.DataFrame(
        {
            "time_s": [i * 0.1 for i in range(10)],
            "ma10_count_rate": [60, 61, 66, 68, 67, 60, 59, 65, 66, 60],
            "candidate_background_estimate": [60] * 10,
        }
    )

    diagnostics = compute_residual_diagnostics(
        df,
        value_column="ma10_count_rate",
        background_column="candidate_background_estimate",
        sample_interval_s=0.1,
        positive_threshold=5.0,
    )

    assert diagnostics.n_samples == 10
    assert diagnostics.positive_run_count == 2
    assert round(diagnostics.longest_positive_run_duration_s, 3) == 0.3
    assert diagnostics.positive_area > 0
    assert diagnostics.max_positive_residual == 8.0


def test_estimate_dominant_period_detects_synthetic_cycle():
    sample_interval_s = 0.1
    period_s = 2.0
    times = np.arange(0, 20, sample_interval_s)
    residual = pd.Series(np.sin(2 * np.pi * times / period_s))

    estimate = estimate_dominant_period(residual, sample_interval_s=sample_interval_s)

    assert estimate is not None
    assert abs(estimate["dominant_period_s"] - period_s) < 0.25
    assert estimate["dominant_power"] > 0
