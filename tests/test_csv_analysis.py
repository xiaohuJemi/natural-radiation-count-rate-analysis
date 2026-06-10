import pandas as pd

from src.csv_analysis import build_second_level_components, summarize_long_csv


def test_summarize_long_csv_reports_high_quantile_runs():
    df = pd.DataFrame(
        {
            "row_number": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2025-11-28 17:26:18",
                    "2025-11-28 17:26:18",
                    "2025-11-28 17:26:19",
                    "2025-11-28 17:26:20",
                    "2025-11-28 17:26:20",
                ]
            ),
            "elapsed_s_by_timestamp": [0.0, 0.0, 1.0, 2.0, 2.0],
            "minute": pd.to_datetime(
                [
                    "2025-11-28 17:26:00",
                    "2025-11-28 17:26:00",
                    "2025-11-28 17:26:00",
                    "2025-11-28 17:26:00",
                    "2025-11-28 17:26:00",
                ]
            ),
            "count_rate": [250, 301, 305, 310, 260],
        }
    )
    summary, minute_trend, high_runs = summarize_long_csv(df, high_count_quantile=0.75)
    assert summary["rows"].iloc[0] == 5
    assert summary["above_high_count_samples"].iloc[0] == 2
    assert summary["high_count_threshold"].iloc[0] == 305
    assert minute_trend["n_samples"].iloc[0] == 5
    assert minute_trend["above_high_count_samples"].iloc[0] == 2
    assert len(high_runs) == 1
    assert high_runs["n_samples"].tolist() == [2]


def test_build_second_level_components_separates_background_and_component():
    timestamps = pd.date_range("2025-11-28 17:26:18", periods=40, freq="s")
    values = [200] * 15 + [240, 245, 250, 245, 240] + [202] * 20
    df = pd.DataFrame(
        {
            "row_number": range(1, 41),
            "timestamp": timestamps,
            "elapsed_s_by_timestamp": range(40),
            "minute": timestamps.floor("1min"),
            "count_rate": values,
        }
    )
    components, summary, high_runs = build_second_level_components(
        df,
        background_window_s=11,
        background_quantile=0.2,
        smooth_window_s=3,
        contribution_quantile=0.9,
    )
    assert "background_estimate" in components.columns
    assert "radiation_component" in components.columns
    assert summary["positive_component_max"].iloc[0] > 20
    assert summary["component_high_seconds"].iloc[0] > 0
    assert len(high_runs) >= 1
