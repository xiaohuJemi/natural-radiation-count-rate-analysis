from pathlib import Path

import pandas as pd

from src.data_loader import load_radiation_data
from src.diagnostics import threshold_sensitivity
from src.features import build_features
from src.models import (
    CandidateWindowConfig,
    detect_candidate_baseline_windows,
    detect_changepoints,
    estimate_baseline_from_candidate_windows,
    extract_threshold_events,
    fit_gmm_states,
    select_gmm_components,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datas" / "data.xlsx"


def _features():
    return build_features(load_radiation_data(DATA_PATH))


def test_changepoint_result_reports_truncation_status():
    series = pd.Series([1.0] * 40 + [4.0] * 40 + [2.0] * 40)
    result = detect_changepoints(series, min_segment_size=20, max_segments=4)
    assert result.max_segments == 4
    assert result.chosen_segments <= 4
    assert isinstance(result.at_max_segments, bool)
    assert all(isinstance(k, int) for k in result.bic_by_segments)


def test_gmm_model_selection_outputs_candidate_range():
    df = _features()
    selection = select_gmm_components(df, component_range=range(2, 5))
    assert selection["n_components"].tolist() == [2, 3, 4]
    assert selection["bic"].notna().all()
    states, silhouette, full_selection = fit_gmm_states(df, n_states=None)
    assert states.notna().sum() > 0
    assert full_selection["n_components"].min() == 2
    assert full_selection["n_components"].max() == 8
    assert isinstance(silhouette, float)


def test_threshold_sensitivity_flags_poor_discrimination():
    df = _features()
    table = threshold_sensitivity(df, baseline_windows=[100, 200])
    assert set(table["baseline_window"]) == {100, 200}
    assert "threshold_discriminative" in table.columns
    assert table["very_high_ratio"].between(0, 1).all()


def test_candidate_baseline_windows_select_stable_low_count_segment():
    df = pd.DataFrame(
        {
            "time_s": [i * 0.1 for i in range(120)],
            "ma10_count_rate": [60.0, 60.2, 59.8, 60.1, 60.0, 59.9] * 10
            + [75.0, 78.0, 82.0, 86.0, 90.0, 88.0] * 10,
        }
    )
    windows = detect_candidate_baseline_windows(
        df,
        value_column="ma10_count_rate",
        sample_interval_s=0.1,
        config=CandidateWindowConfig(window_seconds=3.0, min_duration_s=2.0, max_selected_windows=2),
    )
    selected = windows[windows["selected_candidate"]]
    assert not selected.empty
    assert selected["window_mean"].max() < 61.0
    assert selected["is_candidate_baseline_window"].all()


def test_estimate_baseline_from_candidate_windows_uses_selected_windows():
    df = pd.DataFrame(
        {
            "time_s": [i * 0.1 for i in range(80)],
            "ma10_count_rate": [60.0, 60.2, 59.8, 60.1] * 20,
        }
    )
    windows = detect_candidate_baseline_windows(
        df,
        value_column="ma10_count_rate",
        sample_interval_s=0.1,
        config=CandidateWindowConfig(window_seconds=2.0, min_duration_s=1.0, max_selected_windows=1),
    )
    estimate = estimate_baseline_from_candidate_windows(
        df,
        windows,
        value_column="ma10_count_rate",
        sample_interval_s=0.1,
    )
    assert estimate.source == "selected_candidate_windows"
    assert estimate.n_windows == 1
    assert 59.5 <= estimate.background <= 60.5
    assert estimate.sigma > 0


def test_extract_threshold_events_groups_continuous_crossings():
    df = pd.DataFrame(
        {
            "time_s": [i * 0.1 for i in range(10)],
            "ma10_count_rate": [60, 61, 70, 72, 71, 61, 80, 83, 60, 59],
            "threshold": [65] * 10,
            "baseline": [60] * 10,
        }
    )
    df["above"] = df["ma10_count_rate"] >= df["threshold"]
    df["excess"] = (df["ma10_count_rate"] - df["threshold"]).clip(lower=0)
    events = extract_threshold_events(
        df,
        flag_column="above",
        value_column="ma10_count_rate",
        threshold_column="threshold",
        excess_column="excess",
        baseline_column="baseline",
        sample_interval_s=0.1,
        min_duration_s=0.2,
    )
    assert len(events) == 2
    assert [round(value, 3) for value in events["duration_s"].tolist()] == [0.3, 0.2]
    assert events["peak_value"].tolist() == [72.0, 83.0]
    assert events["excess_area"].iloc[0] > events["excess_area"].iloc[1] / 2
