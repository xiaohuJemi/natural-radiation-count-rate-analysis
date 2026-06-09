from pathlib import Path

import pandas as pd

from src.data_loader import load_radiation_data
from src.diagnostics import threshold_sensitivity
from src.features import build_features
from src.models import detect_changepoints, fit_gmm_states, select_gmm_components


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
