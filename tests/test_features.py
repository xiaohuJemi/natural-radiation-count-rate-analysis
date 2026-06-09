from pathlib import Path

from src.data_loader import load_radiation_data
from src.features import build_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datas" / "data.xlsx"


def test_build_features_expected_missing_values():
    df = build_features(load_radiation_data(DATA_PATH))
    assert df["ma30_count_rate"].isna().sum() == 29
    assert df["ma60_count_rate"].isna().sum() == 59
    assert df["slope30_count_rate"].isna().sum() == 29
    assert "count_rate_percentile" in df.columns

