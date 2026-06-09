from pathlib import Path

from src.data_loader import load_radiation_data, validate_workbook


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datas" / "data.xlsx"


def test_load_radiation_data_shape_and_columns():
    df = load_radiation_data(DATA_PATH)
    assert len(df) == 1407
    assert list(df.columns) == [
        "row_number",
        "sample_index",
        "count_rate",
        "ma30_count_rate_source",
    ]
    assert df["sample_index"].iloc[0] == 1
    assert df["sample_index"].iloc[-1] == 2813


def test_validate_workbook_ma30_values():
    validation = validate_workbook(DATA_PATH)
    assert validation.trailing_ma30_rows == 1378
    assert validation.trailing_ma30_matches == 1378
    assert validation.max_ma30_error < 1e-9  # floating-point tolerance
    # Formula check depends on whether column C contains formulas or values
    assert isinstance(validation.formula_rows_checked, int)
    assert isinstance(validation.formula_rows_matching, int)
    assert isinstance(validation.formula_check_available, bool)

