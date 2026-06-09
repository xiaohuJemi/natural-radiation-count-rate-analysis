from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class DataValidation:
    row_count: int
    trailing_ma30_rows: int
    trailing_ma30_matches: int
    max_ma30_error: float
    formula_rows_checked: int
    formula_rows_matching: int
    formula_check_available: bool
    note_cells: list[tuple[str, str]]


def load_radiation_data(path: str | Path) -> pd.DataFrame:
    """Load the teacher-provided count-rate workbook into normalized columns."""
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None, engine="openpyxl")
    df = raw.iloc[:, :3].copy()
    df.columns = ["sample_index", "count_rate", "ma30_count_rate_source"]
    df = df[pd.to_numeric(df["sample_index"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["count_rate"], errors="coerce").notna()]
    df["sample_index"] = df["sample_index"].astype(float).astype(int)
    df["count_rate"] = df["count_rate"].astype(float)
    df["ma30_count_rate_source"] = pd.to_numeric(
        df["ma30_count_rate_source"], errors="coerce"
    )
    df = df.reset_index(drop=True)
    df["row_number"] = np.arange(1, len(df) + 1)
    return df[["row_number", "sample_index", "count_rate", "ma30_count_rate_source"]]


def validate_workbook(path: str | Path) -> DataValidation:
    """Verify that workbook column C is a trailing 30-point moving average."""
    path = Path(path)
    wb_values = load_workbook(path, data_only=True, read_only=True)
    ws_values = wb_values["Sheet1"]
    wb_formula = load_workbook(path, data_only=False, read_only=True)
    ws_formula = wb_formula["Sheet1"]

    note_cells: list[tuple[str, str]] = []
    for row in ws_values.iter_rows():
        for cell in row:
            if cell.value is not None and cell.column > 3:
                note_cells.append((cell.coordinate, str(cell.value)))

    count_values: list[float] = []
    source_ma: list[tuple[int, float]] = []
    formula_rows_checked = 0
    formula_rows_matching = 0
    for row_idx in range(1, ws_values.max_row + 1):
        count = ws_values.cell(row_idx, 2).value
        ma_value = ws_values.cell(row_idx, 3).value
        formula = ws_formula.cell(row_idx, 3).value
        if isinstance(count, (int, float)):
            count_values.append(float(count))
        if isinstance(ma_value, (int, float)):
            source_ma.append((row_idx, float(ma_value)))
        if isinstance(formula, str) and formula.startswith("="):
            formula_rows_checked += 1
            expected_formula = f"=SUM(B{row_idx - 29}:B{row_idx})/30"
            expected_average = f"=AVERAGE(B{row_idx - 29}:B{row_idx})"
            normalized = formula.replace(" ", "").upper()
            if row_idx >= 30 and normalized in {expected_formula.upper(), expected_average.upper()}:
                formula_rows_matching += 1

    matches = 0
    max_error = 0.0
    for row_idx, ma_value in source_ma:
        if row_idx < 30:
            continue
        expected = float(np.mean(count_values[row_idx - 30 : row_idx]))
        error = abs(expected - ma_value)
        max_error = max(max_error, error)
        if error < 1e-9:
            matches += 1

    return DataValidation(
        row_count=len(count_values),
        trailing_ma30_rows=len(source_ma),
        trailing_ma30_matches=matches,
        max_ma30_error=max_error,
        formula_rows_checked=formula_rows_checked,
        formula_rows_matching=formula_rows_matching,
        formula_check_available=formula_rows_checked > 0,
        note_cells=note_cells,
    )
