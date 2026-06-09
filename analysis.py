from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_radiation_data, validate_workbook
from src.diagnostics import write_diagnostics
from src.features import build_features
from src.models import (
    assign_threshold_states,
    compute_domain_thresholds,
    detect_changepoints,
    fit_gmm_states,
    fit_isolation_forest,
    summarize_segments,
)
from src.plots import (
    configure_matplotlib,
    create_all_plots,
    plot_baseline_qq,
    plot_changepoint_bic,
    plot_diagnostic_lines,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datas" / "data.xlsx"
OUTPUT_DIR = ROOT / "outputs"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = load_radiation_data(DATA_PATH)
    validation = validate_workbook(DATA_PATH)
    features = build_features(raw)

    thresholds = compute_domain_thresholds(features, baseline_window=200)
    features["threshold_state"] = assign_threshold_states(features, thresholds)
    features["gmm_state"], silhouette, gmm_selection = fit_gmm_states(features, n_states=None)
    features["is_high_count_anomaly"] = fit_isolation_forest(features, contamination=0.06)

    changepoint_result = detect_changepoints(features["ma30_count_rate"], min_segment_size=40, max_segments=15)
    segments = summarize_segments(features, changepoint_result.boundaries)
    figures = create_all_plots(features, changepoint_result.boundaries, thresholds, OUTPUT_DIR)

    diagnostic_outputs = write_diagnostics(features, OUTPUT_DIR)
    configure_matplotlib()
    diagnostic_figures = [
        OUTPUT_DIR / "diagnostics" / "fig08_changepoint_bic.png",
        OUTPUT_DIR / "diagnostics" / "fig09_changepoint_max_segments_sensitivity.png",
        OUTPUT_DIR / "diagnostics" / "fig10_threshold_sensitivity.png",
        OUTPUT_DIR / "diagnostics" / "fig11_baseline_qq_plot.png",
        OUTPUT_DIR / "diagnostics" / "fig12_gmm_aic_bic.png",
        OUTPUT_DIR / "diagnostics" / "fig13_gmm_silhouette.png",
        OUTPUT_DIR / "diagnostics" / "fig14_isolation_forest_sensitivity.png",
    ]
    plot_changepoint_bic(changepoint_result.bic_by_segments, diagnostic_figures[0])
    plot_diagnostic_lines(
        _read_csv(diagnostic_outputs["changepoint_max_segments"]),
        "max_segments",
        ["chosen_segments"],
        diagnostic_figures[1],
        "Change-Point Sensitivity to max_segments",
        "max_segments",
    )
    plot_diagnostic_lines(
        _read_csv(diagnostic_outputs["threshold_sensitivity"]),
        "baseline_window",
        ["very_high_ratio"],
        diagnostic_figures[2],
        "Very-High Count Ratio by Baseline Window",
        "baseline_window",
    )
    baseline = features["ma30_count_rate"].dropna().iloc[: thresholds.baseline_window]
    plot_baseline_qq(baseline, diagnostic_figures[3])
    plot_diagnostic_lines(
        gmm_selection,
        "n_components",
        ["aic", "bic"],
        diagnostic_figures[4],
        "GMM AIC/BIC by Component Count",
        "n_components",
    )
    plot_diagnostic_lines(
        gmm_selection,
        "n_components",
        ["silhouette"],
        diagnostic_figures[5],
        "GMM Silhouette by Component Count",
        "n_components",
    )
    plot_diagnostic_lines(
        _read_csv(diagnostic_outputs["isolation_forest_sensitivity"]),
        "contamination",
        ["n_high_count_anomalies"],
        diagnostic_figures[6],
        "High-Count Anomalies by Isolation Forest Contamination",
        "contamination",
    )

    processed_path = OUTPUT_DIR / "processed_radiation_features.csv"
    segments_path = OUTPUT_DIR / "changepoint_segments.csv"
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    state_counts_path = OUTPUT_DIR / "state_counts.csv"

    features.to_csv(processed_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "threshold_state": features["threshold_state"].value_counts().index,
            "n_samples": features["threshold_state"].value_counts().values,
        }
    ).to_csv(state_counts_path, index=False, encoding="utf-8-sig")

    best_gmm = gmm_selection.loc[gmm_selection["bic"].idxmin()].to_dict()
    threshold_counts = features["threshold_state"].value_counts().to_dict()
    very_high_ratio = float(threshold_counts.get("very_high_count", 0) / len(features))
    summary = {
        "positioning": {
            "current_claim_level": "exploratory_time_series_analysis",
            "validated_coal_gangue_recognition": False,
            "reason": "single unlabeled count-rate sequence; no ground truth or sensor metadata",
        },
        "data": {
            "rows": int(len(features)),
            "sample_index_min": int(features["sample_index"].min()),
            "sample_index_max": int(features["sample_index"].max()),
            "count_rate_min": float(features["count_rate"].min()),
            "count_rate_max": float(features["count_rate"].max()),
            "count_rate_mean": float(features["count_rate"].mean()),
            "count_rate_std_sample": float(features["count_rate"].std(ddof=1)),
            "ma30_min": float(features["ma30_count_rate"].min()),
            "ma30_max": float(features["ma30_count_rate"].max()),
            "ma30_mean": float(features["ma30_count_rate"].mean()),
            "ma30_std_sample": float(features["ma30_count_rate"].std(ddof=1)),
            "sample_step_values": {
                str(k): int(v)
                for k, v in features["sample_index"].diff().dropna().value_counts().sort_index().items()
            },
        },
        "missing_values": {
            col: int(features[col].isna().sum())
            for col in ["ma30_count_rate", "ma60_count_rate", "std30_count_rate", "slope30_count_rate", "gmm_state"]
        },
        "workbook_validation": {
            "row_count": validation.row_count,
            "trailing_ma30_rows": validation.trailing_ma30_rows,
            "trailing_ma30_matches": validation.trailing_ma30_matches,
            "max_ma30_error": validation.max_ma30_error,
            "formula_rows_checked": validation.formula_rows_checked,
            "formula_rows_matching": validation.formula_rows_matching,
            "formula_check_available": validation.formula_check_available,
            "note_cells": validation.note_cells,
        },
        "thresholds": {
            **thresholds.__dict__,
            "state_counts": threshold_counts,
            "very_high_ratio": very_high_ratio,
            "discriminative_warning": very_high_ratio <= 0.30,
        },
        "gmm": {
            "selected_by_bic": best_gmm,
            "silhouette_score_for_selected": silhouette,
            "state_counts": features["gmm_state"].value_counts(dropna=False).to_dict(),
            "interpretation": "diagnostic comparison only; not validated coal-gangue labels",
        },
        "isolation_forest": {
            "contamination": 0.06,
            "high_count_anomaly_samples": int(features["is_high_count_anomaly"].sum()),
            "interpretation": "high-count outlier screening only; not roof-rock evidence",
        },
        "changepoints": {
            "chosen_segments": changepoint_result.chosen_segments,
            "max_segments": changepoint_result.max_segments,
            "min_segment_size": changepoint_result.min_segment_size,
            "at_max_segments": changepoint_result.at_max_segments,
            "bic_converged": changepoint_result.bic_converged,
            "boundaries": changepoint_result.boundaries,
            "bic_by_segments": changepoint_result.bic_by_segments,
            "interpretation": "exploratory count-rate segmentation only",
        },
        "outputs": {
            "processed_features": str(processed_path),
            "segments": str(segments_path),
            "state_counts": str(state_counts_path),
            "figures": [str(path) for path in figures],
            "diagnostic_tables": {name: str(path) for name, path in diagnostic_outputs.items()},
            "diagnostic_figures": [str(path) for path in diagnostic_figures],
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

