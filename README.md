# Natural Gamma-Ray Count-Rate Time-Series Analysis

This project analyzes natural gamma-ray count-rate time series from fully mechanized / top-coal caving faces. The study covers three categories of data from distinct sources:

- `datas/data.xlsx`: An older unlabeled univariate count-rate series, used for exploratory time-series analysis.
- `new_data/*.xlsx`: Three caving-process count-rate series labeled by working condition — normal caving, minor caving, and excessive caving — analyzed as a weakly-labeled problem.
- `new_data/2025-11-28.csv`: A long-duration count-rate series from another source, used for trend analysis, threshold-excursion detection, and high-count segment extraction.

These datasets originate from different sources. They cannot be merged into a single training set, nor do they yet constitute a validated generalized coal-gangue recognition system. This project is scoped as: **a methodological pre-study of local background modeling, adaptive threshold construction, and information mining under limited-data conditions for natural gamma count-rate time series.**

## Project motivation

Natural gamma-ray count-rate data should be treated as a physical system time series, not as independent tabular samples. In a top-coal caving face, the detector records a time-varying count process produced by the local radiation background, the moving coal-gangue flow on the scraper conveyor, detector geometry, shielding, and low-level radioactive counting fluctuation.

For the current data, the useful modeling view is:

```text
X(t) = B(t) + R(t) + e(t)
```

where `X(t)` is the observed or smoothed count rate, `B(t)` is the local background level of the current data segment, `R(t)` is the relative high-count contribution above the local background, and `e(t)` is statistical counting fluctuation. This makes the problem a time-series information-mining task: estimate a local baseline, decide whether later rises exceed expected fluctuation, and summarize sustained high-count events.

The project therefore focuses on local background windows, adaptive thresholds, Poisson fluctuation bands, change-point segmentation, and event-level features. It does not claim validated generalized coal-gangue classification under multi-mine or multi-support conditions.

## Data Boundaries

- The advisor confirmed that the three data categories come from different sources and are not directly comparable.
- The three Excel files are processed at a `0.1 s` sampling interval; only the time-series count rate is studied.
- Identifiers such as `192/194/198` are not used as modeling features.
- Parameters like `85.4 cps` and `60 cps` originate from on-site calibration results in Wei Minghui's thesis. They serve as engineering baselines in this study, not as universal thresholds transferable across mines.
- Currently there is only one sequence per working condition. The number of sampling points within a sequence cannot substitute for the number of independent sequences.

## Local Data Setup

Data files and PDF references are not committed to the repository. Before running, ensure the following files are present locally:

```text
datas/data.xlsx
new_data/normal_caving.xlsx
new_data/minor_caving.xlsx
new_data/excessive_caving.xlsx
new_data/2025-11-28.csv
references/*.pdf
```

`.gitignore` already excludes `datas/data.xlsx`, `new_data/`, `references/*.pdf`, and per-point output CSVs.

## Environment Setup

Anaconda Python is recommended, or create a virtual environment manually:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Analysis

The unified entry point is `main.py`:

```powershell
# Old unlabeled data analysis
python main.py --analysis old

# Three Excel caving-condition sequence analysis
python main.py --analysis new

# CSV long-series analysis
python main.py --analysis csv

# Run all analyses
python main.py --analysis all
```

Compatibility entry point (legacy):

```powershell
python new_data_analysis.py
```

## Main Outputs

Old data outputs:

- `outputs/processed_radiation_features.csv`
- `outputs/changepoint_segments.csv`
- `outputs/state_counts.csv`
- `outputs/analysis_summary.json`
- `outputs/fig*.png`
- `outputs/diagnostics/`

New Excel outputs:

- `outputs/new_data/caving_condition_summary.csv`
- `outputs/new_data/caving_time_series_components.csv`
- `outputs/new_data/background_strategy_summary.csv`
- `outputs/new_data/background_strategy_components.csv`
- `outputs/new_data/caving_stage_segments.csv`
- `outputs/new_data/caving_changepoint_summary.csv`
- `outputs/new_data/caving_stage_min_size_sensitivity.csv`
- `outputs/new_data/caving_stage_rule_sensitivity.csv`
- `outputs/new_data/caving_initial_baseline_summary.csv`
- `outputs/new_data/caving_threshold_sensitivity.csv`
- `outputs/new_data/caving_poisson_components.csv`
- `outputs/new_data/caving_poisson_summary.csv`
- `outputs/new_data/candidate_baseline_windows.csv`
- `outputs/new_data/adaptive_threshold_components.csv`
- `outputs/new_data/adaptive_threshold_summary.csv`
- `outputs/new_data/high_count_events.csv`
- `outputs/new_data/high_count_event_summary.csv`
- `outputs/new_data/fig_background_overlay_*.png`
- `outputs/new_data/fig_stage_segments_*.png`
- `outputs/new_data/fig_initial_background_window_sensitivity.png`
- `outputs/new_data/fig_initial_5s_baseline_boxplot.png`
- `outputs/new_data/fig_threshold_sensitivity_by_baseline.png`
- `outputs/new_data/fig_dynamic_crossing_by_strategy.png`
- `outputs/new_data/fig_radiation_component_by_strategy.png`
- `outputs/new_data/fig_background_range_by_strategy.png`
- `outputs/new_data/fig_stage_min_size_sensitivity.png`
- `outputs/new_data/fig_stage_rule_sensitivity.png`
- `outputs/new_data/fig_poisson_fluctuation_*.png`
- `outputs/new_data/fig_poisson_excess_summary.png`
- `outputs/new_data/fig_candidate_baseline_windows_*.png`
- `outputs/new_data/fig_adaptive_threshold_methods_*.png`
- `outputs/new_data/fig_adaptive_longest_run_summary.png`
- `outputs/new_data/fig_adaptive_excess_area_summary.png`
- `outputs/new_data/fig_high_count_event_*_summary.png`
- `outputs/new_data/fig_high_count_event_timeline_*.png`

CSV outputs:

- `outputs/new_data/csv_long_series_summary.csv`
- `outputs/new_data/csv_long_series_minute_trend.csv`
- `outputs/new_data/csv_second_level_components.csv`
- `outputs/new_data/csv_component_summary.csv`
- `outputs/new_data/csv_component_high_runs.csv`
- `outputs/new_data/csv_long_series_high_runs.csv`
- `outputs/new_data/csv_long_series_normalized.csv`
- `outputs/new_data/fig_csv_minute_trend.png`
- `outputs/new_data/fig_csv_high_count_events.png`
- `outputs/new_data/fig_csv_background_components.png`
- `outputs/new_data/fig_csv_relative_component.png`

## Current Methods

- Old data: moving average, rolling standard deviation, threshold diagnostics, change-point detection, GMM diagnostics, anomaly screening.
- New Excel: `ma10` smoothing, candidate local-background window mining, fixed vs. adaptive threshold comparison, Poisson fluctuation confidence bands, change-point stage segmentation, high-count event extraction.
- CSV: file-level statistics, minute-level trends, second-level smoothing, local background estimation, relative background radiation contribution, and high-count segment mining.

## Representative results

### Candidate local-background windows

![Candidate local-background windows](outputs/new_data/fig_candidate_baseline_windows_normal_caving.png)

The three short Excel caving-condition sequences contain stable low-count windows that can serve as candidate local-background references. Across the sequences, 8 non-overlapping representative windows were selected, and the candidate background estimates cluster around 60–62 cps.

### Adaptive threshold comparison

![Adaptive threshold comparison](outputs/new_data/fig_adaptive_threshold_methods_normal_caving.png)

The `candidate B + 3 sigma` threshold is sensitive and better interpreted as an early-warning line. The Poisson 99% upper band is more suitable for separating random counting fluctuation from sustained high-count contribution. The `candidate B + reference delta` threshold is stricter and reflects the local-background adaptive-threshold idea.

### High-count event mining

![High-count event area summary](outputs/new_data/fig_high_count_event_area_summary.png)

Representative conclusions:

1. Global mean count rate alone is insufficient to characterize the three caving-condition sequences; duration, excess area, and event structure carry more discriminating information.
2. Under the Poisson 99% threshold, the excessive-caving sequence shows longer total high-count event duration and larger total excess area than the minor-caving sequence.
3. Different data sources must be analyzed under their own local backgrounds. The long CSV sequence has a much higher estimated local background range, about 195.16-243.63 cps, so a single absolute threshold should not be directly reused across sources.

## Verification

```powershell
python scripts/run_tests.py
python -m compileall src analysis.py main.py new_data_analysis.py tests
python main.py --analysis new
python main.py --analysis csv
```

## Scope of Conclusions

Supported by this work:

> The count-rate sequences exhibit well-defined temporal variation patterns. Background estimation, engineering thresholds, and excursion duration together enable explanatory analysis that distinguishes different caving conditions.

NOT supported by this work:

> The method has been validated as a general intelligent coal-gangue recognition system across multiple faces, supports, or mines.
