# Natural Gamma-Ray Count-Rate Time-Series Analysis

本项目用于分析综采/综放工作面自然伽马射线计数率时间序列。当前研究包含三类来源不同的数据：

- `datas/data.xlsx`：旧的无标签单变量计数率序列，用于探索性时序分析。
- `new_data/*.xlsx`：三个带工况文件名的放煤过程计数率序列，按正常放煤、少量放煤、过量放煤做弱标签分析。
- `new_data/2025-11-28.csv`：另一来源的长时间计数率序列，用于趋势、阈值越限和异常高值段分析。

这些数据来源不同，不能强行合并训练，也不能直接声称已经完成泛化煤矸识别。当前项目定位为：**自然伽玛计数率时间序列的局部本底建模、自适应阈值构建和有限数据条件下的信息挖掘方法学预研究**。

## Project motivation

Natural gamma-ray count-rate data should be treated as a physical system time series, not as independent tabular samples. In a top-coal caving face, the detector records a time-varying count process produced by the local radiation background, the moving coal-gangue flow on the scraper conveyor, detector geometry, shielding, and low-level radioactive counting fluctuation.

For the current data, the useful modeling view is:

```text
X(t) = B(t) + R(t) + e(t)
```

where `X(t)` is the observed or smoothed count rate, `B(t)` is the local background level of the current data segment, `R(t)` is the relative high-count contribution above the local background, and `e(t)` is statistical counting fluctuation. This makes the problem a time-series information-mining task: estimate a local baseline, decide whether later rises exceed expected fluctuation, and summarize sustained high-count events.

The project therefore focuses on local background windows, adaptive thresholds, Poisson fluctuation bands, change-point segmentation, and event-level features. It does not claim validated generalized coal-gangue classification under multi-mine or multi-support conditions.

## 数据边界

- 老师确认三类数据来源不同，没有绝对对应关系。
- 三个 Excel 文件按 `0.1s` 采样间隔处理，只研究时间序列计数率。
- `192/194/198` 等编号不作为建模特征。
- `85.4cps`、`60cps` 等参数来自韦明辉论文中的现场标定结果，在本文中作为工程基线，不作跨矿井通用阈值。
- 当前每种工况只有一条序列，不能用采样点数量替代独立样本数量。

## 本地数据

数据和 PDF 文献默认不提交到仓库。运行前请在本机放置：

```text
datas/data.xlsx
new_data/normal_caving.xlsx
new_data/minor_caving.xlsx
new_data/excessive_caving.xlsx
new_data/2025-11-28.csv
references/*.pdf
```

`.gitignore` 已默认忽略 `datas/data.xlsx`、`new_data/`、`references/*.pdf` 和逐点输出 CSV。

## 环境准备

推荐使用 Anaconda Python，或自行创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式

统一入口为 `main.py`：

```powershell
# 旧无标签数据分析
python main.py --analysis old

# 三个 Excel 工况序列分析
python main.py --analysis new

# CSV 长时间序列分析
python main.py --analysis csv

# 全部分析
python main.py --analysis all
```

兼容入口：

```powershell
python new_data_analysis.py
```

## 主要输出

旧数据输出：

- `outputs/processed_radiation_features.csv`
- `outputs/changepoint_segments.csv`
- `outputs/state_counts.csv`
- `outputs/analysis_summary.json`
- `outputs/fig*.png`
- `outputs/diagnostics/`

新 Excel 输出：

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

CSV 输出：

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

## 当前方法

- 旧数据：移动平均、滑动标准差、阈值诊断、变点检测、GMM 诊断、异常筛选。
- 新 Excel：`ma10` 平滑、候选局部本底窗口挖掘、固定阈值与自适应阈值对比、泊松涨落置信区间、变点阶段分割、高计数事件提取。
- CSV：文件级统计、分钟级趋势、秒级平滑、局部本底估计、相对本底辐射贡献和高计数段挖掘。

## Representative results

### Candidate local-background windows

![Candidate local-background windows](outputs/new_data/fig_candidate_baseline_windows_normal_caving.png)

The three short Excel caving-condition sequences contain stable low-count windows that can be used as candidate local-background references. Across the sequences, 8 non-overlapping representative windows were selected, and the candidate background estimates concentrate around 60-62 cps.

### Adaptive threshold comparison

![Adaptive threshold comparison](outputs/new_data/fig_adaptive_threshold_methods_normal_caving.png)

The `candidate B + 3 sigma` threshold is sensitive and better interpreted as an early-warning line. The Poisson 99% upper band is more suitable for separating random counting fluctuation from sustained high-count contribution. The `candidate B + reference delta` threshold is stricter and reflects the local-background adaptive-threshold idea.

### High-count event mining

![High-count event area summary](outputs/new_data/fig_high_count_event_area_summary.png)

Representative conclusions:

1. Global mean count rate is not sufficient to explain the three caving-condition sequences; duration, excess area, and event structure are more informative.
2. Under the Poisson 99% threshold, the excessive-caving sequence shows longer total high-count event duration and larger total excess area than the minor-caving sequence.
3. Different data sources must be analyzed under their own local backgrounds. The long CSV sequence has a much higher estimated local background range, about 195.16-243.63 cps, so a single absolute threshold should not be directly reused across sources.

## 验证

```powershell
python scripts/run_tests.py
python -m compileall src analysis.py main.py new_data_analysis.py tests
python main.py --analysis new
python main.py --analysis csv
```

## 正确结论边界

可以说：

> 当前计数率序列存在可描述的时间变化规律；基于本底估计、工程阈值和越限持续时间，可以对不同放煤工况进行解释性分析。

不能说：

> 当前方法已经在多工作面、多支架、多矿井条件下验证了煤矸智能识别能力。
