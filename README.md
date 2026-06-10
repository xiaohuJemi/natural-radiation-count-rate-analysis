# Natural Gamma-Ray Count-Rate Time-Series Analysis

本项目用于分析综采/综放工作面自然伽马射线计数率时间序列。当前研究包含三类来源不同的数据：

- `datas/data.xlsx`：旧的无标签单变量计数率序列，用于探索性时序分析。
- `new_data/*.xlsx`：三个带工况文件名的放煤过程计数率序列，按正常放煤、少量放煤、过量放煤做弱标签分析。
- `new_data/2025-11-28.csv`：另一来源的长时间计数率序列，用于趋势、阈值越限和异常高值段分析。

这些数据来源不同，不能强行合并训练，也不能直接声称已经完成泛化煤矸识别。当前项目定位为：**计数率时间序列规律分析、本底影响拆分、工程阈值复核和后续煤矸识别方法准备**。

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
- 新 Excel：`ma10` 平滑、`85.4cps` 固定阈值、本底估计、相对辐射贡献拆分、动态阈值对照、阶段分割敏感性分析、泊松涨落置信区间分析。
- CSV：文件级统计、分钟级趋势、秒级平滑、本底估计、相对本底辐射贡献和本底漂移分析。

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
