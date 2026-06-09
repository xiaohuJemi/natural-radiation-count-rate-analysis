# Natural Gamma-Ray Count-Rate Exploratory Time-Series Analysis

> **English abstract:** This project analyzes natural gamma-ray count-rate time series from a fully-mechanized top-coal caving face. It builds rolling statistical features, applies change-point detection (dynamic programming with corrected BIC), GMM-based state partitioning (with AIC/BIC model selection), and high-count anomaly screening. **Important:** The current dataset is a single unlabeled count-rate sequence — no ground truth, sensor metadata, or multi-face validation is available. All model outputs are therefore exploratory count-level partitions, not validated coal-gangue state labels. This project serves as a methodological pre-study and diagnostics framework for future labeled coal-gangue recognition research.

<br>

# 自然射线计数率探索性时序分析

本项目用于分析综采/综放工作面自然射线计数率数据。当前只有一组无标签计数率序列，因此项目定位为**探索性数据分析与方法预实验**，不能直接声称已经完成煤矸识别验证。

## 当前数据边界

- `datas/data.xlsx` 未包含在仓库中（`.gitignore`）。如需运行，请将原始数据文件放入 `datas/` 目录。
- 第 3 列已校验为第 2 列的 30 点后向移动平均，不是独立传感器数据。
- 当前缺少人工标签、放煤窗口动作日志、视频标注、传感器参数和多组独立数据。
- 输出结果只能解释为“计数率阶段变化”和“无标签状态划分”，不能直接解释为真实煤、夹矸或顶板岩石状态。

## 环境准备

推荐使用 Anaconda Python，或自行创建虚拟环境安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

如果本机已有 Anaconda，也可以使用 Anaconda Python 运行：`python main.py`

## 主要功能

- 读取并校验 `datas/data.xlsx`。
- 构造移动平均、滑动标准差、差分、斜率和分位数等特征。
- 进行探索性阈值分层、动态规划变点检测、GMM 组件选择和高计数异常筛选。
- 生成参数敏感性分析，回应审阅意见中关于 BIC、阈值、GMM 和 Isolation Forest 的问题。
- 输出图表、诊断表和修订后的分析摘要。

## 输出文件

- `outputs/processed_radiation_features.csv`：处理后的完整特征表。
- `outputs/changepoint_segments.csv`：变点分段统计表。
- `outputs/state_counts.csv`：探索性阈值状态计数。
- `outputs/analysis_summary.json`：核心指标、限制说明和输出路径。
- `outputs/fig*.png`：主要数据图。
- `outputs/concept01_radiation_mechanism_diagram.png`：概念示意图（非数据结果图）。
- `outputs/diagnostics/`：参数敏感性分析表和诊断图。
- `reports/no_new_data_optimization_plan.md`：无新增数据条件下的优化计划。
- `reports/review_response_matrix.md`：审阅意见回应表。

## 验证

```powershell
python -m py_compile analysis.py main.py src\data_loader.py src\features.py src\models.py src\diagnostics.py src\plots.py
python -m pytest
python main.py
```

## 当前结论的正确表述

可以说：

> 当前计数率序列存在明显阶段变化，变点检测和统计诊断可以描述这些变化，并为后续有标签煤矸识别研究提供方法准备。

不能说：

> 当前方法已经识别出煤、夹矸和顶板岩石状态。

