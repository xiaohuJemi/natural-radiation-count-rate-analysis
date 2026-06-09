# 自然射线计数率探索性时序分析报告

## 1. 当前定位

本报告基于 `datas/data.xlsx` 中的一组自然射线计数率序列，进行探索性时序分析。当前数据没有人工标签、放煤窗口动作记录、视频标注、传感器参数和多组独立样本，因此所有模型输出只能解释为“计数率序列的统计变化”，不能直接解释为真实煤、夹矸或顶板岩石状态。

## 2. 数据事实

- 样本数：1407 行。
- 采样序号：1 到 2813，步长为 2。步长原因目前未知，只能作为事实报告。
- 原始计数率范围：40-166。
- 原始计数率均值：100.25，样本标准差：25.71。
- 第 3 列共 1378 个有效值，全部等于第 2 列的 30 点后向移动平均。
- 第 3 列公式结构也已校验，1378 个公式均符合 30 点后向移动平均形式。

因此，当前数据本质上是一条单变量计数率时间序列，第 3 列不提供额外传感器信息。

## 3. 已修正的问题

### 3.1 变点检测 BIC

原实现使用 `k * log(n)` 作为 BIC 惩罚项，现已改为 `2k - 1` 个参数，即 `k` 个段均值和 `k - 1` 个变点位置。

修正后，`max_segments=15` 时 BIC 仍在最大候选段数处下降，当前结果为：

- 选择段数：15。
- 是否达到最大候选段数：是。
- BIC 是否收敛：否。

这说明不能说“BIC 自然选择了 15 段”，只能说“在当前候选范围内，BIC 仍倾向更多分段，结果受上限约束”。

### 3.2 阈值法

阈值状态已从煤矸语义标签改为计数水平标签：

- `baseline_like_count`
- `elevated_count`
- `high_count`
- `very_high_count`

当前基线窗口 200 下，`very_high_count` 样本为 764 个，占 54.3%。因此该阈值方案不具备良好的区分性，不能作为可靠煤矸状态判据。

### 3.3 GMM

GMM 不再固定为 4 类，而是对 2-8 类计算 AIC、BIC 和轮廓系数。当前 BIC 最低对应 6 类，但对应轮廓系数约为 0.236，聚类结构仍较弱。因此 GMM 只保留为诊断性对照方法，不作为主识别方法。

### 3.4 Isolation Forest

Isolation Forest 输出改名为“高计数异常筛选”。当前 `contamination=0.06` 下筛出 36 个高计数异常样本。该结果不能解释为顶板岩石混入，只能说明这些点在统计特征上属于高计数异常。

## 4. 新增诊断输出

新增输出位于 `outputs/diagnostics/`：

- `changepoint_max_segments_sensitivity.csv`
- `changepoint_min_size_sensitivity.csv`
- `threshold_sensitivity.csv`
- `poisson_thresholds.csv`
- `gmm_model_selection.csv`
- `isolation_forest_sensitivity.csv`
- `fig08_changepoint_bic.png`
- `fig09_changepoint_max_segments_sensitivity.png`
- `fig10_threshold_sensitivity.png`
- `fig11_baseline_qq_plot.png`
- `fig12_gmm_aic_bic.png`
- `fig13_gmm_silhouette.png`
- `fig14_isolation_forest_sensitivity.png`

这些诊断文件用于回应审阅意见中关于参数随意性、BIC 截断、GMM 类数选择和异常检测敏感性的质疑。

## 5. 当前能够支持的结论

可以支持：

- 当前计数率序列存在明显阶段变化。
- 30 点移动平均能够降低原始计数率的随机波动。
- 变点检测可用于描述计数率曲线的阶段边界，但分段数受参数影响。
- GMM 聚类结构较弱，不适合作为当前数据的主识别依据。
- 阈值法对基线选择敏感，当前阈值不具备足够区分性。

不能支持：

- 不能证明高计数阶段一定对应顶板岩石混入。
- 不能证明算法已经实现煤矸识别。
- 不能报告识别准确率、召回率或 F1。
- 不能作为 SCI 或高水平中文期刊投稿的完整实验证据。

## 6. 下一步

在没有新增数据的条件下，下一步只能继续完善探索性报告和代码严谨性。如果要形成真正的煤矸识别论文，必须补充多组独立数据、传感器参数和至少一种 ground truth。

