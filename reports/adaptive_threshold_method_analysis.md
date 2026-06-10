# 自适应阈值方法对比阶段报告

## 1. 本阶段目的

在上一阶段已经识别出“候选局部本底窗口”的基础上，本阶段进一步把候选窗口转化为每组数据内部的局部本底估计，并构建多种阈值策略进行对比。

本阶段仍然只使用计数率时间序列，不引入支架动作、输送机速度、文件编号含义等外部信息。因此结果应表述为“有限数据条件下的自适应阈值信息挖掘”，不能表述为已经确定现场通用煤矸判别阈值。

## 2. 方法设计

每组数据独立处理，先从已选候选本底窗口中估计：

- 局部本底 `B_candidate`
- 本底波动 `sigma_candidate`
- 相对辐射贡献 `R(t)=X(t)-B_candidate`

随后比较 5 种阈值策略：

| 方法 | 含义 | 作用 |
| --- | --- | --- |
| `fixed_absolute_85_4` | 固定绝对阈值 85.4 | 传统固定阈值对照 |
| `static_initial_delta` | 初始 5 s 本底 + 参考增量 | 静态本底阈值对照 |
| `candidate_stat_3sigma` | 候选本底 + 3 倍候选窗口标准差 | 敏感统计预警线 |
| `candidate_poisson_99` | 候选本底对应的泊松 99% 上限 | 区分统计涨落与真实高计数贡献 |
| `candidate_reference_delta` | 候选本底 + 参考增量 | 当前主推的局部本底自适应对比线 |

其中，`candidate_reference_delta` 不是绝对通用阈值，而是把参考增量叠加到各组数据自身候选本底上，用于体现“不同数据采用各自局部本底”的思想。

## 3. 主要结果

| 工况 | 候选本底 | 候选本底波动 | 方法 | 阈值均值 | 触发样本数 | 最长连续触发时间/s | 超限面积 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| excessive_caving | 60.45 | 2.42 | fixed_absolute_85_4 | 85.40 | 76 | 3.2 | 15.98 |
| excessive_caving | 60.45 | 2.42 | static_initial_delta | 85.00 | 83 | 3.3 | 19.16 |
| excessive_caving | 60.45 | 2.42 | candidate_stat_3sigma | 67.71 | 200 | 16.1 | 265.57 |
| excessive_caving | 60.45 | 2.42 | candidate_poisson_99 | 79.00 | 129 | 8.6 | 86.28 |
| excessive_caving | 60.45 | 2.42 | candidate_reference_delta | 85.85 | 68 | 3.1 | 12.74 |
| minor_caving | 62.05 | 2.09 | fixed_absolute_85_4 | 85.40 | 4 | 0.4 | 0.43 |
| minor_caving | 62.05 | 2.09 | static_initial_delta | 86.85 | 2 | 0.2 | 0.07 |
| minor_caving | 62.05 | 2.09 | candidate_stat_3sigma | 68.33 | 66 | 5.7 | 75.48 |
| minor_caving | 62.05 | 2.09 | candidate_poisson_99 | 81.00 | 37 | 2.8 | 11.10 |
| minor_caving | 62.05 | 2.09 | candidate_reference_delta | 87.45 | 0 | 0.0 | 0.00 |
| normal_caving | 61.10 | 2.40 | fixed_absolute_85_4 | 85.40 | 42 | 2.2 | 26.06 |
| normal_caving | 61.10 | 2.40 | static_initial_delta | 86.30 | 40 | 2.2 | 22.45 |
| normal_caving | 61.10 | 2.40 | candidate_stat_3sigma | 68.31 | 155 | 8.7 | 148.86 |
| normal_caving | 61.10 | 2.40 | candidate_poisson_99 | 80.00 | 54 | 5.4 | 53.00 |
| normal_caving | 61.10 | 2.40 | candidate_reference_delta | 86.50 | 39 | 2.1 | 21.66 |

## 4. 阶段性解释

1. `candidate_stat_3sigma` 阈值明显偏低，触发样本多、持续时间长。它更适合作为“轻微升高/预警线”，不宜直接作为显著高计数判别线。
2. `candidate_poisson_99` 阈值处于中间水平，能够把随机统计涨落之外的持续高计数段筛出来，是后续区分“统计波动”和“真实附加辐射贡献”的关键依据。
3. `candidate_reference_delta` 与固定绝对阈值接近，但它不是固定 85.4，而是根据各数据自身候选本底平移得到，因此更符合局部本底自适应思想。
4. 三组数据中，`minor_caving` 在 `candidate_reference_delta` 下没有触发，说明其高计数贡献不足以达到参考增量线；但在泊松 99% 线下仍存在短时超限，说明其可能存在轻微或短时的相对升高。
5. 后续不应只比较是否超过某一条线，而应提取事件级指标，例如持续时间、峰值、超限面积和相对本底增量。

## 5. 输出文件

- `outputs/new_data/adaptive_threshold_components.csv`
- `outputs/new_data/adaptive_threshold_summary.csv`
- `outputs/new_data/fig_adaptive_threshold_methods_normal_caving.png`
- `outputs/new_data/fig_adaptive_threshold_methods_minor_caving.png`
- `outputs/new_data/fig_adaptive_threshold_methods_excessive_caving.png`
- `outputs/new_data/fig_adaptive_longest_run_summary.png`
- `outputs/new_data/fig_adaptive_excess_area_summary.png`

## 6. 下一步

下一阶段应基于本阶段生成的阈值触发结果，提取高计数事件表。事件表应至少包含：数据来源、触发方法、起止时间、持续时间、峰值、最大超限幅度、超限面积、相对本底峰值、事件类型等。
