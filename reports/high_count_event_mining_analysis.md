# 高计数事件信息挖掘阶段报告

## 1. 本阶段目的

上一阶段已经建立了局部本底与多种自适应阈值。若只统计“有多少采样点超过阈值”，仍然偏向单点判断，不能充分体现老师所说的“信息挖掘”。因此本阶段将连续超阈采样点合并为高计数事件，并提取事件级指标。

本阶段仍只使用计数率时间序列，事件不直接等同于已验证的煤矸类别标签，而是表示“相对局部本底和阈值线出现持续高计数贡献的时间段”。

## 2. 事件定义

对每一种阈值方法，若平滑计数率 `ma10_count_rate` 连续超过对应阈值，且持续时间不小于 `0.3 s`，则合并为一个高计数事件。

本阶段保留 5 类阈值方法的事件结果：

- `fixed_absolute_85_4`
- `static_initial_delta`
- `candidate_stat_3sigma`
- `candidate_poisson_99`
- `candidate_reference_delta`

其中论文解释时建议重点使用：

1. `candidate_poisson_99`：用于区分统计涨落之外的真实高计数贡献。
2. `candidate_reference_delta`：用于体现局部本底自适应阈值思想。

## 3. 提取指标

每个事件提取以下指标：

| 指标 | 含义 |
| --- | --- |
| `start_time_s` / `end_time_s` | 事件开始和结束时间 |
| `duration_s` | 事件持续时间 |
| `peak_time_s` | 事件内峰值出现时间 |
| `peak_value` | 事件峰值计数率 |
| `mean_value` | 事件内平均计数率 |
| `max_excess` | 相对阈值的最大超限幅度 |
| `mean_excess` | 相对阈值的平均超限幅度 |
| `excess_area` | 超限面积，反映持续时间和超限强度的综合贡献 |
| `relative_peak_from_background` | 峰值相对候选本底的增量 |
| `rise_slope_per_s` | 事件上升阶段斜率 |
| `fall_slope_per_s` | 事件回落阶段斜率 |

这些指标比单点阈值更适合表达时间序列中的高计数贡献。

## 4. 关键结果

以下表格展示两条重点阈值线的事件结果：

| 工况 | 方法 | 事件数 | 总持续时间/s | 最长事件/s | 总超限面积 | 最大峰值 | 最大相对本底峰值 | 首次事件开始/s | 最后事件结束/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| excessive_caving | candidate_poisson_99 | 2 | 12.6 | 8.6 | 86.05 | 89.9 | 29.45 | 42.5 | 56.5 |
| excessive_caving | candidate_reference_delta | 5 | 6.4 | 3.1 | 12.57 | 89.9 | 29.45 | 42.8 | 56.0 |
| minor_caving | candidate_poisson_99 | 3 | 3.7 | 2.8 | 11.10 | 87.2 | 25.15 | 25.2 | 29.4 |
| minor_caving | candidate_reference_delta | 0 | 0.0 | 0.0 | 0.00 | 0.0 | 0.00 | - | - |
| normal_caving | candidate_poisson_99 | 1 | 5.4 | 5.4 | 53.00 | 98.0 | 36.90 | 38.7 | 44.0 |
| normal_caving | candidate_reference_delta | 2 | 3.9 | 2.1 | 21.66 | 98.0 | 36.90 | 39.1 | 43.7 |

## 5. 阶段性认识

1. 泊松 99% 阈值能够识别出比随机统计涨落更明显的高计数事件，因此适合作为“真实高计数贡献筛查线”。
2. 候选本底加参考增量更严格，保留的是更显著的高计数事件。`minor_caving` 在该方法下没有事件，但在泊松 99% 下有短时事件，说明其更接近轻微升高或短时贡献。
3. `normal_caving` 和 `excessive_caving` 都存在较明显事件，但事件形态不同：`normal_caving` 在泊松 99% 下表现为一个较集中事件，`excessive_caving` 则表现为更长总持续时间和更高总超限面积。
4. 事件级指标比“超过阈值样本比例”更有论文价值，因为它能够描述高计数贡献的持续性、强度和集中程度。

## 6. 输出文件

- `outputs/new_data/high_count_events.csv`
- `outputs/new_data/high_count_event_summary.csv`
- `outputs/new_data/fig_high_count_event_count_summary.png`
- `outputs/new_data/fig_high_count_event_area_summary.png`
- `outputs/new_data/fig_high_count_event_duration_summary.png`
- `outputs/new_data/fig_high_count_event_timeline_normal_caving.png`
- `outputs/new_data/fig_high_count_event_timeline_minor_caving.png`
- `outputs/new_data/fig_high_count_event_timeline_excessive_caving.png`

## 7. 后续工作

下一阶段应把候选本底窗口、自适应阈值和事件信息挖掘三部分整合进论文初稿的“方法”和“结果分析”。建议先更新论文草稿，再更新给老师看的阶段性汇报 Word。
