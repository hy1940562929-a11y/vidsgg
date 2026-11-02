# E105-SOFT-MASS-DECODE：唯一决定性实验合同

2026-10-21。对应 [H105](CAUSE_STATEMENT.md)。`analysis/cause_arch_105_e105/`。

## 1. 臂定义（解码臂；训练共享）

| 臂 | 训练 | 评价解码 |
|---|---|---|
| N | 原生 loss，seed∈{100,101,102}，各 500 步 | 原生 sigmoid>0.5 hull |
| T | **共享**对应 seed 的 N 权重（不另训） | 软质量分位 [0.1,0.9]→区间 |
| X | 同 N 权重 | 时间反转后同分位 |
| C0 | 零更新 | 分别报告 hull / soft / reverse-soft |

因此优化轨迹只有 3 条（N_100/101/102），不是 9 条独立优化——这是有意隔离读出。报告中写明。

## 2. 数据与门槛

同前：E100-A1 机制-dev；评价 400/334/163/1101/2573；eta 公式同 E100（相对 N/X/C0-hull）。  
VERIFIED 仍要求每 seed T−N 与 T−X 净命中>0 且 eta≥0.20 等保护组门槛。  
准入：C0 上 soft 与 hull 候选集合不完全相同（L0 差异或命中差账）；reverse≠soft。

## 3. 出口

TECHNICAL_FAILURE → INTERVENTION_INACTIVE → VERIFIED/SCOPED_SOFT_MASS_DECODE_EFFECT → REJECT（EFFECT_TOO_SMALL / TRUE_PATH_NOT_BETTER / METHOD_ONLY / PROTECTION_FAILED）。

不重开 H100–H104；不做损失重加权。
