# E102-EXTERIOR-NEG-FOCAL：唯一决定性实验合同

2026-10-21。对应 [H102](CAUSE_STATEMENT.md)。可执行。目录 `analysis/cause_arch_102_e102/`。

## 1. 四臂

Hungarian **逐层独立原生**；分类、Dice、cost、hull、正点权重=1 均不变。只改 focal 负点权重。

| 臂 | 操作 |
|---|---|
| N | 原生 `mean` focal |
| T | 对 y=0∩valid：w=β_e=V_e/max(d_e,1)；y=1：w=1；加权平均。d_e=0 或 d_e=V_e 时退回 N |
| X | 在 valid 上均匀随机取起点，放置长度 d_e 的错位伪正集 P̃（与真 P 不同；若无法不同则整段循环移位 ≥1）；对 valid\P̃ 施同一 β_e，P̃ 内 w=1。不依赖评价标签以外的随机：种子 = hash(video,pair,gt_index,seed_train) 的确定性派生 |
| C0 | 零更新 |

匹配成本仍用原生未重加权 focal（避免分配身份混入）。

## 2. 数据与预算

同 E101：评价 400/334/163/1101/2573；events/cache SHA 同前；机制-dev **E100-A1 16/302**；N/T/X×seed100/101/102×500；AdamW 3e−4；冻 encoder/FPN。AutoDL `:16241`。

## 3. 准入

1. 原生重建抽检 max abs≤4e−6（沿用 E101 容差实践）；C0 命中 46/400、29/334。
2. 操纵暴露：短事件 mean(w_neg_T)/mean(w_neg_N) > 1+1e−6；否则 `INTERVENTION_INACTIVE`。
3. §5.1 支持 ≥4/20/20（继承 E100-A1 账）。
4. 容量见证可核验。

## 4–5. 终点与机制

同 E101/E100 风格：eta≥0.20 且 CI 下界>0；每 seed T−N、T−X>0；400/保护组门槛；机制用固定 q0 短组 AUROC 与**区外平均 sigmoid**（T 相对 N 应下降，CI 下界结构：区外均值差 N−T 的下界>0）。

## 6. 互斥出口

TECHNICAL_FAILURE → SCOPE → INTERVENTION_INACTIVE / MECHANISM_UNIDENTIFIABLE → VERIFIED / SCOPED_EXTERIOR_SUPPRESS_EFFECT → REJECT（TRUE_PATH_NOT_BETTER / EFFECT_TOO_SMALL_OR_UNSTABLE / METHOD_ONLY_NOT_CAUSE / PROTECTION_FAILED）。

不重开 H100/H101；REJECT 不自动并行下一假设。
