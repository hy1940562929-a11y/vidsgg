# E103-UNIT-MASS-W1：唯一决定性实验合同

2026-10-21。对应 [H103](CAUSE_STATEMENT.md)。目录 `analysis/cause_arch_103_e103/`。

## 1. 四臂

| 臂 | mask 训练损失 | 推理 |
|---|---|---|
| N | 原生 focal+Dice | 原生 hull |
| T | 单位质量 W1（ρ=softmax_valid(u)，ρ\*=1_GT/d），系数 7；无 focal/Dice 主项 | 同 hull |
| X | 同 T 但 ρ\*=uniform_valid | 同 hull |
| C0 | 零更新 | hull |

分类损失、Hungarian **原生成本**、query 数、层数、预算不变。不改 hull/阈值；不做矩读出；不做 focal 点权。

d=0：该事件 mask 项记 0 并入账。全正 mask（d=V）：T 的 ρ\* 为均匀，与 X 同形——此类事件单独计数，不删。

## 2. 数据与预算

同 E102：400/334/163/1101/2573；E100-A1 机制-dev 16/302；N/T/X×100/101/102×500；AutoDL `:16241`。

## 3. 准入

1. 重建抽检 max abs≤4e−6；C0 46/400、29/334。
2. 操纵：C0 上 mean(L_T) 与 mean(L_N)、mean(L_X) 与 mean(L_T) 可区分；且 mean(W1 toward true) 与 mean(W1 toward uniform) 可区分（不要求 C0 上 true<uniform——弥散预测下常相反；训练后机制项再检验 true 侧改善）。否则 `INTERVENTION_INACTIVE`。
3. 支持 ≥4/20/20；容量见证可核验。

## 4–6. 终点 / 机制 / 出口

同前：eta(T,·)≥0.20 且 CI 下界>0；每 seed T−N、T−X>0；400 与保护组；机制：固定 q0 短组 AUROC 与 **W1(ρ,ρ\*)** 训练后相对 N 下降。

互斥：TECHNICAL_FAILURE → … → VERIFIED/SCOPED_UNIT_MASS_W1_EFFECT → REJECT（EFFECT_TOO_SMALL / TRUE_PATH_NOT_BETTER / METHOD_ONLY / PROTECTION_FAILED）。

不重开 H100–H102；REJECT 不自动改读出并行第二假设。
