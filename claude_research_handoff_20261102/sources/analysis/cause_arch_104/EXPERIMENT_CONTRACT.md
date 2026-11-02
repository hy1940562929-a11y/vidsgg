# E104-DEPTH-MASK-AGG：唯一决定性实验合同

2026-10-21。对应 [H104](CAUSE_STATEMENT.md)。`analysis/cause_arch_104_e104/`。

## 1. 四臂

| 臂 | 训练 | 推理 hull |
|---|---|---|
| N | 原生：逐层独立 match + class/focal/Dice（含 aux） | 末层 mask |
| T | 一次 match：成本用末层 class + **四层 mask logits 等权均值**；loss= class(末层)+focal/Dice(均值 mask)；**无**逐层 aux mask/class | 均值 mask |
| X | 同 T，但 mask 只用 **最浅 aux 层（层1）** | 层1 mask |
| C0 | 零更新 | 末层（与 N 同） |

不改 cost_factor、不改点权、不加 W1、不改 hull 阈值。T 相对 N 少了 aux 深监督——若仅因少 aux 而差，走 REJECT，不另开臂补 aux（避免双操纵）。记录 aux 省略为设计边界。

## 2. 数据预算

同 E103：E100-A1 16/302；400/334/…；N/T/X×100/101/102×500；AutoDL `:16241`。

## 3. 准入

1. 重建抽检；C0 46/400、29/334。
2. 操纵：C0 上 fused≠last 的 mask 均值 L1>0（短事件子集）；T 与 N 标量 loss 可区分；否则 `INTERVENTION_INACTIVE`。
3. 支持 ≥4/20/20。

## 4–6. 终点/机制/出口

同前 eta≥0.20 等。机制：固定 q0 短组 AUROC；另报末层 vs 聚合 mask 的区内外 margin。

VERIFIED / SCOPED_DEPTH_MASK_AGG_EFFECT 或 REJECT（EFFECT_TOO_SMALL / TRUE_PATH_NOT_BETTER / …）。不重开 H100–H103。
