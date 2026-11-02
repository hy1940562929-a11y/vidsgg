# E108-DN-SHORT：唯一决定性实验合同

2026-10-21。`analysis/cause_arch_108_e108/`。

## 1. 四臂

| 臂 | 训练 query | 推理 |
|---|---|---|
| N | 仅 9 content | 9 content |
| T | 9 content + DN(noisy true spans) | 9 content |
| X | 9 content + DN(shuffled spans, same classes) | 9 content |
| C0 | 原生，零更新 | 9 content |

禁止同时改 `x`/`f` 适配器、损失系数、hull。不重开 H100–H107。

## 2–6

同前预算（3 seeds×500，E100-A1 支持）；eta≥0.20 且全 seed T−N>0（及 T−X 结构对照）。准入：init 时 content 路径 T≈N；T/X 均有 DN 槽。VERIFIED/SCOPED_DN_SHORT_EFFECT 或 REJECT。
