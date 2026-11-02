# E109-COEXIST-FILM：唯一决定性实验合同

2026-10-21。`analysis/cause_arch_109_e109/`。

## 1. 四臂

| 臂 | query FiLM 条件 | 训练 |
|---|---|---|
| N | 无（原生 query_embed） | predictor |
| T | 真实 coexist 长度 | predictor + FiLM MLP |
| X | batch 内打乱 coexist 长度 | predictor + FiLM MLP |
| C0 | 无，零更新 | — |

禁止 DN、x/f 适配器、改损失/hull。同前 eta≥0.20、全 seed T−N>0 及 T−X 结构对照。准入：init T≈N（近恒等）；T≠X。VERIFIED/SCOPED_COEXIST_FILM_EFFECT 或 REJECT。
