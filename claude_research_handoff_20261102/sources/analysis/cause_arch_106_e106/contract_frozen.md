# E106-TEMPORAL-ADAPTER：唯一决定性实验合同

2026-10-21。`analysis/cause_arch_106_e106/`。

## 1. 四臂

| 臂 | 前向 | 训练参数 |
|---|---|---|
| N | 原生 `x`→predictor | predictor only（同历史） |
| T | `adapter(x)`→predictor | predictor + adapter |
| X | `adapter(flip_time(x))`→predictor | predictor + adapter（同结构） |
| C0 | 原生，零更新 | — |

损失/Hungarian/hull 原生。Adapter：2×Conv1d(C→C,k=3,padding=1) + GELU，残差；第二层权重零初始化。

## 2. 预算

E100-A1 机制-dev；400/334/…；N/T/X×seed100/101/102×500；AdamW 3e−4；AutoDL `:16241`。

## 3. 准入

重建抽检；C0 46/400、29/334；T 与 N 前向在 C0 初始化下接近（adapter≈id，max abs 或 loss 差账）；X 与 T 在同一初始权重下对 batch 的 mask L1>0（反转有效）。支持 ≥4/20/20。

## 4–6. 终点/机制/出口

同前 eta≥0.20 等。机制：固定 q0 短组 AUROC。VERIFIED/SCOPED_TEMPORAL_ADAPTER_EFFECT 或 REJECT。不重开已关闭配方。
