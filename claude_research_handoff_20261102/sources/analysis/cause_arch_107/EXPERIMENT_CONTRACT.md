# E107-MASKFEAT-ADAPTER：唯一决定性实验合同

2026-10-21。`analysis/cause_arch_107_e107/`。

## 1. 四臂

| 臂 | mask_features | 训练 |
|---|---|---|
| N | 原生 `f` | predictor |
| T | `adapter(f)` | predictor + adapter |
| X | `adapter(flip_time(f))` | predictor + adapter |
| C0 | 原生 `f`，零更新 | — |

`x`/Hungarian/loss/hull 原生。禁止同时改 `x` 适配器（不重开 H106）。

## 2–6

同 E106 预算与 eta 门槛；准入：init T≈N（近恒等）、T≠X（反转）；支持 E100-A1。VERIFIED/SCOPED_MASKFEAT_ADAPTER_EFFECT 或 REJECT。
