# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**（eta(T,N)≈0.012≪0.20；T−X 未全 seed >0）

实验：E109-COEXIST-FILM。R5 后唯一假设。合同 SHA256 `407ba76f757f16c98f4a262c562ac9d4068a98cd45d83e537e5377637b40fb39`。

## 准入与训练

- Native rebuild 0；zero-init FiLM 近恒等；9/9，4500 更新。

## 334

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 38 | 35 | 39 | **+3** | −1 |
| 101 | 37 | 36 | 38 | **+1** | −1 |
| 102 | 42 | 35 | 36 | **+7** | +6 |

- eta(T,N)≈**0.012**；eta(T,X)≈0.0045。
- T 全 seed ≥N，但幅度远低于门槛，且相对打乱长度 X 不稳。

**REJECT。** 不采纳 H109。不重开 H100–H108。根因仍 OPEN。
