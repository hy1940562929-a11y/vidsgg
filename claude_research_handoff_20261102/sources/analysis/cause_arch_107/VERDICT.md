# REJECT

**子因：EFFECT_TOO_SMALL_OR_UNSTABLE**（eta(T,N)<0 且全 seed T−N≤0；远低于 0.20）

实验：E107-MASKFEAT-ADAPTER。R3 synthesis 后唯一假设。合同 SHA256 `888a991d7da358dbf2127740ce6925597ccdb958cdd84b7756b9b824e77e525d`。

## 准入与训练

- Adapter 近恒等：init T↔N mask L1=0；T↔X L1≈1.48（`f` 时间反转有效）。
- 支持 E100-A1；C0 residual334=29、main400=46。
- 训练 **9/9**，4500 更新。

## 334

| seed | T | N | X | T−N | T−X |
|---:|---:|---:|---:|---:|---:|
| 100 | 28 | 35 | 0 | **−7** | +28 |
| 101 | 33 | 36 | 0 | **−3** | +33 |
| 102 | 28 | 35 | 0 | **−7** | +28 |

- eta(T,N)≈**−0.019**；eta(T,X)≈0.100；eta(T,C0)≈0.002。
- 反转 `f` 的 X 臂崩到 ~0 命中，说明对照暴露有效；但适配 `f` 相对原生 N **全面更差**。

**REJECT。** 不采纳 H107。不重开 H100–H106。根因仍 OPEN。
