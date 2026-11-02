# H115：短区间失败是否因为缺少 ActionFormer 式多尺度时间特征？

2026-10-21。[R12](../cause_arch_residual_synthesis_20261021_r12/).

> **H115：残余短关系失败，主要因为解码器只看见单尺度 cached `f`；在 `f` 上加 ActionFormer 式 1×/2×/4× 时间 FPN（零初始化残差融合）应稳定优于原生 N，以及同一 FPN 但时间反转 `f` 的 X。**
