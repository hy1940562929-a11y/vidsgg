# H114：短区间失败是否因为 Hungarian 缺少 DETR 式时间 GIoU 代价？

2026-10-21。[R11](../cause_arch_residual_synthesis_20261021_r11/). Lit: DETR (ECCV’20), Moment-DETR (NeurIPS’21), DINO (ICLR’23), ActionFormer (ECCV’22).

> **H114：长共存短关系残余区间失败，主要因为原生 `bipartite_match` 只有 class+mask+dice、缺少对预测 mask 诱导跨度与 GT 跨度的时间 GIoU/tIoU 几何代价，导致短事件在赋值上被 mask/dice 主导；在保持原生损失系数与 mask→hull 解码下，对 T 增加 λ·(1−GIoU) 匹配项，对 X 使用时间打乱的 GT 跨度计算同一代价，应使 T 稳定优于 N 与 X。**

不是 H110（按时长**重加权**已有代价）；不是 H113（cxw **解码替换** hull）。
