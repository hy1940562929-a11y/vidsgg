# H113：短区间失败是否因为缺少 Moment-DETR 式连续跨度监督？

2026-10-21。[R10](../cause_arch_residual_synthesis_20261021_r10/). Lit: Moment-DETR (NeurIPS’21), ActionFormer (ECCV’22).

> **H113：长共存短关系残余区间失败，主要因为模型仅用二值 mask+hull、缺少对事件 (center,width) 的连续集合预测监督；在原生 mask/CE 上增加 Moment-DETR 式 cxw 头与 L1（匹配 query），并以 span 解码评估，应稳定优于仅 hull 的 N，以及 span 目标时间打乱的 X。**
