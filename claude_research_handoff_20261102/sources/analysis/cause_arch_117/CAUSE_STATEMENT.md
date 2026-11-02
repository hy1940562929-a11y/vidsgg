# H117：短区间失败是否因为 mask 损失在长网格上均匀对待跨度内帧、缺少中心度加权？

2026-10-21。[R15](../cause_arch_residual_synthesis_20261021_r15/). Lit: ActionFormer (ECCV’22), FCOS (ICCV’19), TriDet (CVPR’23).

> **H117：长共存短关系残余区间失败，主要因为原生 masked focal+dice 对 GT 跨度内每一帧等权，短事件的中心证据被长网格上的平坦正样本稀释；在保持原生匹配、损失系数结构与 mask→hull 解码下，对 T 用 GT 中心高斯（centerness）重加权 focal/dice 的 `loss_mask`，对 X 用时间打乱的 GT (cx,w) 生成同一权重，应使 T 稳定优于 N 与 X。**

不是 H110（匹配代价按时长标量）；不是 H112（batch 过采样）；不是 H114（匹配 GIoU）；不是 H116（cxw 辅助损失）。
