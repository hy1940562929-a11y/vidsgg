# H116：短区间失败是否缺 Moment-DETR 式跨度辅助损失（保留 hull）？

2026-10-21。R13。Lit: Moment-DETR, DETR, Mask DINO.

> **H116：在保持 mask→hull 评估下，对匹配 query 增加 (cx,w) L1+GIoU 辅助损失应稳定优于 N，以及 GT 跨度打乱的 X。** 不是 H113（曾替换解码）。
