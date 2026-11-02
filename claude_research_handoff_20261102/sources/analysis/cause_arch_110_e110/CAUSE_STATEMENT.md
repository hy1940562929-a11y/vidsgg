# H110：短区间失败是否因为 Hungarian 在长短共存时偏向长 GT？

2026-10-21。**交 E110**。[R6](../cause_arch_residual_synthesis_20261021_r6/).

> **H110：长共存短关系残余失败，主要因为原生匹配代价使 query 在同 pair 多 GT 时更易绑到长事件；在保持原生损失系数与 hull 下，对短 GT 降低（T）/对长 GT 降低（X）匹配代价，应使 T 稳定优于 N 与 X。**

不是 H100（跨层 assignment 身份），不是 H101/H102（focal 损失重加权）。
