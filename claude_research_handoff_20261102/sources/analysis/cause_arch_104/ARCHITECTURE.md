# M104-DEPTH-MASK-AGG：解码层时间 mask 等权聚合

2026-10-21。**UNVERIFIED / NOT_ADOPTED**。对应 [H104](CAUSE_STATEMENT.md)。

```text
层1..4 mask logits → mean_l → 匹配 + focal/Dice + hull
末层 class logits → 分类损失（与聚合 mask 共用一次 Hungarian）
```

X：仅层1 mask。N：原生逐层匹配+损失，推理末层。

文献：多层特征聚合 / deep supervision 读出；Stable-DINO 等谈匹配稳定（H100 已测分配）。本设计测 **mask 证据聚合**，novelty 不确定。
