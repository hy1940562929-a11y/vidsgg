# M102-EXTERIOR-SUPPRESS：区外负点升权的 mask 监督

2026-10-21。**UNVERIFIED / NOVELTY_UNCERTAIN / NOT_ADOPTED**。对应 [H102](CAUSE_STATEMENT.md)。

## 设计

```text
对已匹配事件 e：P=GT正集，N=valid\P
T: w_t = β_e (t∈N), 1 (t∈P)；β_e = V_e / max(d_e,1)（短→强区外抑制）
   focal ← ∑ w_t φ_t / ∑ w_t m_t
X: 同等 β，但负升权施加在「随机错位、等长伪区间」之外的点上（抑制错误位置）
Dice / class / Hungarian / hull 不变
```

## 文献

Focal / hard-negative mining、实例分割 exterior/boundary loss、时序定位内外对比（AutoLoc 等）已有近邻；本设计在 VrdONE 二值 mask 上做**区外负权**且用错位对照，不声称首创 hard-negative。E091 几何辅助与 H101 正点归一均已停，本方案不是它们的换名。

VERIFIED 仅支持限域区外抑制效应；REJECT 则停止本配方。
