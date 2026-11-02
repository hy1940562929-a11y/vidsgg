# E113-SPAN-CXW

| 臂 | 训练 | 评估解码 |
|---|---|---|
| N | native | hull |
| T | native + L1 cxw (true GT) | span→interval |
| X | native + L1 cxw (shuffled GT spans) | span→interval |
| C0 | none | hull |

3 seeds×500；E100-A1；eta≥0.20；全 seed T−N>0 且 T−X>0。准入：span head 近零 init 时 T≈N masks；shuffle 改变 X span targets. VERIFIED/SCOPED_SPAN_CXW_EFFECT 或 REJECT。
