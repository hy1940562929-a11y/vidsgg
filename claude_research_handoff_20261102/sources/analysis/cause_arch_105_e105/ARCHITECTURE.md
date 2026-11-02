# M105-SOFT-MASS-DECODE：软质量分位区间读出

2026-10-21。**UNVERIFIED / NOT_ADOPTED**。

对每个 query 的 mask logits z_t、valid m_t：
1. p_t = σ(z_t)·m_t；Z=∑p；若 Z≈0 → 无候选。  
2. ρ_t = p_t/Z。  
3. 找最小 [a,b] 使 ∑_{t<a}ρ ≤ q_lo 且 ∑_{t≤b}ρ ≥ q_hi（默认 q_lo=0.1, q_hi=0.9），再投影到 stride4+min5 帧规则（与原生 hull 相同的帧换算）。  
4. 类别仍用末层 argmax。

X：在 valid 上 **时间反转** ρ 后再分位（同质量、错位置）。

训练：原生 VrdONE loss；不改匹配。文献对齐见 synthesis 地图 C3–C4。
