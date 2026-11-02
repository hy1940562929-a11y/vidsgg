# M103-UNIT-MASS-W1：单位事件时间分布监督（读出暂保留 hull）

2026-10-21。**UNVERIFIED / NOVELTY_UNCERTAIN / NOT_ADOPTED**。对应 [H103](CAUSE_STATEMENT.md)。继承 [090 M090](../mechanism_synthesis_090/disruptive_proposal.md) 的监督核，**本轮不改推理读出**。

## 训练对象

对已匹配 query-事件：
- 令 u_t 为 mask logit，ρ_t = softmax_{t∈valid}(u_t)
- ρ\*_t = 1[t∈GT]/d ，d=\|GT∩valid\|
- L_mask = (∑_k \|F_ρ(k)−F_ρ\*(k)\| Δ) / max(d_frames, Δ)  （一维 W1，按 GT 时长尺度归一）
- 分类 CE 与 Hungarian（原生 class+focal+dice **成本**，成本不反传新 W1）保持；Dice/focal **主损失**在 T/X 中关闭，由 L_mask 替代（系数取原 loss_mask+loss_dice 之和=7，避免总尺度漂移）

X：ρ\* 为 valid 上均匀分布（反集中），同 W1 与系数。

## 文献

Wasserstein/CDF 时间定位、弱监督内外对比、090 草案；E091 已否定「二值上叠加几何辅助」。单位质量**替换**二值主监督在本项目仍为未验。不声称 OT 首创。
