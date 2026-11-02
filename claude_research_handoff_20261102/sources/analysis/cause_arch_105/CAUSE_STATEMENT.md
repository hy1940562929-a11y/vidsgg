# H105：硬阈值 hull 是否丢弃了已有软时间质量中的短区间？

2026-10-21。**交 E105**。**NOT_VERIFIED_CAUSE**。依据 [residual synthesis](../cause_arch_residual_synthesis_20261021/RESIDUAL_SYNTHESIS.md) + [literature map](../cause_arch_residual_synthesis_20261021/LITERATURE_MAP.md)。

## 为何在地图上存活

- H101–H103：改 **训练损失**、保留 hull → REJECT（含 T 更差）。  
- 088：分数相对 null 有定位信息；拆 hull 不够当主因。  
- 078：换起止头失败。  
- Lit：ActionFormer / Moment-DETR 用 **连续区间解码**；本基线是 mask→sigmoid>0.5 hull。  
- **未隔离检验**：同一套原生权重下，仅把软 mask 质量解成区间（分位/期望）是否恢复 334。

> **H105：在长共存短关系上，原生 sigmoid>0.5 首末正点 hull 丢弃了 mask 软质量中可用的短支撑；对 valid 上归一化的 sigmoid 质量做质量分位区间解码（保持训练与分类不变），应比同权重 hull、以及时间轴打乱后的同分位解码，更稳定地改善残余短区间命中，并保护长关系。**

唯一操纵：**推理/评价解码**。训练全部为原生 focal+Dice（三 seed 各训一次，N/T/X 共享权重）。
