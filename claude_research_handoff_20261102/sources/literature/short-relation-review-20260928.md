# 短持续关系：先例、效果与下一步诊断

核查日期：2026-09-28。Commander 整合 Direction 原文调查与 Explorer 公开源码核查。文献报告、本项目导出观察、源码事实和待验证解释分别记录。本轮没有连接 AutoDL 或运行模型。

## 结论

短持续关系较难已有直接先例，时间粒度、变长提案、时间定位与局部/全局表示也已有方法。现有核查不能支持“首次发现短关系困难”。同时，所读论文没有给出直接对应本项目 **VidOR `<1秒` GT、Top100、同一输入与匹配器** 的解决证据。可继续的问题是：当前配置中哪些具体环节导致亚秒关系损失，以及固定其它环节后是否能低成本恢复。

## 已核查原文

### 1. Diagnosing Errors in Video Relation Detectors，BMVC 2021

[原文](https://isis-data.science.uva.nl/cgmsnoek/pub/chen-diagnosing-bmvc2021.pdf)，§2.2、§3.2、Fig.5–6。

**文献报告**：按绝对秒数分 S=(0,10]、M=(10,20]、L>20。Fig.5 印出的 FNR：

| 数据集/模型 | S | M | L |
|---|---:|---:|---:|
| VidOR / Liu 2020 | 90% | 79% | 64% |
| VidOR / Su 2020 | 88% | 82% | 70% |
| ImageNet-VidVRD / Liu 2020 | 82% | 79% | 85% |
| ImageNet-VidVRD / Su 2020 | 89% | 84% | 82% |

VidOR 短组更难已有直接证据；ImageNet-VidVRD/Liu 说明这不是所有模型条件下单调成立的规律。§2.2 默认每视频Top200；不能与本项目Top100或`<1s`直接比较。Fig.6 的 VidOR 短组 6.77/7.06 mAP gain 是移除相应漏检GT的oracle诊断，不是新方法提高了这些数值。

**uncertainty**：Fig.4 总体FNR与Fig.5分组数字不能直接用同一micro分母加权复算；原文未明确解释聚合差异，Fig.5专属代码未复核。因此这里只引用论文图中报告的组别FNR，不换算为本项目同定义Recall。

### 2. Beyond Short-Term Snippet / VRD-STGC，CVPR 2020

[原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Beyond_Short-Term_Snippet_Video_Relation_Detection_With_Spatio-Temporal_Global_Context_CVPR_2020_paper.pdf)，Introduction、Fig.2、方法部分。

**文献方法**：用多尺度滑窗在对象轨迹上生成不同长度的关系提案，结合空间/时间图上下文；主要针对短clip缺乏长关系上下文及逐段预测的重复计算。它已经处理“不同关系需要不同时间范围”，但不是针对`<1s`专门证明有效。下述Social Fabric同特征对比中的6.85为该工作在VidOR上的对照，不把不同论文/特征的总分当短关系修复率。

### 3. Social Fabric，ICCV 2021

[原文](https://arxiv.org/pdf/2108.08363)，§3–6、Fig.5、Table 5。

**文献方法**：用交互基元组合编码tubelet pair，生成变长交互提案。**文献结果**：在与Liu 2020相同visual/I3D/motion特征下，VidOR detection mAP 6.85→9.13。Fig.5还按时长比较，作者报告各组改善。

关键边界：该时长图纵轴是 **P@1，不是关系检测Recall**；图无精确数值标签，所查正文未明确三个时长阈值，不估读为精确亚秒收益。整体与变长关系改善已有证据，`<1s`召回是否解决仍未建立。

### 4. TSPN，2021预印本；同题工作2026正式发表

[2021 v1原文](https://arxiv.org/pdf/2107.07154v1)，§III、§IV-E(c)、Tables III/VI；[ESWA 2026页面](https://www.sciencedirect.com/science/article/pii/S0957417425031185)。以下数值均取2021版本，不混合版本。

**文献方法**：对整个对象对共存区间划分temporal sectors，联合预测关系和时间占据。**文献结果**：ImageNet-VidVRD test，sector数4/8/16/32，对应R@100 6.25/11.49/14.13/14.50，mAP 9.79/14.74/18.90/18.69。VidOR val总体mAP 7.61、R@100 10.71，每视频保留Top100。

时间量化粒度影响整体检测已有实测，不是新问题。Eq.(5)后的多数覆盖标正规则可能影响极短关系是结构推论，文中这些数字不是短关系专门召回。

### 5. BIG，CVPR 2022

[原文](https://arxiv.org/pdf/2112.04222)，§4.3、Tables 2/6。

**文献方法**：先在整段对象轨迹对上分类，再用多实例grounding定位关系发生区间，减少固定提案依赖。**文献结果**：VidOR同一消融中BIG-C→BIG，mAP 8.29→8.54，R@100 9.65→10.04。多实例grounding提升了反复出现关系的fraction recall。

这证明定位阶段有实际总体贡献；没有据此得到`<1s`专门改善。BIG的不同特征/变体不能混为同一实验，发布命中缓存与本项目VrdONE也不是共享输入消融。

### 6. HCM，2023

[原文](https://arxiv.org/pdf/2307.08984)，§III、§IV-B、Tables I/II。

**文献方法**：clip对象空间图与关系时间图，并改进关系关联，反对简单将长tube视为更优。**文献结果**：ImageNet-VidVRD test无上下文→完整GCN，R@100 16.38→19.96；greedy→relaxed greedy，R@100 19.96→21.45、mAP 28.85→29.68。它支持上下文与关联都有影响，不支持“更大时间上下文必然更好/更坏”。这些消融不是亚秒子集结果。

### 7. VrdONE，ACM MM 2024

[原文v2](https://arxiv.org/html/2408.09408v2)，Fig.3、§3.4、Tables 3/7/10、Appendix C。

**文献方法**：局部到全局交互与多尺度编码，联合谓词分类和一维实例mask。作者明确面向短暂/持续关系，不能说忽略过短关系。

**文献结果**：VidOR Table10的边界回归→实例分割，mAP 11.28→11.86、R@100 13.85→14.21。Table7换为GT轨迹及实体类别后，R@100 14.21→47.79；这是输入oracle，不是可部署提升，也未分解短关系。

正文及附录已核查的表格未提供`<1s`分组Recall。总体改善和定性短关系成功案例不等于亚秒漏检已解决；本项目0.51%也不能用来推翻原文的总体结果。

## 文献如何约束下一步

1. 先确认真实推理配置和候选路径，不能以短时低召回直接指向历史干扰。
2. 固定GT与官方评价，分别检查输入轨迹、采样表示、mask边界、分类/TopK和贪心分配。
3. 先做现有缓存的CPU审计；有过滤前mask时才能直接验证某条预测在哪个步骤被删除。若没有，不把最终CSV当作过滤前证据。
4. 若只有提高K或放宽长度门槛就恢复，优先作为简单对照/配置诊断；若轨迹覆盖充分、正确谓词证据存在，而特定边界或采样误差仍稳定损害短关系，再设计可干预机制。

下一步人工包：`handoff/outgoing/VIDOR-SHORT-AUDIT-006/`。短关系观察不会自动成为主方向。

## 检索覆盖与限制

实际使用原文/作者PDF、arXiv、CVF、官方GitHub和期刊页面；搜索组合包含 VidVRD/VidOR、short duration/short-lived、relation length、temporal localization/granularity、2025/2026。检索截至2026-09-28。

VSTRD 2025已读期刊摘要，目标偏长段关系断裂，未取得亚秒专项结果，未纳入效果表。TSPN正式2026页面已核，实验表只引用2021版本。VidVRD-II本轮未完整取得原文；OpenReview `Hhr220TJFh` 遇验证页面，未核实标题/结果，不当作证据。近年覆盖不完整，故结论是“未核实与当前协议直接对应的解决证据”，不是“不存在后续工作”。
