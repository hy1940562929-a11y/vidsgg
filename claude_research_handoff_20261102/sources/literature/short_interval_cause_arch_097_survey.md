# 097 文献对齐：局部状态比较，而非再加几何损失

日期：2026-10-20。选择**新增本文件**；不覆盖[091的26项背景综述](short_interval_cause_arch_091_survey.md)。097补查动态读出、上下文条件化和视频实例query近邻。下面的阅读层级按实际覆盖标明；不声称穷尽领域，也没有复现这些论文。

F*＝已查看原文关键方法段，不等于全文逐页审读；A＝原论文摘要/官方摘要级。091继承＝仅复用已存笔记，阅读层级以原表为准。

## 1. 对新假设最直接的约束

| 工作与原始来源 | 阅读层级 | 关键设计 | 对097的意义／新颖性限制 |
|---|---|---|---|
| [VrdONE, 2024](https://arxiv.org/html/2408.09408v2) | F*＋本地已核接口 | 局部/全局时序编码，关系query联合产生分类与mask | 不能宣称它不看邻域。097只检验冻结这些特征后，最后决策是否需要额外的对齐状态比较 |
| [DyFADet, 2024](https://arxiv.org/html/2407.03197v1) | F*：§3.1–3.3 | 输入相关的特征移位、权重与感受野；动态编码器和动态检测头处理不同时间范围 | **最近邻，高创新性风险。** “静态头不适应不同长度→动态头”已有明确工作，不能包装成097首创 |
| [ContextDet, 2024](https://arxiv.org/abs/2410.15279) | A | 自适应上下文聚合，组合不同范围的时间上下文 | 局部上下文改善边界并非新想法。其具体算子与097是否等价，摘要不足以排除，正式方法前需全文比对 |
| [CondInst, ECCV 2020](https://arxiv.org/abs/2003.05664) | A | 按实例生成小型动态卷积mask头 | 实例条件化和动态参数已有先例；VrdONE的query生成判别向量也不能被叫完全静态 |
| [SeqFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880547.pdf) | A：PDF摘要 | 逐帧定位/特征聚合与视频实例表示结合，产生跨帧实例mask | 单一视频实例query并非必然失败。097不能以“一段只有一个query”为数学缺陷 |
| [Mask2Former, CVPR 2022](https://arxiv.org/html/2112.01527v3) | F*：§3.1–3.2 | masked attention限制特征聚合区域，query形成mask预测 | 已有以局部证据辅助query mask的架构；点积mask成功的反例阻止“形式必然不可分”这一过强主张 |
| [Mask2Former for Video Instance Segmentation, 2021预印本](https://arxiv.org/abs/2112.10764) | A | 将mask分类框架用于视频实例分割 | 空间/时间mask路线已有跨视频扩展；但实例持续时间不等同关系成立时间，不能直接当成短关系已解决 |
| [AdaTAD++, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Agrawal_Scaling_Action_Detection_AdaTAD_with_Transformer-Enhanced_Temporal-Spatial_Adaptation_ICCV_2025_paper.html) | A：官方检索摘要 | 时空适配及分阶段训练，兼顾时间与空间信息处理 | 表征/适配也是竞争解释；097冻结编码器只能定位本读出配方的条件性效果，不排除上游不足 |

代码线索：[DyFADet作者仓库](https://github.com/yangle15/DyFADet-pytorch)、[CondInst/AdelaiDet作者仓库](https://github.com/aim-uofa/adet)、[SeqFormer作者仓库](https://github.com/wjf5203/SeqFormer)。本轮未运行或审计这些仓库；“有代码链接”不等于已完成兼容性验证。

## 2. 与已调研监督路线的分工

091已经覆盖BIG、VSGN、ActionFormer、TriDet、TemporalMaxer、DenoiseLoc、DETR/TadTR/TETAD、UVAST、GlobalPointer、半马尔可夫/分段模型、QROT等。具体版本、链接、阅读深度以091原表为准。

- 多尺度/短片段增强路线强调时间分辨率或特征汇聚；本轮不扩融合、粗尺度或采样实验。
- 端点分布、区间去噪及区间联合打分路线改变时间输出或监督；不能因为它们在其它任务有效，就忽略本项目078和E091的负结果。
- query/event对齐与分段结构路线提示输出容量和重复事件分配的重要性；097先核固定slot的可实现上界，但不追加query重选或新的多事件解码任务。
- E091只否决当前几何主配方。文献中的成功并不推翻本项目负结果；本项目负结果也不否定那些工作在原任务的贡献。

## 3. 当前能提出的差异与不能宣称的新意

097拟检验的是：**在同一实体对、相同冻结语义与视觉表征、相同输出容量下，正确时间对齐的局部状态参照，是否比同参数错位/全局参照大幅改善短关系区间；同权重撤回参照是否撤回收益。**

这是一项具体的机制识别设计，不是已经证明的research gap。动态时间头、邻域聚合、条件卷积、query mask都已有近邻。若只实现一个小MLP并提高少量命中，不能据此声称架构创新充分。

潜在差异只能来自：关系“成立时间”与实体持续存在的分离、固定实例/谓词条件下的严格对照、以及被实验支持的状态判断机制。目前这些创新性与效果仍**未验证**；DyFADet和ContextDet构成必须正面说明的相似性。

设计落点：[新原因陈述](../analysis/cause_arch_097/CAUSE_STATEMENT.md)、[架构草稿](../analysis/cause_arch_097/ARCHITECTURE.md)、[唯一决定性合同](../analysis/cause_arch_097/EXPERIMENT_CONTRACT.md)。没有把其它任务的论文指标移作本项目成绩。
