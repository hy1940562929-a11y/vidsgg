# 091 文献对齐：短事件原因与时间输出架构

核查日期：2026-10-20。范围：2004–2026，26项外部工作，另对照本项目078、011、090。本轮为调研与设计，没有复现实验。

**用户硬交付标准**：最终需实验验证短关系困难的原因，不能用失败定位或未决结论堆叠替代；架构必须先经广泛文献对齐，再直接针对该原因。本表完成文献环节，**不替代原因实验，也不证明架构有效**。

## 1. 调研得到的直接约束

1. **VrdONE已有独立事件mask、query–instance匹配和Dice监督**；BIG也已有多实例时间定位。不能声称首次把关系当事件。
2. **短事件已有直接研究**：VSGN、ASM-Loc、E²DVC分别从尺度、proposal采样、事件关注分配处理。不能把“长事件多、短事件被忽略”当作未被提出的现象。
3. **时间分布、区间几何和OT已有直接近邻**：TriDet、DisTime、DenoiseLoc、BMA均须对比。仅把softmax、W1或联合端点加到VrdONE上，新颖性不足。
4. 当前可追问的细问题是：**已有实例监督下，特定mask损失与最终区间几何之间的不一致，是否确实是本项目可干预的错误来源？** 这是待验证问题，不是调研已经证明的根因。

## 2. 检索与阅读口径

实际来源：arXiv正文/PDF、CVF、ECVA、PMLR、OpenReview、ACL Anthology、NeurIPS论文页、出版社原文及作者仓库；复用本地文献笔记，但关键判断返回原文或源码。

实际关键词族：`short action temporal localization duration imbalance`；`temporal boundary distribution optimal transport`；`mask interval supervision query instance matching`；`VrdONE BIG multi instance grounding`；`segment frame alignment UVAST`；`event equalized dense captioning`；`segmental CRF span prediction`。扩展并核对具体近邻名称。

- **F**：取得原文并核读所列方法/实验关键节，属于“全文可查、关键节已读”，不表示逐字通读所有附录。
- **A**：仅摘要/官方论文页；不据此推断未看到的实现。
- **P**：原文局部索引摘录可读，直接全文获取失败；与F分开。
- 表中“已有设计”为文献事实；“联系/缺口”为本项目解释。代码“已核到”只表示作者链接/仓库可访问，**没有安装、运行或核验可复现性**。
- 没有跨任务拼接benchmark排名。文献方法有效，不等于已解释VidOR的334/347失败。

## 3. 视频关系与query—事件对齐

| 工作、任务、来源 | 阅读级别/位置 | 已有关键设计 | 与本失败点的联系及缺口 | 代码核验 |
|---|---|---|---|---|
| **VrdONE: One-stage Video Visual Relation Detection，MM2024**；VidVRD。[原文](https://arxiv.org/html/2408.09408v2) | F：§3.3–3.5、Eq.11–13；本地训练源码及官方模型文件 | 局部到全局表示；每条关系实例的mask；类别+mask二分匹配 | 最直接基线。已有事件监督仍出现短区间不足；不能据此断言匹配、损失或融合哪一个是原因 | [作者仓库](https://github.com/lucaspk512/vrdone)；`MaskVRD`是其实现类名，本轮未确认另有同名独立论文 |
| **Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs（BIG），CVPR2022**。[原文](https://arxiv.org/pdf/2112.04222) | F：§3.3、§4.3/Table6、附录C | 三元组分类后多实例grounding，同pair同谓词可有多个事件；时间bin容量有限 | “一次语义、多次事件”已经被处理；本项目尚未证明bin/query容量是主要原因 | [作者仓库](https://github.com/Dawn-LX/VidSGG-BIG) |
| **End-to-End Object Detection with Transformers（DETR），ECCV2020**。[原文](https://arxiv.org/html/2005.12872) | F：§3.1、§4.4 | 无序实例集合、一对一匹配、背景query、实例mask扩展 | query、匹配和事件实例化的基础先例；图像检测问题不能直接作为短关系原因 | [作者仓库](https://github.com/facebookresearch/detr) |
| **Relaxed Transformer Decoders for Direct Action Proposal Generation（RTD-Net），ICCV2021**。[原文](https://arxiv.org/pdf/2102.01894) | F：§3–4、附录A.2 | 边界表示；从严格匹配到放宽top-k匹配；完整性评分 | “短事件获得太少正query”有可比较方法，但089没有训练匹配证据，不选作本轮原因 | [作者仓库](https://github.com/MCG-NJU/RTD-Action) |
| **End-to-end Temporal Action Detection with Transformer（TadTR），TIP2022，预印本2021**。[原文](https://arxiv.org/html/2106.10271) | F：§III-A/B及相关消融 | 时间可变形注意；action query；迭代区间修正；局部RoIAlign预测定位质量 | 局部证据、区间修正和分类/定位评分分工都有先例；质量评分不能创造当前库中不存在的区间 | [作者仓库](https://github.com/xlliu7/TadTR) |
| **TE-TAD: Towards Full End-to-End Temporal Action Detection via Time-Aligned Coordinate Expression，CVPR2024**。[原文](https://arxiv.org/html/2404.02405) | F：§3.3–3.4、§4 | 实际时间坐标、按sector选query、query数随视频长度调整 | 显式处理坐标长度敏感性；VrdONE是mask输出，不能直接套用其中心坐标问题作为根因 | [作者仓库](https://github.com/Dotori-HJ/TE-TAD) |
| **Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation（UVAST），ECCV2022**。[原文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950052.pdf) | F：§3.1–3.3、Table4 | 段序列预测；segment index监督帧—段注意力；时长对齐；位置区分重复动作 | 帧—事件对齐不是新机制；其有序互斥分段不能直接套到同pair多谓词重叠 | [作者仓库](https://github.com/boschresearch/UVAST) |
| **SegPVSG: Panoptic Video Scene Graph Generation via Temporal Focusing and Generative Augmentation，ICML2026**。[官方页](https://proceedings.mlr.press/v306/li26bx.html) | A：官方摘要、仓库说明；本轮PDF获取失败 | 时间聚焦定位显著片段，再识别关系；生成关系片段增强 | 已直接讨论关系时间稀疏与片段聚焦；未读全文，不能确认其监督几何或短组统计 | [作者仓库](https://github.com/ticatt/SegPVSG) |
| **VSTRD: An end-to-end video spatio-temporal relation detection transformer，Neurocomputing2025**。[出版社](https://doi.org/10.1016/j.neucom.2025.131307) | A：摘要、Highlights | 动态query联合检测、轨迹关联与关系分类；分离对象/时空编码 | 不能声称关系领域没有动态query；具体短区间训练机制未核实 | 未核到官方代码 |

## 4. 短事件、时间分布与区间监督

| 工作、任务、来源 | 阅读级别/位置 | 已有关键设计 | 对本原因和架构的约束 | 代码核验 |
|---|---|---|---|---|
| **AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos，ECCV2018**。[原文](https://arxiv.org/pdf/1807.08333) | F：§3.1–3.2、补充§8.4 | 边界预测，区间内外平均激活对比；绕开仅靠CAS阈值 | 直接区间、长度归一化并非新想法；弱监督缺边界不能移植成VrdONE的问题 | [作者仓库](https://github.com/zhengshou/AutoLoc)，旧Caffe/Python2 |
| **VSGN: Video Self-Stitching Graph Network for Temporal Action Localization，ICCV2021**。[原文](https://arxiv.org/html/2011.14598) | F：§3、损失、短动作分析 | 放大短片段与原片段拼接；跨尺度图；GIoU定位 | 短事件已有直接方法；收益混合表示、采样与正样本数。其ActivityNet XS≤30秒不等于本项目<3秒 | [作者仓库](https://github.com/coolbay/VSGN) |
| **ASM-Loc: Action-aware Segment Modeling for Weakly-Supervised Temporal Action Localization，CVPR2022**。[原文](https://arxiv.org/html/2203.15187) | F：§4.1–4.3、§5.3、附录E | proposal时长加权与CDF逆采样；段内/段间注意；伪实例监督 | 已观察短事件低置信度/漏检；严重错位proposal仍难修复。不能把CDF采样与输出事件分布混同 | [作者仓库](https://github.com/boheumd/ASM-Loc) |
| **ActionFormer: Localizing Moments of Actions with Transformers，ECCV2022**。[原文](https://arxiv.org/html/2202.07925) | F：架构、Eq.7、center sampling | 局部注意、多尺度；两侧距离；Focal+DIoU，正样本归一化 | 强反例：领域早有显式区间几何监督。091的数学观察只针对核验过的VrdONE损失 | [作者仓库](https://github.com/happyharrycn/actionformer_release) |
| **TriDet: Temporal Action Detection with Relative Boundary Modeling，CVPR2023**。[原文](https://arxiv.org/html/2303.07347) | F：§3.3–3.4、Eq.4–10 | 起止与中心偏移组合的边界分布；期望读出；IoU损失 | “分布→解析时间”已有先例；完整区间密度与端点分布不同，但区别本身不保证创新 | [作者仓库](https://github.com/dingfengshi/TriDet) |
| **TemporalMaxer: Maximize Temporal Context with Only Max Pooling for Temporal Action Localization，2023预印本**。[原文](https://arxiv.org/html/2303.09055) | F：方法、Eq.7 | 局部max pooling金字塔；距离预测；Focal+DIoU | 简单表示也可定位，不应把复杂融合当必需；不是最近一年工作 | [作者仓库](https://github.com/TuanTNG/TemporalMaxer) |
| **DenoiseLoc，ICLR2024**；语言定位/TAL。[原文](https://arxiv.org/html/2304.02934)、[正式版](https://openreview.net/pdf?id=bLpUtGyf9g) | F：§3.2–3.3 | 区间query细化；GT区间扰动训练；Hungarian、L1+GIoU；推理无GT扰动 | 事件式区间、去噪和距离监督已有；不能只换起止头。公开实现主要面向QVHighlights | [作者仓库](https://github.com/frostinassiky/denoiseloc) |
| **DisTime，ICCV2025**；视频LLM时间定位/密集描述。[原文](https://arxiv.org/html/2505.24329v1) | F：§3.2–3.4、Eq.4–5 | 起止时间bin分布；anchor加权期望；distribution focal+1D IoU；时间再编码 | 归一化分布、期望读出高度近邻。变量是边界而非事件内部密度；换任务不自动产生新颖性 | [作者仓库](https://github.com/josephzpng/DisTime) |
| **Event-Equalized Dense Video Captioning（E²DVC），CVPR2025**。[官方页](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Event-Equalized_Dense_Video_Captioning_CVPR_2025_paper.html) | P：官方摘要+原PDF§3.4–3.5索引摘录；直接PDF获取失败 | 聚类伪事件初始化query参考点、event-enhanced encoder；区间/置信度及Hungarian、gIoU、caption等训练 | 作者明确提出长事件获更多关注、短事件欠捕获；与最初“长压短”故事高度重合。不是单位内部密度/W1，也不能作为本项目因果证明 | 本次未核到；不等于不存在 |
| **Boundary Measure Alignment via Optimal Transport for Temporal Action Detection（BMA），CMC2026，9月15日发表**。[原文](https://www.techscience.com/cmc/v89n2/68788/html) | F：§3.2–3.5、§4.5、限制 | 概率导数形成归一化起止measure，GT边界高斯化，entropic OT；训练辅助，原推理不变 | **最接近监督几何论证**；其全视频边界聚合区别于本草案匹配实例内部密度，但原文已提instance-aware方向，创新风险高 | 原文资料向作者索取；未核到官方仓库 |
| **Boosting Point-Supervised Temporal Action Localization through Integrating Query Reformation and Optimal Transport（QROT），CVPR2025**。[原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Boosting_Point-Supervised_Temporal_Action_Localization_through_Integrating_Query_Reformation_and_CVPR_2025_paper.pdf) | A：本轮仅原文摘要 | 点标签构造query，OT伪标签辅助proposal匹配 | query+OT已有；OT用于分配不等于时间质量位移。本轮不据摘要评价细节 | 未核到 |

## 5. 跨任务基础：不能重新发明已有区间预测

| 工作/来源 | 阅读程度 | 已有思想及迁移边界 |
|---|---|---|
| **Semi-Markov Conditional Random Fields for Information Extraction，NeurIPS2004**，[论文页](https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html) | A：官方摘要 | 直接给可变长度段赋标签、段级特征；“由点改段”不是新思想。互斥序列分段不同于重叠关系集合。 |
| **Segmental Recurrent Neural Networks，ICLR2016**，[原文](https://arxiv.org/pdf/1511.06018) | F：§2–3、Eq.1–4 | 标签和时长的联合分布、段表示及动态规划；不能把“事件+持续时间”宣称首创。 |
| **Segment-Level Sequence Modeling using Gated Recursive Semi-Markov Conditional Random Fields，ACL2016**，[原文](https://aclanthology.org/P16-1134.pdf) | F：§2–3，Eq.3–4 | 共享递归段表示与半马尔可夫结构；已有段内属性/长度建模，非本项目损失原因证据。 |
| **Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition，2022预印本**，[原文](https://arxiv.org/pdf/2208.03054) | F：§3.2–3.4 | 起止对打分、相对位置、多标签区间损失；“联合起止+区间比较”已有人做，也与本项目078有重叠。 |
| **Multitask Learning with CTC and Segmental CRF for Speech Recognition，Interspeech2017**，[原文页](https://arxiv.org/abs/1702.06378) | A：摘要、版本记录 | 同一编码器比较/结合逐帧和段级目标，提供控制表示后研究监督的先例；不能套用其结果。 |
| **The Lovász-Softmax Loss，CVPR2018**，[官方页](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html) | A：官方摘要、作者README；PDF本轮获取失败 | 训练代理与IoU不一致是老问题；mask IoU仍不同于解码后连续区间的时间几何。[作者实现](https://github.com/bermanmaxim/LovaszSoftmax)已核到。 |

## 6. 对090的实质修订

保留“每事件时间质量分布”作为候选变量，但**不把事件归一化、分布、OT分别当贡献**。撤下矩读出作为优先设计：远端少量质量对方差产生距离平方放大，可再次拉长短区间。091改为同一W1几何下的合法区间投影，见[架构草稿](../analysis/cause_arch_091/ARCHITECTURE.md)。

最接近的对照必须包括：VrdONE实例mask（直接基线）、BMA几何监督（强近邻）、TriDet/DisTime边界分布（输出变量近邻）、E²DVC事件等权（问题动机近邻）。它们并不都能直接在VidOR上无改动运行；本轮只是文献比较，没有跨任务效果排名。

本地负结果保留：[078](../analysis/span_readout_078/REPORT.md)的联合端点CE曾出现总体588/4739→577/4739，新增4、损失15；不能再次把“换起止头”当未经尝试的新解法。[011备选B](../methods/vidor-architecture-011-candidates.md)早已提出未训练的半马尔可夫事件过程，也不是本轮新发现。

**新颖性判决：`HIGH RISK / NOVELTY_UNCERTAIN`。** 目前值得验证的差异在于“限定失败群体的可操纵监督几何证据＋完整事件密度与区间决策闭环”，不能宣称已取得独特机制贡献。没有查遍所有论文；摘要级资料未参与具体公式优劣判断。

## 7. 硬交付自检

| 要求 | 本轮状态 |
|---|---|
| 广泛调研后再提架构 | 已覆盖26项外部工作；按F/A/P标记，近邻与反证保留 |
| 以实验确认完整原因 | **尚未达到**；本表没有新增因果结果，唯一决定性实验写入CAUSE_STATEMENT |
| 方案直接对准原因，不开旁支 | 只保留监督—决策时间几何一致性这一条；不扩融合、gate、query搜索或拆hull |
| 创新性/性能已验证 | **未达到**；文献重合显著，仍需实证，不能称已可投稿 |
