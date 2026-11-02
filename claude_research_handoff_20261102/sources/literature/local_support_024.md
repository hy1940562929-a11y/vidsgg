# 024：局部时间支持候选问题的近邻工作

2026-10-08。Direction只读检索返回，Commander整理。文献与本项目实验分开；没有把作者报告性能当成本项目复现。本轮未穷尽检索，未验证任何新方法。结论：**宽泛的“事件专属局部时间支持，保留全局语义”已有直接先例，novelty风险HIGH。**

| 一级来源与版本 | 已读方法内容 | 对当前叙事的约束 |
|---|---|---|
| [VrdONE，MM2024，arXiv v2](https://arxiv.org/html/2408.09408v2)，§3.1、3.3–3.5、Appendix D；[官方predictor](https://github.com/lucaspk512/vrdone/blob/main/models/predictor.py) | BSA/关系编码器使用局部注意力及多尺度结构；实例decoder为全局self/cross-attention，各query分别产生类别与mask，联合匹配监督。附录比较边界回归、语义分割、实例分割。 | 不能称首次联合分类定位、多尺度或多实例时间支持。全局访问不等于query权重相同；023给九个query统一限制是实验设定，不能写成原模型没有query区分。 |
| [BIG，CVPR2022](https://arxiv.org/pdf/2112.04222)，§3.1–3.3、Fig.5；[官方代码](https://github.com/Dawn-LX/VidSGG-BIG) | 先分类与实体连接，再用SPO语言query、I3D及实体共存范围定位；多个bin输出前景、边界和置信度，支持同一三元组多次发生。 | 全局识别后谓词条件多实例定位、分类定位分阶段已有先例；不同输入及完整系统效果不能归因于单一局部操作。 |
| [SegPVSG/TFN，ICML2026正式记录](https://proceedings.mlr.press/v306/li26bx.html)，[PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf) §1、3.1.2–3.1.4、4.1–4.3；[官方模型](https://github.com/ticatt/SegPVSG/blob/main/models/tfn/model.py) | temporal queries读取全局pair特征，各自预测中心、宽度与Gaussian mask，引导区间专属注意力再多标签分类；diversity loss鼓励不同时间模式。包含OpenPVSG及ImageNet-VidVRD。保留相关全局上下文是作者明确主张。 | 最直接近邻：不同query不同范围、局部聚焦结合全局上下文、先定位后识别均已覆盖。所读部分未给出024这种对其他原正确事件的损失追踪，这仅是未见相关实验，不是已证实研究空白。 |
| [TSPN / What and When to Look?，arXiv v2](https://arxiv.org/pdf/2107.07154v2)，§III.E–F、V；[官方代码](https://github.com/sangminwoo/Temporal-Span-Proposal-Network-VidVRD) | 划分实体共存跨度为sectors，预测谓词×sector占据，表达多关系时间支持。粒度依赖共存时长与sector数。 | 逐谓词时间支持已有先例。版本uncertainty：仓库当前标ESWA2025，本次技术阅读仍为arXiv v2；不得以模板页眉认定期刊。 |
| [TadTR，TIP2022](https://arxiv.org/pdf/2106.10271)，§III.A、Fig.3–4；[官方代码](https://github.com/xlliu7/TadTR) | action query以参考时间点读取稀疏时间位置，迭代精修区间及参考点；区间对齐特征预测actionness参与排序。任务为TAD。 | query-specific时间读取、局部精修及定位质量排序均已有先例；跨任务迁移这些操作不自动构成创新。 |

Direction实际读取TFN方法正文与官方模型；Commander又核对正式PMLR记录和VrdONE正文入口。TFN正文Eq.6与旧项目笔记所提示的代码归一化差异仍须在复现时冻结版本核查，不在此推断效果。

当前项目的023统一限制结果为目标5/177→4/177、全体855/2573→794/2573；Gout目标5/177→18/177但全体855/2573→818/2573。GT辅助混合与上下文前向是不同操作。024可以解释这些操作的副作用，不能据此宣布局部读取必有效。

候选问题状态为 `NEEDS_VERIFICATION / NOVELTY_UNCERTAIN`，非正式主方向：在固定原预测上，定位改善与其他事件损失分别来自同一query还是不同query？原候选资格、分数及排名是否变化？Direction建议先用既有缓存分解，Commander已纳入024。结果见 `analysis/decoder_loss_024/REPORT.md`，以实际reviewed计数为准。

若仅证明人为统一限制造成副作用，应按此范围结束；不据此拼装新局部注意力架构。仅限制选定query的真实前向将需要新的冻结协议，但目前未获机制/训练准入，不启动GPU。
