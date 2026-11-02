# 区间质量与关系评分：最近邻核查

日期2026-10-05。Direction独立只读本地报告并检索一级来源，Commander按其返回整合落盘。状态NOVELTY_UNCERTAIN；018尚无结果。本记录是文献证据，不是本项目复现结果。

结论：区间条件谓词分类、定位置信度参与关系排序均有直接先例。只给VrdONE增加IoU/quality head或换重排公式，不足以成为新的方法贡献。

## 本项目实际约束

- 016：追加68208候选，Top100新增0/损失14，Top200新增4/损失15。14条Top100原正确预测几何和分数不变，因名额竞争后移。这只是此次追加策略的已验证缺陷，不能自动推广为VrdONE原生失败的主因。
- M010：2000步class_only与full均新增3/损失2、几何不变，full未超过class_only，短关系没有净收益。时间耦合的额外价值未获支持。
- 009 R2：mask均值重排mAP约0.068242→0.071666，但Top100由139/1209降至135/1209，新增6/损失10；时间激活均值高不保证排序更可靠。
- M011：local恢复9/1047原C，但总体新增11/损失29；pooled净增1。恢复45581条过滤候选后，相对local新增2/损失2，固定C和短组不变。

事实来源：`analysis/native_decoding_016_results/REPORT.md`、`handoff/incoming/VIDOR-COMMON-FAILURE-009/REPORT.md`、`handoff/incoming/VIDOR-ARCHITECTURE-010/REPORT.md`、`handoff/incoming/VIDOR-ARCHITECTURE-011/REPORT.md`、`analysis/m011_fallback_20261005/REPORT.md`。

## 四个直接或近邻先例

**BIG，CVPR2022。** 三元组条件的多实例时间定位已有前景分类、边界回归和centerness分支。官方模型计算sigmoid(centerness)×sigmoid(foreground)，取各bin的最大值；评价代码将原三元组均分乘bins_probs，得到区间分数。还保留主客体共存的原跨度作为候选，并可依据grounding置信度将其分数置零。直接覆盖“谓词条件定位置信度参与关系排序”和“保留原跨度同时修正置信度”的宽泛表述。该置信度不等同于解码后区间IoU回归，也没有证明解决我们固定池的具体排序缺口。[论文§3.2/Fig.5](https://arxiv.org/pdf/2112.04222)、[官方模型530–576行附近](https://github.com/Dawn-LX/VidSGG-BIG/blob/main/models/grd_model_v5.py#L530)、[官方评分融合237–247行附近](https://github.com/Dawn-LX/VidSGG-BIG/blob/main/tools/eval_vidor.py#L237)。代码链接为检索时main，行号可能随上游更新。

**SegPVSG/TFN，ICML2026。** 先预测多个区间，转为Gaussian masks，由区间mask引导注意力和特征汇聚，再预测关系类别。以与GT区间IoU>0.5的关系作为分类正标签，推理按关系分数选择、合并区间。直接覆盖“先定位、再用区间证据重判谓词”及“区间条件关系评分”。其PVSG/VidVRD结果不等于本项目VidOR复现。[正式论文页](https://proceedings.mlr.press/v306/li26bx.html)、[论文§3.1.2–3.1.4](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf)、[官方模型](https://github.com/ticatt/SegPVSG/blob/main/models/tfn/model.py)。

**Social Fabric，ICCV2021。** 第一阶段逐时刻预测interactivityness，生成交互区间；第二阶段在区间内采样、编码实体对、预测谓词，再结合主体和客体分数。直接覆盖“对不同时间候选分别提取证据并分类”。不是显式IoU质量回归，不能说其已解决最终质量校准。[论文§4 Stage2](https://arxiv.org/pdf/2108.08363)、[作者代码](https://github.com/shanshuo/Social-Fabric)。

**TadTR，TIP2022，邻近动作检测。** 明确指出分类分数不敏感于定位质量。预测区间及上下文RoIAlign后，由质量头回归最大GT IoU，再结合类别分数排序。直接覆盖“用区间质量修正分类排序”的动机及基础方案。原头类别无关、任务是动作检测；适配VidOR应作基线，不能仅凭适配声称机制新颖。[论文§III.A/Fig.4/6](https://arxiv.org/pdf/2106.10271)、[官方实现](https://github.com/xlliu7/TadTR/blob/master/models/tadtr.py)。

## 三个不能混同的量

1. 实体对是否存在该谓词：语义成立。
2. 谓词、具体区间与两条轨迹是否共同满足官方条件：候选可匹配性。
3. 在有限名额和其他预测存在时，是否增加独特GT命中：集合中的新增价值。

普通分类主要回答第一项，质量头估计第二项；第三项依赖重复覆盖与一对一分配，单个候选质量高不等于有新增价值。

待验证问题：固定输入、候选池和输出预算后，实际区间与谓词联合决定的可匹配性，是否存在稳定、可学习且超出常规质量校准的排序信息？这不是已确定的主方向；谓词嵌入、区间池化与IoU监督本身均不是足够的创新点。

## 最低证据要求

- 018按独特GT区分原池已覆盖、追加池才覆盖、两池均未覆盖，并保留预算约束。候选数、单GT各自oracle和纯tIoU不能冒充实际恢复。
- 在同pair/predicate下提供不同区间的错序见证，同时量化重复覆盖；区分质量错误、复制分数与名额竞争。
- 仅在训练集拟合，用视频不相交开发集检查局部证据是否优于原score、mask均值、OIC、时长/类别校准。简单元数据能解释收益时，不宣称新视觉机制。
- 同输入/候选/预算比较普通IoU头、区间条件分类及class_only；完整方法必须超过这些基线，而非只超过原模型。
- 后续需在未参与反复设计的评价视频上核验总体净收益、固定C、短长组和新增/损失。native32不是独立泛化证明。

如果新增独特覆盖很小，或收益只是修复复制score/重复处理，或普通质量头足以解释全部收益，应收缩论文主张；不以工程修补包装CVPR创新。输入固定时，C中16828/25373受到当前输入缺失或几何不足约束，关系评分本身无法恢复这些样例。本文检索不构成穷尽性首创证明，也不保证此问题足以支撑CVPR论文。
