# 时间支持与候选选择：直接先例核验

日期：2026-10-05。Direction 只读调研，Commander 整理保存。文献已核验；本项目新机制与创新性仍待实验。作者报告数字均不是本项目复现。

## 必须分开的三个问题

1. 固定实体轨迹后，全部原始预测中有没有匹配区间？这是候选支持缺失。
2. 网格能表示区间，但预测或解码未产生区间，原因是什么？这是时间激活与解码问题。
3. 好候选已存在，却没进入正式 TopK 或在匹配中被排除？这是评分、重复与竞争问题。

GT27 的 2,563/2,573 网格可行、1,731/2,573 默认 hull 可用，支持检查第2项，尚不支持历史关系干扰的解释。M011 回退恢复45,581候选但Top100只新增2/损失2，不能把候选数视为恢复GT数。

## 最接近的工作

| 工作 | 已有内容及作者报告 | 对本项目的约束 |
| --- | --- | --- |
| VrdONE，MM 2024 | 局部注意力、FPN、多query、实例时间mask；附录比较边界回归、语义分割、实例分割，VidOR R@100为13.85、13.78、14.21（论文百分制）。 | 局部、多尺度、多实例时间建模不能作为首次提出。需对照其已有语义分割变体。 |
| BIG，CVPR 2022 | 以含谓词的三元组查询做多实例时间grounding；多个bin预测前景、边界及置信度。VidOR BIG-C→BIG，mAP8.29→8.54、R@1009.65→10.04。 | 谓词条件定位、同pair/predicate多段事件都有直接先例；额外I3D输入不是纯定位改动。 |
| TSPN，2021预印本／2022修订 | 将共现区间分成sector，预测类别及类别相关sector占据。v1在VidVRD报告sector4→16时mAP9.79→18.90。 | 逐谓词时间支持已有先例；需注明版本，不能视为本项目短关系分组结果。 |
| SegPVSG／TempFocusNet，ICML 2026 | query→中心/宽度→Gaussian mask→聚焦区间内关系分类；高分同谓词区间合并；另报无TopK召回。ImageNet-VidVRD上作者报告VrdONE/TFN R@10021.58/29.18。 | “先定位再识别”已有直接方法，且并非仅PVSG。不等于我们VidOR实验收益。 |
| BMN，ICCV 2019 | start×duration候选图，区间内外采样，IoU质量监督，结合边界概率和Soft-NMS。 | 区间完整性、质量学习及质量排名已有先例。 |
| TCANet，CVPR 2021 | 局部/全局特征，边界附近起止偏移与内部中心/长度精修，同时学习IoU。 | “区间内外证据＋边界精修＋质量头”不能独立承担新意。 |
| TadTR，TIP 2022 | 扩大预测区间至1.5倍后RoIAlign，预测IoU actionness并修正分类排名；HACS作者报告28.51→30.83 mAP。 | 固定候选池质量排序的直接简单基线；增益不能外推VidOR。 |
| TriDet，CVPR 2023 | FPN、边界响应与相对偏移分布、期望解码；现行官方代码起止head输出num_classes。THUMOS14作者报告68.3→69.3 mAP。 | 类别条件边界、尺度/时长、分布式边界已有先例；可检验同后验下不同决策规则，但不能预设其有效。 |

原始来源：

- VrdONE：[论文 §3.4–3.5／附录D](https://arxiv.org/html/2408.09408v2)，[官方代码](https://github.com/lucaspk512/vrdone/blob/main/models/maskvrd.py)。
- BIG：[论文](https://arxiv.org/pdf/2112.04222)，[官方代码](https://github.com/Dawn-LX/VidSGG-BIG)。
- TSPN：[v1论文](https://arxiv.org/pdf/2107.07154v1)，[作者代码](https://github.com/sangminwoo/Temporal-Span-Proposal-Network-VidVRD)。
- SegPVSG：[PMLR正式出版记录](https://proceedings.mlr.press/v306/li26bx.html)，[论文](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf)，[模型](https://github.com/ticatt/SegPVSG/blob/main/models/tfn/model.py)，[注意力实现](https://github.com/ticatt/SegPVSG/blob/main/models/tfn/attention.py)。正文Gaussian作用于softmax前，当前代码softmax后乘权再归一；复现必须注明采用版本。
- BMN：[论文](https://arxiv.org/pdf/1907.09702)，[Paddle维护实现](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/paddlevideo/modeling/backbones/bmn.py)，[质量损失](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/paddlevideo/modeling/losses/bmn_loss.py)。维护实现不冒充当年冻结代码。
- TCANet：[论文](https://arxiv.org/pdf/2103.13141)，[官方TBR实现](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch/blob/main/tbr.py)。
- TadTR：[论文](https://arxiv.org/pdf/2106.10271)，[官方代码](https://github.com/xlliu7/TadTR/blob/master/models/tadtr.py)。
- TriDet：[论文](https://arxiv.org/pdf/2303.07347)，[现行官方实现](https://github.com/dingfengshi/TriDet/blob/master/libs/modeling/meta_archs.py)。

## 相对效用不是质量排序的同义词

绝对质量 q(I) 是候选与兼容GT的最大IoU。编辑收益 Δq=q(I_a)−q(I_0) 比较同一候选的改动与原区间。集合边际覆盖U(c|S)又是另一个受重复、TopK和匹配影响的量。

同一候选中 argmax_a[q(I_a)−q(I_0)]=argmax_a q(I_a)。减去共同no-op分数或套单调tanh均不改变动作排序。因此M012潜在区别只能落在训练目标、误差校准、保留原结果的决策门槛等，而不是“相对增益”的名称。必须与同特征、同动作集、同参数预算的absolute-IoU head采用相同差分/门槛比较。

## 当前可检验的窄问题

实体输入固定后，漏检区间能否由既有query响应通过标量阈值表达？若不能，究竟是数值饱和、多时段共同激活被hull合并，还是缺少可区分的目标峰？这是015诊断的问题，不预设哪一种机制成立。

候选池没有好区间，才继续检查支持表达；已有好区间但排名不足，才检查固定池质量排序。之后必须比较已有简单机制和完整检测收益。目标组改善但总体下降、只有增加候选数量才改善、只有GT oracle改善，都不能作为有效主方法。

## 暂不能写的结论

- 不能说谓词条件时间支持、时长分支、IoU质量头或边界精修是首次提出。
- 不能从GT27非随机子集推断完整VidOR规模。
- 不能把任意GT阈值、动作或query选择的上界写成模型恢复数。
- 不能说M012已有效；当前仍未训练，M010/M011及统一阈值的负结果必须保留。

## 015B启动前补查：多阈值、局部峰与连通段已有先例

| 工作 | 已有机制 | 对015B之后方法设计的约束 |
| --- | --- | --- |
| SSN/TAG，ICCV2017 | 将actionness补序列视为一维地形，多水位watershed得到basins，再依据前景占比合并邻近basins；多阈值并集与NMS。 | 局部峰、多阈值分组与受控合并本身不新。 |
| AutoLoc，ECCV2018 | 高阈值过拆、低阈值引入背景/合并实例；学习中心/长度，以区间内外激活差作OIC损失。补充Fig5/6展示两类阈值错误。 | 标量阈值不能兼顾完整性与分离已有论述；其特定阈值案例不等于本项目穷举全部阈值的不可达证据。 |
| BaS-Net，AAAI2020 | 多阈值CAS→每组连续片段形成proposal→OIC评分→NMS；论文阈值0–0.5步长0.025。 | 多阈值CC/OIC是直接现成基线。 |
| CoLA，CVPR2021 | 多阈值T-CAS、连续片段与NMS；训练用膨胀/腐蚀选边界困难片段做对比学习。 | 改善边界附近响应也有先例，需证明额外关系条件的重要性。 |
| CTAP，ECCV2018 | 低质量actionness会漏掉可行段；PATE选择滑窗来补充分组遗漏，再统一排序和调整边界。 | 网格可行而激活分组不可行、以独立候选补充分组也有近邻先例。 |

原始来源：

- [SSN论文§5与Fig3](https://arxiv.org/pdf/1704.06228)，[作者项目](https://yjxiong.me/others/ssn/)。
- [AutoLoc论文及补充§8.1](https://arxiv.org/pdf/1807.08333)，[官方代码](https://github.com/zhengshou/AutoLoc)。
- [BaS-Net论文§3.4/4.1](https://arxiv.org/pdf/1911.09963)，[官方推理](https://github.com/Pilhyeon/BaSNet-pytorch/blob/master/test.py)。
- [CoLA论文§3.3/3.5/4.2](https://arxiv.org/pdf/2103.16392)，[官方代码](https://github.com/zhang-can/CoLA)。
- [CTAP论文§1/3](https://arxiv.org/pdf/1807.04821)，[官方代码](https://github.com/jiyanggao/CTAP)。

上述文献不证明已解决当前VidOR样例，但否定“多阈值拆段本身新颖”的表述。若分量oracle补回很多，而OIC、原query分数、简单质量头不能可靠选中，可继续检验“同一query的多个局部支持中，哪个片段具有正确关系语义”。若已有简单基线已经解决，则如实报告，不强行另起模块。

全hull族H和分量族CC不嵌套；GT跨内部低谷时H可正确而CC失败。只有H∪CC上界保证保留H，且该并集仍不能替代固定预算下的实际收益。
