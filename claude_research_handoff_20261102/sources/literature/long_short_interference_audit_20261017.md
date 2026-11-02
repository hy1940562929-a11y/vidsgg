# 持续关系是否干扰短关系的文献证据审计

检索截止：2026年10月17日。目的：核查真实长短关系并存的规模证据，以及持续关系上下文损害短事件定位的机制证据。

**结论：在下面实际核读的材料中，尚未找到直接建立这两个严格命题的论文。但是，长短关系并存的动机、时序融合的负结果、关系惯性、背景反事实干预及局部时间聚焦，都已有明确先例。** 不能把这一结果写成“整个领域无人研究”或“我们的机制已经成立”。

## 证据标准

- **S1a，共存规模**：同一主体实例和客体实例上，短关系与另一个不同谓词的长关系真实重叠；给出长短阈值、重叠规则、样本单位、分子与分母。实体长期共存、类别平均时长、多谓词数量均不能替代该统计。
- **S1b，受损规模**：上述群体中，多少实例确实因持续关系上下文而受损。S1a计数本身不能建立S1b，需要与有效干预结果连接。
- **S2，定向机制**：保留目标短事件及必要边界邻域的证据，固定实体输入、模型、采样、时间坐标和序列长度，改变特定持续关系的上下文，测量正确短关系的原始mask或边界如何改善或恶化。排序影响单列，不能替代时间输出受损的证据。输出发生变化不等于受损；换训练方法后总体指标提高也不单独识别该原因。

“部分相关”表示涉及若干条件，**不表示部分证明了因果结论**。“未见”仅适用于列出的阅读范围。论文使用“causal”“interventional”“bias”不自动满足S2。

## 检索范围与可核验程度

检索面向2017—2026年的VidVRD、VidSGG、Dynamic SGG、PVSG，以及相关视频时刻定位工作。通过通用学术检索发现条目，再查arXiv、CVF、ECVA、PMLR、ICLR、AAAI、出版社及作者托管原文；沿重点论文的参考文献扩展。公开GitHub书目仅用于发现候选，不用于证明技术结论。

关键词组包括任务名与 duration、short-lived、persistent、enduring、overlap、co-occurrence、temporal bias、interference、temporal fusion、relation transition、context、counterfactual、intervention、oracle、ablation 的组合，另用论文全名查附录与版本。

下表记录38项候选：32项本领域及6项相邻任务。阅读范围逐项列出；这不是38篇均已全文逐字读完，也不是全领域完整普查。尚未取得的正文或附录保留缺口，不能作为“没有相关实验”的证据。版本不重复计数；匿名TFN稿不作为SegPVSG之外的独立成果。

## 本领域逐篇记录

| 工作及原始来源 | 实际核读范围 | 与规模或机制最相关的内容 | 对S1与S2的判断 |
|---|---|---|---|
| [GSTEG，CVPR 2019](https://arxiv.org/pdf/1903.10547) | 主文§3、§4.1–4.2、Table1、Fig4；补充§1–3 | 固定时间依赖与观察条件门控的结构消融；加入时间依赖时部分召回下降，但mAP上升 | 有时序结构负结果；未见指定长短并存统计或S2 |
| [Beyond Short-Term Snippet，CVPR 2020](https://pkumyd.github.io/paper/CVPR2020_VideoVRD.pdf) | §1、§3.1、§4.3–4.4、Fig1–2 | 类别平均持续时长；多尺度轨迹窗口、时空图；重点解决短切片难覆盖长事件 | 类别均值不是S1a实例比例；未见S2 |
| [Diagnosing Errors in Video Relation Detectors，BMVC 2021](https://isis-data.science.uva.nl/cgmsnoek/pub/chen-diagnosing-bmvc2021.pdf) | §3.2–3.3、Fig5–6及错误定义 | 按≤10秒、10–20秒、>20秒分组分析漏检；VidOR短组更困难。Fig6删除漏检GT得到诊断性mAP变化 | 直接支持时长相关失败；不是<3秒定义，不是长短关系共存统计，也不是实际恢复或S2 |
| [Social Fabric，ICCV 2021](https://arxiv.org/pdf/2108.08363) | 方法、实验、Fig5–6 | duration分组的P@1及长程交互建模；同特征比较 | 有时长分组效果，不能当检测R@100；未见S1/S2 |
| [VSGG-Net，Sensors 2021](https://www.mdpi.com/1424-8220/21/9/3164) | 出版社/PMC索引返回§2–4关键段；直接整页读取受限 | §4.3有约10帧的get_off漏检案例，作者联系到30帧最小窗口；另有遮挡轨迹错误 | 存在性案例和作者解释；全文覆盖不足，不作全面排除 |
| [TRACE，ICCV 2021](https://arxiv.org/pdf/2108.08121) | §3.2、§4.1、§4.3–4.5、Tables1–3；附加定性图 | 时间融合使beneath、standing on部分指标下降；作者推测空间信息受扰。每对平均多关系统计 | 有融合负结果；没有指定持续关系、短mask干预，也无S1a |
| [STTran，ICCV 2021](https://arxiv.org/html/2107.12309) | §4.1、§4.4–4.6；附录§6.2–6.4 | 报告135,484个多空间/接触谓词的人物对；训练视频打乱/反转消融 | 多谓词计数没有对应总pair分母及长短时间交叉；训练扰动不满足S2 |
| [IVRD，ACM MM 2021](https://doi.org/10.1145/3474085.3475540) | 会议PDF未取得；替代读[第一作者博士论文第3章](https://arxiv.org/html/2503.12447v1#Ch3)§3.4.2、§3.5、Table3.3 | 谓词原型及后门调整缓解频繁标签的偏置 | 语义频次干预，非持续时长干预；替代章节未见S1/S2，不能冒充会议全文审计 |
| [TSPN，2021预印本，核v2 2022](https://arxiv.org/pdf/2107.07154) | §III-E/F、§V、TableVIII | 在实体共存区间预测各类别的时间sector；改变sector数量测试量化影响 | 已有逐谓词时间支持；未见长短共存规模或S2。未用2026期刊版本替换本次核读版本 |
| [BIG，CVPR 2022](https://arxiv.org/pdf/2112.04222) | §1、§3.2、§4、Table6、附录C/Fig9 | 同pair多谓词、多次发生；三元组条件的多实例时间定位；重复次数统计 | 同谓词多实例比例不是不同长短谓词并存比例；未见S2 |
| [MVSGG，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870368.pdf) | §1、§3.2、§4.3/Table6；[补充](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870368-supp.pdf)§2、4 | 针对前后关系条件偏置进行元学习；时间偏置训练组消融 | 是关系时序偏置近邻；没有固定短事件的持续上下文干预；未见S1a |
| [VRDFormer，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_VRDFormer_End-to-End_Video_Visual_Relation_Detection_With_Transformers_CVPR_2022_paper.pdf) | 直接PDF失败；官方PDF索引§4.3、Tables3–6关键段 | recurrent query、重激活、Mean/LSTM/Self-attention时间聚合比较 | 相关结构实验；全文/附录覆盖不足，不作全面排除 |
| [DSG-DETR，WACV 2023](https://arxiv.org/html/2112.09828) | §4.2–4.3、§5.4/Tables2–4；附录A–B | 无/短/长上下文对照；对象一致性与关系建模 | 总体长上下文获益，且对象准确率也变化；未见S1/S2 |
| [TEMPURA，CVPR 2023](https://arxiv.org/html/2304.00733) | §4.3/Tables4–5；附录A–C | 类别原型memory、不确定性与对象时序一致性；HEAD/BODY/TAIL按样本频次 | 原型memory不是历史关系状态；类别频次不是时间长度；未见S1/S2 |
| [HCM，2023](https://arxiv.org/html/2307.08984) | §1、方法、§4实验与定性 | 局部细粒度与全局运动需要不同时间信息；分层关系时空图 | 已讨论长短关系差异；未见同pair交叉规模及S2 |
| [TR²，ICRA 2023](https://arxiv.org/pdf/2305.08522) | 主文§III、§IV-D/E、Eq1–2、TablesIII–IV、Fig4–5；未查配套视频 | 用文本关系变化监督视觉关系变化；根据当前和上一帧特征调节历史信息；按关系变化程度分组 | 关系变化与历史调节已有直接方法；分组不是长短重叠统计，训练模块消融不是固定目标的S2 |
| [STKET，IEEE TIP 2024](https://arxiv.org/pdf/2309.13237) | §III-B、§IV-C/D、Fig3–4/8–9；[作者书目](https://github.com/HCPLab-SYSU/STKET) | 类别实体对条件的谓词共现/转移；上下文窗口过长可降低效果 | 共现矩阵不是同实例长短时间交叉；窗口消融不是S2 |
| [TD²-Net，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28137/28277) | 方法D-Trans、Tables3–6、Fig3–4；另行附录未取得 | 选择可靠对象上下文，抑制模糊/遮挡噪声，兼顾长尾 | 上游上下文筛选与训练结构消融；未见严格S1/S2 |
| [OED，CVPR 2024](https://arxiv.org/html/2405.16925) | §3.4、§4.3–4.4/Tables4–5；独立附录未核 | 参考query逐步筛选；GT参考query oracle与实际筛选比较 | 支持参考质量重要，oracle不等于持续关系上下文；未见短mask干预 |
| TPT，IEEE TMM 2024，[正式DOI](https://doi.org/10.1109/TMM.2023.3283879) | 仅取得[作者机构摘要](https://research.polyu.edu.hk/en/publications/end-to-end-video-scene-graph-generation-with-temporal-propagation/)与[代码说明](https://github.com/gitzyong812/TPT_TMM23)，全文未取得 | query propagation关联实体、temporal dynamics encoder读取实体历史、relation propagation增强关系一致性 | 摘要未报告S1/S2，不据此推断正文不存在；2023年在线发表、2024年卷期 |
| [VrdONE，ACM MM 2024](https://arxiv.org/html/2408.09408v2) | 引言、Fig3、§3、主文消融、附录D/Tables8–10 | 引言已举in front of长期成立与shake hand with短暂共存；Fig3为类别平均时长；局部到全局、联合类别与mask | 已有相同共存动机，但无S1a总体计数；窗口/长度/输出形式消融不是S2 |
| [Motion-aware Contrastive Learning，AAAI 2025](https://arxiv.org/html/2412.07160v3) | 方法、实验、消融及定性；无独立附录 | 静态偏好、运动信息、时间打乱对比学习与动态关系召回 | 打乱改变目标动作证据，不能识别特定持续背景作用；未见S1a |
| [NL-VSGG，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/2827d59323b37dc46df8475f1b2ffd00-Paper-Conference.pdf) | 正式23页PDF；正文§2–3、附录H/I、超参及局限 | 不同动作时长导致caption-frame对齐困难；弱监督伪标签对齐 | 是时长差异近邻；未见同pair长短共存统计或S2 |
| [DiffVSGG，CVPR 2025](https://arxiv.org/html/2503.13957v1) | §3–4及所含补充结果/失败分析 | 上一帧图预测条件化当前去噪，模块消融 | 已有历史信息更新架构；没有目标固定的指定上下文干预；未见S1a |
| [VISA，CVPR 2025](https://arxiv.org/html/2503.00548v1) | 主文方法/§4；记忆理论、消融、失败例及PVSG设置 | 记忆更新与适应速度，更新率消融 | 记忆适应与本问题相关，但未见短mask损害S2及S1a |
| [UNO，WACV 2026](https://arxiv.org/html/2509.06165v2) | v2主文方法、实验、消融；补充只有摘录 | 对象一致性和动态关系slots；一致性消融并非所有TopK指标都改善 | 不能泛化为实体一致性一定有益或一定压制短关系；未见S1/S2 |
| [SegPVSG / TFN，ICML 2026](https://proceedings.mlr.press/v306/li26bx.html) | [正式18页PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf)§3–4、附录A–D、Tables1–16 | 时间区间聚焦；同pair长短案例；附录A.1允许稳定长关系背景上的局部事件生成 | 最近邻之一；长期阈值是生成规则，未见自然数据S1a统计或固定目标的S2 |
| [TraSeR / SVG2，2026预印本](https://arxiv.org/html/2602.23543v3) | v3正文§3–5、附录B/E | Table6全轨迹压缩单独使用时关系召回下降，局部时间窗口恢复 | 全局压缩的架构负结果，非具体长谓词的干预；GT轨迹/开放语义匹配与VrdONE协议不同 |
| [PAWS，2026预印本](https://arxiv.org/html/2603.21559v1) | §3–4及附录A/C/D/E.1；正式录用状态未另核 | 非交互pair污染上下文；按非交互比例分组，PAM在高干扰组收益更大 | 不同pair间干扰，非同pair不同长短谓词；分组收益非固定样本增删干扰 |
| [FReMuRe，2026预印本](https://arxiv.org/html/2604.17298v1) | §2–3、消融表 | 频次感知分支与梯度冲突动机；关注head/tail类别 | 高频谓词不是长持续时间谓词；未见S1/S2 |
| [SceneGraphVLM，2026预印本](https://arxiv.org/html/2605.13667v1) | 主文方法、结果表、§5.2和局限 | 前帧GT图与预测图比较，研究错误传播 | 改变整张历史图质量，不是固定轨迹的短事件定位干预；未见S1a |
| [RTAformer，2026-10-14预印本](https://arxiv.org/html/2610.20091v1) | §3–4、Tables2–4、§5、附录S1.3/TableS4及S2；[作者页称ACCV2026](https://limgeuntaekk.github.io/) | 明确提出持续实体间的关系语义惯性。AG有35,523/105,356个持续pair邻接满足谓词切换规则 | 直接提供切换群体规模与方法消融；切换不是长短并存，未见固定短目标的持续谓词干预 |

## 相邻任务的干预证据

这六篇研究文本查询下的视频时刻定位，不直接产生VidOR中带实体轨迹的关系检测结果。

| 工作及来源 | 原文定位 | 实际干预与控制 | 与S2的差别 |
|---|---|---|---|
| [DCM，SIGIR 2021](https://arxiv.org/pdf/2106.01534) | 方法与§4.1.3 | 内容/位置解耦、后门调整；在视频开头插入随机片段进行OOD评价 | 保留目标内容，却改变总长度和目标时间位置；检验位置偏置 |
| [IVG-DCL，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Nan_Interventional_Video_Grounding_With_Dual_Contrastive_Learning_CVPR_2021_paper.pdf) | 方法、实验与消融 | 语义混杂代理的后门调整；区间内外对比学习 | 模型化因果调整，未见指定持续关系的输入干预 |
| [TD-DETR，ICCV 2025，核arXiv v2](https://arxiv.org/html/2501.07305v2) | §3.1、§4.3、附录S4.2/TableS5 | 除训练合成外，确实构造替换背景的验证集；还做遮掉目标的虚假定位测试 | 目标保留的背景替换已有先例，但包含抽样/拼接与GT坐标更新；无持续谓词定义、短事件分层 |
| [CICR，ACM MM 2025，核v2](https://arxiv.org/html/2507.04958v2) | 方法、实验与分布外测试 | 文本后门调整和视觉反事实计算分支 | 对模型内部计算路径作反事实，不是视频中持续谓词上下文的定向替换 |
| [CVA，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_CVA_Context-aware_Video-text_Alignment_for_Video_Temporal_Grounding_CVPR_2026_paper.html) | [arXiv v1](https://arxiv.org/html/2603.24934v1)§3.1/3.3、附录A.5；正式补充§5 | 训练保持目标及邻域、原位置替换背景，约束边界一致；冻结测试诊断则移除目标保留背景 | 不能说它没有干预；但测试方向与S2不同，也无同pair长短谓词划分 |
| [CAPD，2026预印本v2](https://arxiv.org/html/2609.34581v2) | 方法Counterfactual Group Importance及实验 | 冻结教师，固定位置索引/查询/输出前缀，逐时间组替换特征，用JSD衡量输出影响 | 被替换组不限于目标外；JSD无方向，影响大小不等于对正确短边界的损害 |

## 最有影响的近邻证据

**VrdONE已经提出长短关系同时存在的动机。** 引言的长期in front of与短暂shake hand with，与本项目表述非常接近。但Fig3是谓词类别平均持续时间，不是同一实体对的长短共存比例；BSA、窗口和训练长度消融也未识别“持续长谓词造成压制”。

**SegPVSG已操作稳定关系背景中的局部交互。** 正式附录A.1在寻找生成片段插入位置时，允许忽略持续时间/视频长度>0.8的稳定关系；新事件片段为5秒。这是增强策略，不能当成自然样本共存比例，也没有保留同一短目标后改变长关系上下文。Table2确有ImageNet-VidVRD上VrdONE/TFN R@100=21.58/29.18，不是VidOR或本项目复现结果。

**TraSeR已有全轨迹压缩损害关系预测的负结果。** Table6的VidOR relation recall为：无两个resampler 14.58；仅全轨迹9.04；仅窗口16.78；两者16.30。该文使用GT轨迹、开放词汇语义匹配与tIoU评估；不能直接与VrdONE检测R@100比较。它支持保留局部时间信息，但没有把下降归因到另一种长谓词。

**RTAformer直接占据了关系惯性的宽泛问题表述。** §4.3/Table4同冻结ViT骨干下，OED到RTAformer在切换帧的WC R@10为34.3→39.1，非切换帧45.9→47.6。35,523/105,356约为33.7%，统计单位是相邻标注帧中的持续实体对匹配；不是视频GT关系数，更不是短关系受长关系干扰的比例。该工作提供分组差异和架构干预，但不满足本审计的严格S2。

该文的“非切换”仅指不满足其谓词集合互斥变化规则，并非谓词完全不变；切换帧内其他未切换关系也参与帧级评价。因此，切换帧召回差距不能等同于旧关系残留率，更不能等同于短关系受长关系压制的比例。

以上R@K为各论文按其协议报告的百分制指标；论文表未给对应逐实例命中数/GT数，不能反推或补造原始分子分母。

## 需要避免的推断

1. **短关系更难 ≠ 长关系在压制它。** BMVC诊断、量化分辨率、轨迹质量均提供其他解释。
2. **长短关系共存 ≠ 共存造成损害。** S1a与S1b必须拆开。
3. **实体持续存在 ≠ 另一谓词持续成立。** 同pair、不同谓词、时间重叠需从真实标注分别验证。
4. **模型改进有效 ≠ 动机中的因果机制已被隔离。** 结构改动通常同时影响表达能力、训练和候选排序。
5. **上下文扰动改变输出 ≠ 长谓词抑制短谓词。** 干预可能改变动作证据、边界连续性、时间位置或引入分布外噪声；需要匹配对照及有方向的损害量。
6. **帧级SGG结果 ≠ VidOR时间区间检测结果。** AG标注稀疏，前后标注帧不一定连续；STTran附录还区分空间关系与接触/注意关系的主客体方向，统计时不能混淆。

## 覆盖缺口与引用边界

- 2017原始VidVRD作者PDF已找到，但本轮直接读取失败；未列入已核读矩阵。[来源](https://www-users.cse.umn.edu/~guo00109/assets/publication/mm17-shangxd.pdf)
- IVRD正式会议全文未取得；博士论文章节作为替代来源单列，不假定二者所有实验相同。
- TPT仅核对摘要与作者代码说明，不能对其全文是否包含S1/S2作排除判断。
- VRDFormer、VSGG-Net主要核读了官方索引的关键段落；OED、TD²-Net和UNO的独立补充材料覆盖不完整。
- 匿名“Temporally Aligned Relation Modeling for Panoptic Video Scene Graph Generation”稿件[入口](https://openreview.net/pdf?id=Hhr220TJFh)访问受限，不用其摘要替代SegPVSG正式稿，不重复计数。
- 本轮另发现STVRM等期刊工作，但未取得完整正文；不能将未审计工作归为“不存在证据”。
- 没有进行完整Scopus/Web of Science导出、逐条去重与全领域引用闭包；不声称穷尽所有论文。搜索结果缺失或全文不可得均不等于不存在相关实验。

## 可用于研究定位的结论

已有研究充分涉及长短时间尺度差异、多关系共存、参考上下文筛选、时序偏置和关系更新。本轮未查到的是：**同一实体对上不同持续时间谓词共存的阈值化总体统计，及特定持续关系上下文对固定短事件定位产生有方向损害的受控证据。**

这是一个尚需实证建立的具体问题。其文献空缺不能替代项目内的规模与机制验证；也不能仅通过把一般背景改名为长期关系获得新颖性。
