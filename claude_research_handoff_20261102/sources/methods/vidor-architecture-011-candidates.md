# M011：从共享证据图到关系实例与事件过程

2026-09-29，Commander整合Innovation与源码审查。状态：候选设计，效果与新颖性均待验证。执行参数以 `tasks/VIDOR-ARCHITECTURE-011/PROTOCOL.md` 为准；不将三种架构合并为一个未经消融的大模型。

## 设计依据

M010固定32视频中，full与class_only均新增3/丢失2条Top100命中；五条变化候选的几何完全相同，mask_only没有指标变化。梯度审查没有发现错误detach；BCE对错误且饱和logit并非普遍无纠错梯度。共享谓词并集证据、全局标量门控、冻结特征及GT-pair到检测pair的域差仍是竞争解释。

因此首先验证实例条件区间学习是否能产生分类校准之外的几何修复。现有证据不支持直接宣称历史关系污染、参与者切换或全局门控就是根因。

## A：实例条件的结构化起止区间（本轮实现）

目标：实体输入可行、正确谓词存在，但关系时间范围偏宽、偏窄或偏移的实例。短关系、同谓词分离事件可能落入此类，但时长本身不是已确认病因。

冻结原FPN特征F、分类G和mask L。将256维F逐位置LayerNorm、Linear投影到64维并GELU。原softmask池化形成前景代理u；maxpool9扩张softmask后减去原mask，形成邻近时间环并池化v。空环令v=0并提供缺失标志。邻近环不是所有谓词的负例，也不是物理背景。

对query q、谓词p、时间t，拼接H_t、u、v、u-v、16维谓词embedding、softmask、两个锚点相对坐标、有效时间归一化坐标和背景缺失标志，共277维。共享MLP 277→64→2输出起点/终点logit残差，末层零初始化。约35,682参数，以实际计数为准。代理token不是原decoder隐藏query。

原mask的首末正点给出s0/e0及距离先验S0/E0。联合分布为

`P(s,e) = exp(S[s]+E[e]-logZ), s<=e`。

使用前缀logsumexp求logZ、前缀最大值求合法MAP，均为O(T)，避免T²区间矩阵。允许单点，然后经过原stride4、offset、最短5帧过滤。原G、Top6和视频Top200不变。零初始化须实际复现原最终输出，训练后可产生或删除原过滤候选，需记录。

训练使用原G/L冻结的Hungarian分配与逐GT实例端点。crop边缘端点保守视为未知，单边已知时对另一端求边际，双端未知不贡献NLL但保留评价分母。未匹配query的原Top6用固定0.1联合分布KL保留原锚点分布。具体数学及预算见协议。

两版本同初始化、顺序、预算：A-local使用实际H_t；A-pooled仅将MLP逐位置H替换为有效均值广播，u/v等仍从原序列形成。该对照移除新增头读取逐点视觉差异的能力，不是移除全部时间信息。

支持条件：local超过pooled，固定分类下出现同候选时间几何恢复，目标组净收益且原正确关系损失不抵消收益。若仅排序变化、仅统一裁剪、local不超过pooled或GT-pair开发集改善而检测退化，就不能把实例局部证据机制写成成立。

约束：未解决原Q9容量、stride4采样、Top6以外谓词、实体候选缺失。原错误mask也可能产生错误原型，多个同谓词事件可能吸走端点。这些均应检查。

[TriDet](https://arxiv.org/abs/2303.07347)已有边界分布；[TCANet](https://arxiv.org/abs/2103.13141)已有局部/全局边界细化；[BMN](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html)已有边界匹配；[SoLa](https://arxiv.org/abs/2211.06023)已有冻结特征轻量适配。待验证差异是关系实例前景/邻近时段条件能否区分共享谓词证据中的实例归属，而不是首次使用这些组件。

## B：关系开始、续存与结束的事件过程（后备架构，未训练）

结构改变：不再每个query给一个mask，而对每个pair/predicate预测可反复OFF→ON→OFF的过程。不同谓词独立保留并存能力；从F预测开始a、结束b、内部支持c，区间势能 `a[s]+b[e]+sum(c[s:e])`，以二状态或半马尔可夫模型学习与解码多个事件。

目标：分离事件被连接、结束后错误续存、短事件被连续支持吞没。可证伪预测是跨GT间隔的错误激活减少、事件数和结束位置改善，同时不把噪声过度切分。

理论上事件数不绑定Q9，但现有缓存继承>9关系过滤，不能用它验证突破Q9容量。需重新覆盖该部分训练数据，固定事件评分、未知区域处理与长度偏好后才能执行。不能运行后挑最有利长度归一化。

[ASRF](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html)已有分类/边界分支；[Segmental Spatiotemporal CNN](https://arxiv.org/abs/1602.02995)已使用半马尔可夫分段。可能差异需要落实在关系并存、实体身份条件及显式终止的具体机制与证据上。

## C：跨实体对的参与者状态图（NOT_READY_FOR_INNOVATION）

结构改变：联合读取同一主体、谓词、时间下全部候选客体及null，预测各关系边保留、结束、新建。允许多个客体同时参与，不将watch/next_to硬编码为唯一客体。

目标：物体仍存在，但关系参与者已切换，旧关系没有撤销。当前009/010结果尚未证实该机制占比或重要性。首先需同输入下审计切换后的撤销延迟及真实并行关系，再与独立边预测对照。

现有pair独立裁剪、独立采样且有数量上限，缺席客体可能只是没被抽到；不适合直接构成同步候选集合。需要完整时间对齐的实体身份和候选关系覆盖。现在硬做跨pair softmax会混淆缓存缺失与无关系。

[TPT](https://research.polyu.edu.hk/en/publications/end-to-end-video-scene-graph-generation-with-temporal-propagation/)已有身份传播与关系时序建模。允许并行关系的主动撤销是否形成实质差异，仍需进一步核查。

## 本轮决定

只执行A-local和A-pooled，每个2000步、900秒上限，同固定32train+8adapter-dev；固定32检测视频仅作已曝光探索评价。本轮不自动扩64/835、不增加按验证结果挑选的变体。保存全部成功和失败，完成后按真实结果更新证据，不自动采用为主方法。

## 2026-09-29 实验后补记（保留上面的训前设计）

A两版已完成真实训练与固定32评估。local恢复9条C但总体新增11/损失29，mAP下降；pooled恢复4条C、新增4/损失3，仅微小正向点估计。local未超过pooled，当前实现的逐位置视觉收益不成立。B/C仍未训练。完整证据：`handoff/incoming/VIDOR-ARCHITECTURE-011/REPORT.md`。

“Top6以外谓词”约束指固定query不能产生该query原Top6外谓词；不意味着原26条诊断桶全部不可达，因为别的query可能已经有正确谓词但跨度不匹配。原120桶也可能通过修好另一个高分候选恢复。桶定义和数量不变，此处修正其能力解释。
