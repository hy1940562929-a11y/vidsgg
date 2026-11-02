# 075：固定候选的事件内正负平衡与时间排序小试

状态：执行前设计，尚无本轮结果。用户于2026-10-14明确授权在AutoDL端口16241进行此前设计的实验。仅在私人新目录运行；不修改基线源码、环境、原checkpoint或数据。不是正式主方向/主方法采用。

## 问题和范围

073已确认长共存短关系存在固定Top100时间输出缺口；69/325是逐GT网格oracle机会，6/325是原query阈值机会，并非实际恢复。074独立复用缓存进行原始响应错误分型。本实验检验：固定表示与候选预算时，事件内正负时间监督的平衡和相对排序能否带来真实时间匹配净收益。

原损失已按匹配实例平均，不能称原实现完全没有事件均衡。042/043已检查padding/有效长度归一化，043没有正式收益；075的新增因素为事件内部正负分别平均及时间排序。049普通端点头与058路径组合的负结果保留。

## 数据、基线、预算

- 使用043封存训练缓存：32个官方training来源视频，350个分段实体对、563个目标；8个training来源dev视频、89pair、169目标仅作诊断。源SHA绑定049 CACHE_INVENTORY。
- 缓存继承旧导出配额与loader筛选；017已记录13个截断保留目标。本次不冒充全训练分布。
- 官方检测评价复用047/048各64个已曝光验证视频，共4739条GT；与train/dev视频集合不相交。不是独立泛化验证。
- 附加连接073固定331个长共存短关系目标（325个C）、长共存持续关系；同时保留原049 T/U/V历史群体。无GT推理分支选择。
- 三臂同初始化、同mask_embed参数、同训练batch次序和原基线固定Hungarian匹配，500步，AdamW lr=1e-5、weight_decay=.05、clip_norm=1。seed固定75，不挑checkpoint，不补跑seed/步数救门槛。
- 原编码器、FPN、query输入、分类结果、候选身份、分数、排名和保存候选数固定，仅更新mask_embed；原始未训练模型另作baseline。
- 一次基线forward供三臂共用。总GPU流程硬上限1800秒；超时记执行未完成，不当作假设被否定。

## 三臂目标

设逐点focal为 BCE(z,y)*(1-p_t)^2*(0.25*y+0.75*(1-y))。

1. original：保持原focal按补齐长度平均和原Dice，按全部匹配实例平均。
2. balanced：每个实例分别对有效正点和有效负点的focal取均值，再对存在的两类取等权平均；若仅一类存在则取该类均值。保留原alpha/gamma。训练初始化全train四层实例等权计算k_B=mean(original_focal)/mean(balanced_focal)，冻结系数，原Dice不变。
3. ordered：对ranking可监督实例，将focal部分替换为0.5*k_B*balanced_focal + 0.5*k_R*rank_loss；不可监督实例保持balanced。rank_loss=mean softplus(1-z_positive+z_negative)。k_R仅用初始化训练eligible实例计算，使其rank项初始平均值等于对应k_B*balanced_focal平均值。原Dice不变。原loss_mask/loss_dice权重保持不变。

初始损失均值匹配只控制初始标量尺度，不等于梯度范数或全训练过程完全相等；保存各臂初始总梯度范数、两两方向余弦、500步梯度裁剪次数与分项损失，不作超出该控制的机制宣称。

## 排序监督的支持域

- positive为当前固定匹配事件有效内部点；边界向内保护1个采样点。太短而无内部点者跳过新增ranking，不删除原监督。
- negative必须在有效输入域，位于同一pair同一谓词全部已保留真实episode的并集及其1点边界保护区之外。其他真实episode绝不作为ranking负例。
- censored_positive_cells>0的整个pair跳过ranking，保留A/B原监督。训练缓存与原目标/谓词集合逐项检查；如果无法确认对应全集，停止排序训练准备。
- 每实例最多16正点×16负点，确定性均匀选择，三臂可比部分共用样本。没有可监督负点则不施加ranking。
- 报告eligible事件/全部事件、按时长和共同支持分层的数量及所有排除原因。无标注区域只作数据集监督条件，不宣称人工确认的视觉负例。
- 时长与连续共同支持从原训练标注及其SHA绑定映射获得，不用裁剪后mask长度冒充原始时长。如果长共存短关系训练eligible为零，停止本次排序小试并报告训练支持不足。

## 技术门槛和保存

原模型源码、配置、checkpoint、cache与评价来源SHA绑定；训练前原投影逐元素复现缓存mask；首个原生视频未训练投影/原候选区间复现；同query不同谓词共享时间区间。

训练前后保存同一train/dev固定输入的mask、分项损失及时间排序诊断，不能用不同batch的滑动均值代替固定对照。保存初始和第500步权重、schedule、配置及日志。

原生推理保持mask阈值0.5、首末正点区间和最少5帧；空或过短mask按统一规则回退原区间，所有臂一致并报告次数。预测封存后才读取验证GT做正式评价。

所有验证视频全体GT通过官方tube匹配，报告mAP、Top50/100/200、每GT新增/损失/净增、原C恢复，以及073目标、短/亚秒/持续组。mAP为视频AP均值，不能写成命中比例。所有百分数附分子分母；mAP不是计数比例。

## 继续与停止条件

主候选ordered须同时满足：

1. 在047和048两批，整体mAP及Top100命中均不低于baseline、original、balanced；
2. 合并073目标Top100命中严格超过三个对照，且每一批目标净变化均不为负；
3. 原073固定Top100网格可行、原阈值不可达的目标中至少1条实际正式命中新增加；以冻结身份清单评价，精确突破声称另核数值排序一致性；
4. 两批至少3秒组及有支持样本的长共存持续组Top100不低于baseline。

未通过不扩大本版本，不搜索新的loss权重、seed、阈值或checkpoint挽救；分别记录低于实用门槛、统计不确定和技术执行错误。balanced若胜于original可作为独立监督线索，但不能替ordered通过排序主门槛。

这些是小试筛选条件，不是显著性/泛化证明。若通过，下一步需冻结方法并补未参与选择的数据及近邻方法对照；若不通过，仅约束当前表示、监督、样本和500步预算，不推断所有短关系方向无效或底层特征完全没有信息。

本轮不自动进入区间选择器、记忆模块或全量训练。区间选择器只有原始分数改善且残留错误确实在读出阶段时才另行审查。


## Numerical compatibility revision before training, 2026-10-14
First launch stopped before training; preserve results_v1. Both saved probes check all111 cached batches. All88 training batches remain bit-exact. Only adapter_dev batch22 with B=1 differs (maximum absolute3.814697265625e-6, mean3.018026717427347e-8). Actual sigmoid>0.5 masks, decoded hulls and four-layer ordered Hungarian assignments are unchanged. The new wrapper permits only this exact dev batch and absolute error<=4e-6 (no relative tolerance); all training and native inference gates remain strict. Old cached masks retain fixed matching authority. No scientific loss, seed, schedule, evaluation or continuation criterion changes. Output is results_v2. Independent reviewer admission: PASS_WITH_CONCERNS.
