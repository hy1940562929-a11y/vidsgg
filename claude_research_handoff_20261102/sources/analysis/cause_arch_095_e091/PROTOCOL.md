# 095：E091-GEOMETRY-CONTRACT 执行冻结

唯一上位合同：091 `CAUSE_STATEMENT.md` §5、`REVIEW.md`、`ARCHITECTURE.md`几何投影P；093交付为阅读版。094材料门槛已通过。本文只固定实现细节，不改变三臂、成员或出口。

- A/B/S；seed90/91/92；每次500步，AdamW lr=3e-4、weight_decay=0、默认betas/eps、梯度L2裁剪1；最后checkpoint。
- 原共享三层Conv1d mask embedding全部参数可更新，但只接缓存decoder层3。层0–2不参与更新后的深监督；encoder/query/分类/轨迹/匹配不更新。不新建分类、端点、gate或融合头。
- train沿用88个原批次（350pair/563事件），每seed用NumPy default_rng(seed)逐epoch无放回shuffle，截前500个批次；三臂共享相同次序。开发23批（89pair/169事件）仅核查原生重建，不挑轮数。
- A直接调用已SHA核验原`masked_sigmoid_focal_loss`和`masked_dice_loss`：完整512长度focal、valid屏蔽、独立事件平均、权重2/5。
- 几何项：valid内`softmax(logsigmoid(z))`；GT独立事件mask除正点数；有效相邻CDF绝对差×4原帧／max(完整GT帧长,4)。padding不产生运输距离。
- S在valid正/负集合内分别固定置换，padding恒等。随机流来自`SHA256("E091/seed/video/pair/event_id")`前8字节，NumPy default_rng；原目标、匹配和原损失不变。每seed保存所有置换索引，训练不重抽，推理不用置换。
- 校准按原event_id字符串排序的前64个有效训练事件。每事件计算原项、B项、S项对有效logit的L2梯度范数；系数为`0.25 × median(norm_original) / median(norm_aux)`。B/S分别校准，之后冻结；中位数分母为0或非有限则停止。不是逐步梯度配平。
- 原400/163、2573原固定F、pair_idx及1101长组保持，旧H准入381/400。334残余名单来自封存089逐GT结果`T_exists && !T_C_hit`，仅用于预设NR评价。
- P按全部合法采样长度遍历，固定长度利用W1平移凸性找到最早最优起点，再跨长度比较；其结果须与朴素全枚举合成测试一致。不使用GT，不按候选长度除，不设置并列容差。C保持088 float32 sigmoid/float64均值与原并列规则。H准入仅从零更新分数冻结。
- 同槽预测复用，不因同pair有多条GT重复获得预测容量。P/C及原分数先封存SHA，再评价；零更新为Z(P)和C0(C)。
- 保持091全部P/E/C/长组/全体/null/NR统计与互斥出口。100次位置null沿088哈希，仅namespace改91，各库单独计数再平均。
- 首先运行合成验算与原生初始化重建（4e-6、离散输出一致）；不放宽门槛。训练前以无GT合成序列估算精确解码成本，4 CPU进程，预计超过1800秒即技术停止，不改变搜索集合。

所有数据和源码的SHA写入`config.json`及运行`execution_freeze.json`。只有本实验，不采用M091、不改RQ、不排队后续实验。任何阳性也仅支持合同限定的干预效应；NR不是原失败原因的可归因比例。
