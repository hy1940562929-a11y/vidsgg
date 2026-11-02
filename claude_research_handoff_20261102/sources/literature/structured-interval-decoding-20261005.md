# 单连续区间解码：已知数学对照，非新方法

2026-10-05，Direction 只读文献与数学核验；无实验、无方法采用。016 的 H0/C05/UOIC 三臂保持不变。

若固定query的各时刻标签在给定输入后独立服从 Bernoulli(p_t)，合法序列为空或单个连续区间，且没有额外非均匀区间先验，则对区间 I：

`log P(Y=1_I) = sum_t log(1-p_t) + sum_{t in I} logit(p_t)`。

因此受约束 MAP 是最大合法连续 logit 和，空区间得分为0。min5原始帧长度必须进入搜索约束；不能先选无限制最大和后再丢弃短结果。零分并列规则必须事先固定。加入时长/边界先验会改变目标。

该规则与hull不同：它考虑内部负证据的总量；与CC不同：它可以跨越较浅低谷；与OIC不同：它不比较局部内外均值。但中间较深低谷仍可使它截断真实长关系，也可能跨越较弱负分间隔合并不同事件。

这些Bernoulli独立性和概率校准条件对本项目为UNVERIFIED。原训练使用sigmoid及focal/Dice，不自动保证这些条件；MAP目标也不直接等于tIoU、vIoU或Top100效用。015D已否定以高oracle覆盖证明有效局部对齐的解释，数学推导同样不构成效果证据。

已核验直接先例：

- [Temporal Action Localization by Structured Maximal Sums，CVPR 2017](https://arxiv.org/pdf/1704.04671)：对区间累计逐帧有符号分数，用最大子段和及结构化动态规划定位，并引入起点、中间、终点分数。它不是本文所述Bernoulli-logit推导，已足以说明最大连续和定位属于既有范式。
- [Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs，arXiv v3](https://arxiv.org/html/2510.00705v3)：UG-Ground使用yes/no概率差及Kadane算法定位。概率差不等于log-odds，不能声称两者输出等价。

未找到VidVRD/VidSGG直接使用上述特定logit推导的一级来源，只是本轮检索边界，不是新颖性证明。若将来使用，最多先作为“水平集解码与累计证据解码”的已知机制对照；当前未列入执行实验。
