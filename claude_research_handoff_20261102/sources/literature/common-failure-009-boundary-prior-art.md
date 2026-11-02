# 边界修正候选：最接近先例

Direction Agent `/root/direction_rebinding` 于2026-09-28核查原始论文及官方实现，Commander保存。本节的论文数字均为作者报告，**不是本项目复现结果**。

独立起止残差、局部边界特征和已有候选时间细化均已有直接先例。`BOUNDARY-RESIDUAL-M1`可作为诊断改进基线，不能以这些操作本身声称首创。尚需验证是否有额外研究贡献。

| 原始来源 | 核实内容 | 对当前方法的限制 |
|---|---|---|
| [VrdONE，MM2024](https://arxiv.org/html/2408.09408v2)，Appendix C.1、D/Table10 | mask的0.5阈值及首末正点决定区间。将mask decoder替换为boundary regression后，作者报告VidOR mAP11.28/R10013.85，对比mask的11.86/14.21。作者还定性报告形态学后处理收益很小。 | 已考虑边界回归；该实验是decoder替换，不是冻结mask后的残差头，不能用其结果直接肯定或否定M1。 |
| [BIG / Classification-Then-Grounding，CVPR2022](https://arxiv.org/pdf/2112.04222)，§3.2、Fig.5、Table2、AppendixB | 以类别三元组为语言query，融合视频特征和实体共现时间先验，用左右边界回归做多实例grounding。作者报告BIG-C→BIG mAP8.29→8.54、R1009.65→10.04。 | VidSGG已存在关系条件定位左右边界；收益同时涉及额外特征、多实例输出及筛选，并非固定其他条件的纯边界效应。 |
| [TCANet，CVPR2021](https://arxiv.org/pdf/2103.13141)，§3.4、Fig.4、Eq10–16、Table11 | 起点、终点局部特征分别预测有符号offset，按proposal宽度缩放后修正已有边界，再与中心/长度回归结合。作者HACS平均mAP：局部首尾37.92、中心/长度38.22、联合38.71。 | 与“局部边界特征→独立残差→细化候选”直接重叠。换符号、小MLP或迁移到VrdONE本身不构成新机制。动作定位效果不等于已在VidOR解决。 |
| [CBR，BMVC2017](https://www.bmva-archive.org.uk/bmvc/2017/papers/paper052/paper052.pdf)，§3.2–3.5、Eq4、Table1 | 拼接片段内外上下文，预测独立起止offset并级联细化；检测阶段有类别条件offset。作者THUMOS14单轮Two-stream mAP@0.5：不回归22.3、frame-level25.3、unit-level27.7。 | 首尾移动、残差细化、类别条件和时间粒度控制已有先例。该系统也预测分类／置信度，不能把其数字解释为固定排序的纯定位提升。 |

## 官方推理路径核查

[VrdONE官方MaskVRD.forward_test](https://github.com/lucaspk512/vrdone/blob/main/models/maskvrd.py)的最终输出路径：sigmoid mask >0.5，首末位置映射原帧，最短长度筛选，裁剪实体轨迹，按类别分数组合排序。已核查路径没有额外的可学习首尾残差头，也没有每query预测阈值。这不是“内部不学习定位”或“所有后续工作都无此机制”的证据。

此次对VidVRD/VidSGG boundary/proposal refinement与adaptive threshold的定向检索，尚未核实直接对应“根据VrdONE query预测mask二值化阈值”的原始方法。**未找到不等于不存在。** BIG是query条件边界回归，不能混称query条件阈值。条件阈值仍应作为必要简单对照。

## 后续需要证明的内容

已有R4正负例只支持部分既有候选的时间范围可改善。若获得训练数据，应冻结轨迹、类别、分数及候选身份，比较原0.5、固定0.75、train拟合全局阈值、query条件阈值、普通边界残差，以及移除query／局部特征的同规模头。测量带符号及绝对首尾误差、连续tIoU、轨迹vIoU和官方新增／丢失，单列候选过滤与排名作用。

如果简单条件阈值已达到相同效果、移除所称关键输入不降低效果、收益只来自过滤／排序，或只发生在少数接近匹配门槛的样例，则对应机制主张未被支持。若只证明已有残差回归在VrdONE上可用，应报告为技术适配；CVPR级新贡献仍需额外证据。
