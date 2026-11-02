# Query类别与时间mask对应：最近邻工作核查

日期2026-10-05。Direction对020协议、既有笔记和四项一级来源定向复核，Commander归档。状态NOVELTY_UNCERTAIN；无新实验事实、无方法采用决定。

|来源|已有机制|对本项目的约束|
|---|---|---|
|VrdONE，MM 2024|论文明确讨论关系分类与定位关联；同query embedding产生类别与mask；Hungarian匹配联合class CE、mask focal及dice|不能写首次关联分类与时间定位，或说VrdONE缺少联合监督。测试时分数是否选中对齐的响应仍需验证。|
|BIG，CVPR 2022|先判断全轨迹对关系，再做三元组条件的多实例时间定位；前景概率与centerness形成定位置信度并乘入关系分数|关系存在与具体发生时段的差异、谓词条件定位、定位置信度校正均已有直接先例。|
|SegPVSG / TFN，ICML 2026|多个区间预测、Gaussian mask引导注意力、区间条件关系分类及IoU分配多标签目标|用区间内特征辅助谓词判断已有VidSGG/PVSG先例；不能把mask池化或区间分类头本身作为首创。|
|TadTR，TIP 2022；邻近TAD|讨论类别置信度与定位质量不一致，用预测区间RoIAlign特征估计GT IoU、校正排名|分类高分不等于定位好、普通质量head已经有先例；其动作检测结果不等于解决VidOR短关系。|

## 一级来源

- VrdONE：[论文引言、3.4–3.5](https://arxiv.org/html/2408.09408v2)，[predictor.py第97行起](https://github.com/lucaspk512/vrdone/blob/main/models/predictor.py#L97)，[联合匹配第449行起](https://github.com/lucaspk512/vrdone/blob/main/models/maskvrd.py#L449)。
- BIG：[论文Fig.1、3.2](https://arxiv.org/pdf/2112.04222)，[定位评分](https://github.com/Dawn-LX/VidSGG-BIG/blob/main/models/grd_model_v5.py#L530)，[最终融合](https://github.com/Dawn-LX/VidSGG-BIG/blob/main/tools/eval_vidor.py#L237)。
- SegPVSG：[正式记录](https://proceedings.mlr.press/v306/li26bx.html)，[论文3.1.2–3.1.4](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26bx/li26bx.pdf)，[官方模型](https://github.com/ticatt/SegPVSG/blob/main/models/tfn/model.py)。
- TadTR：[论文III.A、Fig.4/6](https://arxiv.org/pdf/2106.10271)，[官方实现](https://github.com/xlliu7/TadTR/blob/master/models/tadtr.py)。

本次没有引用作者性能数值，也不把这些文献的结果当成本项目复现。未在这四项发现完全相同的020诊断，不构成不存在其他先例的证明。

## VrdONE原生行为与诊断读出的区别

每query每谓词输出一个跨时间固定标量，同时输出时间mask。固定标量不等于简单全段平均池化，不证明长关系或历史压制。

官方推理为每query保留TopK谓词，共享该query的mask，并根据主体、谓词、客体分数均值排序；参见[maskvrd.py第247行起](https://github.com/lucaspk512/vrdone/blob/main/models/maskvrd.py#L247)。**020的q_c=argmax_q P_q(p)是诊断single代表，不是官方只保留一个query的规则。** P×M跨query取max及其AUROC也是诊断构造，不是官方检测分数。

理想精度下，对固定query与正标量P_q(p)，AUROC(P_q(p)M_q,Y)=AUROC(M_q,Y)。类别分数通过跨query选择或聚合影响时间排序；它无法修复单query内部排序。实际float32并列/下溢仍须按冻结协议处理。

## 020可回答和不能回答的内容

可检验假设：已联合训练的九query输出中，有超出匹配搜索预算位置null的短事件对齐，而目标谓词诊断读出未充分利用它。

正D_bank只支持某query对齐标注时间，可能是其他谓词或通用变化。D_select与D_bank代数相关，不能当独立证据。D_bind的置换打破输出联合分布，不能单凭符号定位训练架构原因。通过仍需无GT方法的实际区间与正式检测验证；不通过则按冻结规则停止或记无定论。

“全局语义压制局部事件”目前UNVERIFIED。普通区间分类或质量评分的先例覆盖风险高；不据本笔记启动新架构。
