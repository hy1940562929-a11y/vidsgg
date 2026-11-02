# 083：亚秒失败的时间表达可行性分解

状态：协议冻结，尚未生成本轮分解。使用已经导入的015原始输出诊断与077逐GT时间读出，不做新训练、推理或模型修改。083是080/081发现后的探索，不当成独立验证。

## 问题与固定人群

081在所选实体段覆盖GT的亚秒组187条中，发现176条在全部同identity查询库中仍无tIoU≥0.5的默认区间。本轮核查：网格本身无法表达、最短5帧拒绝、固定阈值不合适，还是现有mask分数排序和首末hull解码族都无法给出目标区间。

主统计全部GT27的2,573条；分别报告<1、1–3、≥3秒，以及所选实体段覆盖GT的人群。同样列selected命中/oracle-only/neither；187及176是已知结果后选择的诊断焦点，不能作为独立泛化估计。阈值0.5主，0.3/0.7敏感性，不择优。

## 来源一致性与范围

读取015 reproduced_evaluation/per_gt.csv的同SHA本地副本analysis/temporal_support_015/deployment/reference_per_gt.csv和diagnosis_v1/per_gt.csv（从已导入的diagnostic_evidence_v1.tar.gz读取member，避免iCloud占位读取超时），077 boundary逐GT，以及080元数据。必须核验GT集合、身份/谓词/半开时间/FPS；015 default_hull_tiou_min5 必须逐条等于077 oracle_tiou_min5。记录文件SHA；若差异存在，不混合诊断。

015的oracle均在同GT主体/客体身份的所有连续输入段和所有query中搜索；不限定077分类选中的实体段，也不要求正确谓词。它是已有时间形状的诊断上界，不是实际可恢复的完整关系。网格oracle允许理想选择首末网格位置；阈值oracle只改变现有query分数阈值、保留首末hull读出。两者不得混为同一种能力。

## 互斥分解（依次归类）

1. default_hit：默认0.5阈值+最短5帧的任意query已能达到该tIoU。
2. grid_unfiltered_infeasible：即便理想网格首末位置、无最短5帧限制也不能达到。
3. grid_min5_infeasible：无长度限制的网格可行，最短5帧限制后不可行。
4. default_min5_rejected：网格min5可行，但默认阈值的匹配hull被最短5帧拒绝；放宽此过滤即可在上界内找到形状。
5. sigmoid_threshold_reachable：默认失败，但float32 sigmoid的全部可达阈值中有匹配hull。
6. raw_logit_only_reachable：sigmoid阈值不行，原始logit阈值可行（核查饱和/tie造成的数值差异）。
7. grid_feasible_threshold_family_infeasible：网格可行，但全部query的raw-logit可达hull均不能达到。

各类互斥且和为原人群N；第4类单独说明可与阈值可达性重叠，本表按以上顺序归类。保存独立可行标记以便复算其它划分。若存在理论单调性不符（默认超过网格、sigmoid超过raw等），整批失败并报告，不强行归类。

## 能回答与不能回答

若网格足够但现有阈值族不够，可以排除“当前网格根本无法容纳匹配区间”对这些网格可行样例的解释，并量化“只改现有阈值”在本读出族内的上限；不能因此说更密的真实采样一定没有帮助；只对本群体及已存输出成立。不能证明特征缺少视觉信息、训练先验、上下文干扰或哪个层产生了错误。不得把oracle救回数量写成新模型指标；不会改080/081的H1/H2/H5机制判决。

不新设效应大小或显著性门槛；这是已存输出的确定性互斥分解。产生逐GT、分组CSV、来源清单和简短报告，接受独立复核。所有比例显式分子/分母。
