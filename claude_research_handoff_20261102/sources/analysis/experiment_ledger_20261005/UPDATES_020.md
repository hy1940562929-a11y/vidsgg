# 总账补充：020

020已实际完成、导入并独立审查PASS；科学判断为INCONCLUSIVE。此页更新019补充写入时“020尚未运行”的状态，保留之前各实验的历史结果。

27个GT输入探索视频、2573事件、3181单位；短组505/638有效。仅读取冻结输出，30.42秒CPU，无训练或新检测预测。

|主短组统计|视频宏均值|视频bootstrap区间|
|---|---:|---|
|query库对齐D_bank|+0.072618|[+0.045891,+0.095518]|
|校正后的诊断选择差D_select|+0.000636|[−0.055092,+0.059609]|
|校正后的类别对应差D_bind|+0.041238|[−0.003516,+0.085356]|

高GT辅助oracle .891973不能视为检测能力：同搜索预算平移oracle也达到.819355。主gate未通过，同时也不满足预先的否定/等效条件。因此不启动选择器训练，不改阈值或换有利子组。

新增正确、损失正确、C恢复：不适用，不填0。证据ID `EV-VIDOR-QUERY-BINDING-020`；全部字段见`ledger_addendum_020.json`，报告`analysis/query_binding_020_results/REPORT.md`，审查`reviews/QUERY-BINDING-020-RESULT-REVIEW.md`。

下一项稀疏hull诊断仅有草案`methods/temporal_sparse_support_021_draft.md`，未冻结、未运行，不能计入实验结果。
