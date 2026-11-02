# 总账补充：019

原总账及015–018补充保留不变。019已实际完成、导入并通过独立审查。

|实验|变化|数据/对照|结果|正确新增/损失/C恢复|
|---|---|---|---|---|
|019固定输出时间对齐|只改变保存输出的诊断读出与位置对照；模型、输入、采样不变|27个已暴露GT输入验证视频、2573 GT；3181单位中1354可评；single/set、通用mask、平移/反转/中心函数|set AUROC .697246，平移均值 .488981；短组set .590419、通用均值 .593353|不适用：没有新检测预测，不填0|

证据类型：Diagnosis及Correlational Observation。支持冻结GT输入下的标注时间关联；不支持短关系因果、上下文压制或检测性能改进。原016负结果和018有限新覆盖仍有效。

详细字段见`ledger_addendum_019.json`；结果说明见`analysis/temporal_alignment_019_results/REPORT.md`；独立审查见`reviews/TEMPORAL-ALIGNMENT-019-RESULT-REVIEW.md`；Evidence ID为`EV-VIDOR-TEMPORAL-ALIGNMENT-019`。

020仅预注册和实现中，尚无实验结果，不计入已完成总账。
