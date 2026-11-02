# 024总账增补

证据：EV-VIDOR-DECODER-LOSS-024。27视频、2573GT、十条件、25730行事后描述，全部来自已导入023。新GPU前向0，新训练0；无新检测性能试验。完整各臂/分组得失保留在 `ledger_addendum_024.json`，不重复扩大样本量。

F新增36中26为相同几何、分数和raw class/mask的预测进入Top100；损失97中15仍有尾部兼容候选，82无saved200兼容候选。原query有33/97 mask变空、88/97 hull改变、91/97类别分数下降，重叠统计不可相加。

Gout14目标新增均有原A同query保存候选但vIoU不足，GT内mask/class不变后兼容。其63全体损失中，仅2条与目标新增共享query，均来自同视频、同谓词不同片段；不能将这两例当主要机制。

主诊断及hybrid均独立审查PASS。原023科学结论INCONCLUSIVE不变，无训练准入。报告 `../decoder_loss_024/REPORT.md`。保留所有负结果及文献近邻风险，下一次GPU前先在本地完成明确问题与冻结批次。
