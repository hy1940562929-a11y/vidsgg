# 实验总账增补025

实验：VIDOR-NATIVE-ERROR-025。目的：在原生输入下分离输入不可能、实际输出网格不可能、原query时间跨度缺失、谓词Top6及候选排序/分配，确认短共同漏检是否存在足够规模的时间输出缺口。

性质：Diagnosis + descriptive observation，复用009原输出；不是模型干预、机制因果证据或训练。模型/权重/输入/官方顺序/匹配不变。

数据：额外冻结64视频/1985GT，原C1660、短C794、输入可行短C287；已暴露验证集，排除含旧outcome-derived案例，不是独立测试集。

全体：原命中298，输入不可行1159，网格不可行29，原hull无匹配266，Top6缺失35，正确query未命中198。原基线mAP0.112319986752，R100=298/1985。

主结果：短C的对应分桶507/28/186/20/53。186/287为网格可行但原hull缺失，占输入可行短C的64.8084%（186/287×100）；29视频、18谓词。2000视频bootstrap区间[0.501578,0.750031]。PASS_SCALE_ONLY，独立结果审查PASS_WITH_CONCERNS。新增、损失、C恢复不适用，因为没有改模型。

形态：186/186个GT辅助所选见证至少一侧过伸，16/186相对最优网格区间同时欠覆盖；单视频占60/186，watch+in_front_of占110/186。不能由此推断上下文/记忆/读出原因或声称恢复186条。

资源：远端CPU导出57.407秒，本地CPU诊断44.797秒，GPU新前向0；516文件与12800候选绑定已校验。已通知可关闭AutoDL，真实开关机/收费状态未知。

证据：EV-VIDOR-NATIVE-ERROR-025；详细路径 `analysis/native_error_025/REPORT.md`、`results_v1/summary.json`、`per_gt_diagnosis.csv`、`hull_gap_witnesses.json.gz`、`reviews/NATIVE-ERROR-025-RESULT-REVIEW.md`；执行代码在 `frozen_diagnosis_v1/`。

下一步仅准备有明确对照的最小机制实验；无新GPU任务或训练排队，不改变023/024及历史负结果，不正式选择主方向。
