# 086 证据 → 判决

2026-10-20。仅综合既有实验；086交集为CPU统计重组，不是新的模型实验。所有百分比附分子分母；标准化差值的非整数分子不是恢复样本数。

## 来源与结论边界

| 证据 | 核验路径（相对仓库根目录） | 核心数字与性质 | 可以支持／不能支持 |
|---|---|---|---|
| 080 H1 | `analysis/week1_short_relation_diag_080/controls_v1/controlled_summary.csv`；`DECISION_ONEPAGE.md`；`reviews/week1_diag_080_review.md` | GT27控制后短关系漏检差+25.04 pp（521.5672/2083×100）；Correlation | 时长关联；不是已证实训练时长先验 |
| 080 H2 | 同上，`H2/VrdONE_official_miss/all/h2_no_comp_control=True/load_control=False/whole_video` | 长共存差+8.349 pp（452.6867/5422×100），CI5.56–11.71 pp；Correlation | 无其它符合条件长关系标注时也更难；不证明共存特征淹没 |
| 080 H5 | 同上，`H5/VrdONE_official_miss/all/load_control=True/whole_video` | −0.547 pp（−9.2362/1689×100），CI−2.02～1.39 pp；Correlation | 未支持正向额外漏检；不普遍证伪语义竞争 |
| 081 H5 | `analysis/week1_continuous_readout_081/results_v1/summary.csv`；`REPORT.md`；`reviews/week1_continuous_readout_081_review.md` | comp 12.463411/328−solo 5.986721/328=+0.019746 tIoU；solo10视频；Correlation | 连续指标方向也未支持压制；支持不足，不是保护效应 |
| 081 分解 | `analysis/week1_continuous_readout_081/results_v1/per_gt_partition.csv`、`decomposition.csv` | short固定127/966、oracle-only346/966、neither493/966；oracle-only中282/346所选段不覆盖；Diagnosis | 不同失败阶段；不能称改变选段已救回282条 |
| 082 全val | `analysis/week1_native_readout_082/imported_fullval_20261020/{diag_summary.csv,readout_audit.csv,REPORT.md}`；`reviews/week1_native_readout_082_actual_review.md` | solo未命中261/265，comp2750/2790；有预测12235，其中复用7829；stretch3783中2181错时段；Observation | 区分错事件和延长；不是完整模型因果失败分解 |
| 083 | `analysis/week1_query_feasibility_083/results_v1/{per_gt.csv,summary.csv,REPORT.md}`；`run.py`；`reviews/week1_query_feasibility_083_review.md` | 覆盖GT亚秒187：默认11、网格不行1、阈值可达25、hull族不可达150；Diagnosis | 现存分数+hull族限制；不能说编码信息消失；187不等于长共存组 |
| 084 | `analysis/week1_component_readout_084/results_v1/{per_gt.csv,summary.csv,REPORT.md}`；`run.py`；`reviews/week1_component_readout_084_review.md` | 后验150：默认拆分0/150，全阈值分量121/150；全2573固定拆分新增109损失19；Diagnosis | 区间族不同；不是可部署恢复，获胜query可不同 |
| 015D既有容量反证 | `analysis/temporal_support_015/recomputed_null_v1.json`、`review_null_result_v1.json`；以及084报告与review | same-query正确谓词Top6原位置2437/2573，随机五种子2513、2517、2517、2524、2513/2573；本轮已读原汇总与审查回执，不重新跑null | 保留随机容量负结果，不能将它直接移植到150或400子群 |
| 079A | `analysis/short_duration_mechanisms_079/results_20261020/REPORT.md`；`reviews/short_duration_mechanisms_079_actual_review.md`；原始`handoff/incoming/VIDOR-MECHANISMS-079/remote_20261020/A_sampling_architecture/` | dense−插值短961新增2损失2；half短966新增37损失25，全体2573新增53损失126；Intervention | 当前采样、削弱粗尺度配方无推进证据；不是长语义删除 |
| 079B | 同上报告与`results_v3_BC/B_layer_readout/status.json` | 最后dev batch feature差3.8743e-5、mask差0.0028超过冻结4e-6；训练未开始；Technical failure | 细尺度vs融合后可读性尚无结果；不算科学负结果 |
| 079C | 同上报告与`results_v3_BC/C_objective/{summary.json,per_gt_final.csv}` | dev短24/44→23/44，全体121/169→118/169；Intervention | 否定本配方准入，不否定全部训练原因 |
| 085 | `analysis/long_short_context_085/{REPORT.md,counts.json}`；`reviews/long_short_context_085_actual_review.md`；原始`handoff/incoming/VIDOR-LONGSHORT-CONTEXT-085/remote_20261020/infer_v3/` | 7事件3视频；L−N+0.003202，CI跨零；命中0/7→0/7；长mask仅2/7下降；Intervention | 有上下文敏感性，无可靠语义操纵及恢复；机制未决 |
| 086 交集 | 本目录`existing_evidence_intersections.csv`、`intersection_members.json`、`intersection_manifest.json`、`consolidate.py` | covers & coexist≥10 & relation<3：400/23视频，selected10、hull47、全部阈值hull121、分量61、全部阈值分量355；Derived diagnosis | 直接落到锁定目标群体；仍是事后分组与oracle，不是受控新实验 |

## 086交集闭合

400条：默认hull47＋阈值增量74＋网格可行但hull族不可达279＝400。
279条中：全部阈值分量可达234＋仍不可达45＝279。
固定读出10与任意query默认hull47的差37，只是时间形状余量，不是37条正式预测恢复。

原长共存短组466条中，400条所选段覆盖GT，其余66条不是本主定位的支持群体。400不是完整val的30,142，也不是所有短关系966；子群之间不得换分母。

复现命令：`python3 analysis/mechanism_verdict_086/consolidate.py`。仅标准库，来源CSV不变，不调用模型。

## 必须保留的口径冲突

1. **旧080“全val时间表未完成”已由082补齐**；保留旧文档时间戳，不继续作为当前缺项。082的固定读出仍不是官方一对一R@100。
2. **comp定义不同**：080字面另一长标注2973，严格异谓词重叠2790；085另要求覆盖目标至少一半。三者不能混用分母。
3. **1731与1717不同**：前者为hull时间区间覆盖，历史1717为另一二值mask覆盖口径，不是前后性能变化。
4. **187与163不同**：前者覆盖GT亚秒全群体，后者额外要求长共存；150与143也分别是两个群体的hull族失败数。
5. **selected、any-query、阈值/分量oracle、正式检测不同**；oracle跨同身份连续段、不要求谓词正确，禁止写成已恢复关系。
6. **已观察验证集不是独立泛化测试**。GT27非随机且已曝光；不能将086事后交集或后续固定小探针称为独立benchmark。
7. **085主效应视频等权正、事件等权负**，两者都保留；3视频bootstrap不能支撑总体机制。
8. **083“hull族无区间”不等于084“所有读出都无区间”**；进一步更换区间族改变可达性与搜索容量，不自动改变语义可读性。

## 086独立复核

Reviewer对083/084交集另行复算，确认400/163及gain/loss；强调覆盖标记与oracle搜索范围不同。判决接受失败环节定位，不接受“已找到单一架构/训练原因”。最终草稿复核回执另存本目录。
