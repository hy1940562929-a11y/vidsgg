# Week1 短关系诊断080

状态：**LOCAL_DIAG_COMPLETE_FULLVAL_TEMPORAL_PENDING**。真实重算已有结果，无新训练、无模型推理。不能描述为完成了完整val原生时间诊断。

先读 [一页判决](DECISION_ONEPAGE.md)。主要产物：

| 文件 | 范围与用途 |
|---|---|
| results_v1/diag_summary.csv | 全val切片N与显式NA时间字段；GT27完整时间指标；不得混合两种来源 |
| results_v1/official_detection_summary.csv | 全val原Top100正式检测命中，VrdONE/BIG分别保存 |
| results_v1/oracle_tiou_summary.csv | 最好query的时间上界，不是部署读出或新增正确预测 |
| results_v1/per_gt_slices.csv | 30,142条唯一GT、连续共存、长关系伴随标注及四种切片定义 |
| results_v1/per_gt_boundary_joined.csv | 2,573条已保存时间预测的端点重算与分组 |
| results_v1/per_gt_canonical.jsonl | 聚合使用的展开行；同GT按不同定义和子群出现，不能直接跨定义相加 |
| controls_v1/controlled_summary.csv | 42种冻结比较×2种视频bootstrap；84条完整结果 |
| controls_v1/*.json.gz | 每种比较的逐GT共同支持、逐层分子分母及完整判决 |
| readout_audit_v1/readout_audit.csv | 重复事件复用读出、预测不重叠、重叠/不重叠stretch拆分 |
| POSTHOC_VALIDITY_AUDIT.md | 事后测量有效性检查：天花板不等于机制证伪，不撤销负结果 |
| ../../reviews/week1_diag_080_review.md | 独立复核，由Commander保存reviewer回传 |

## 范围与来源

全val为30,142条GT、832个有GT视频；原评估视频总数835，3个无GT视频不产生逐GT行。一个没有预测键的视频`1027_4077228320`的10条GT仍在分母。原Top100 A/B/C/D=2487/1809/25373/473。

时间端点来自已曝光的27视频/2,573条正确GT实体输入诊断。它按同一实体身份、同一目标谓词，在所有连续输入段中选分类分数最高query，并对同身份谓词多个标注事件复用该query。GT时间不参与主query选择，但这是有GT实体和谓词条件的诊断，不是原生Top100部署性能。所选输入段可能不覆盖目标GT；即使覆盖，输出也可能处于另一个时段。

主时间命中758/2573、oracle1731/2573已独立复算。只对已经完整导入的时间记录计算指标，不从正式命中行推造未命中行的时间预测。BIG只有命中缓存，不能报BIG时间表。

原始文件和SHA在results_v1/manifest.json。full835导入manifest明确仅带回命中、候选metadata和provenance，完整预测轨迹仍在服务器。分视频来源为`/root/autodl-tmp/VrdONE-data/commander_common_failure_009/full835/predictions/{video_name}.baseline.json.gz`；其历史基线复核835/835通过、最大分数差0。当前没有关系时间盲的proposal→GT身份映射，不能把proposal sid/oid直接当GT tid。

## 冻结口径

短<3秒；长关系≥3秒；长共存为GT连续共同支持≥10秒；S-S共存<10秒。沿用真实fps。共存缺失/非正/短于目标时长记UNKNOWN。另存L-S，使四格主表外的剩余GT可核对。

四个并列定义：

1. literal：同video、同有序实体对另有≥3秒GT关系，可同谓词、可异时段。
2. strict：另一关系≥3秒，谓词不同，且时间正重叠。
3. strict-half：strict且一条伴随关系覆盖目标至少一半。
4. strict-ge10：strict但伴随关系≥10秒。

solo仅表示未标注符合定义的伴随关系，不证明视觉语义中不存在。共存只表示GT连续共同支持，不证明持续清晰可见。

N包含无输出GT；miss(t)=无有效预测或tIoU<t，t=0.3/0.5/0.7。保留原min5帧输出规则，半开区间，未改变mask阈值。stretch≥2、时长比只在有效预测N_pred上定义，另报stretch/N；平均tIoU用全N，无输出记0。没有导入时间来源时记NA，不记0。

全valall/F与GT27all/F/输入段覆盖/覆盖且同身份谓词唯一标注/亚秒组分列。F始终是**VrdONE原生输入可行性**；在BIG的F子集比较也不意味着BIG输入正确。

## 控制统计

H1控制谓词+共存档；H2控制谓词+0.25秒时长档，并附两侧均无竞争者；H5控制谓词+时长档+共存档，并附视频GT负载。只保留双侧共同支持层，以两侧合计GT量加权。分子可为小数，代表标准化权重，不是恢复样本数。

整视频有放回bootstrap为主，2,000次、seed80；重复视频不去重。077的正Exp(1)视频乘子法另列敏感性。每侧至少20视频、共同支持至少60GT、有效重采样至少95%、支持保留p05至少80%才应用预设3pp观测预言判决。机制没有干预均未决。不能从84行中挑最有利bootstrap、子群或阈值当主结论。

## 复现

在项目根目录运行；输出目录必须不存在。原生全val文件不在本机，重跑仍会如实保留相应NA。

```bash
/Users/xk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s analysis/week1_short_relation_diag_080 -p 'test_*.py'
python3 analysis/week1_short_relation_diag_080/run.py --out /tmp/cvpr080_recomputed
python3 analysis/week1_short_relation_diag_080/readout_audit.py --input /tmp/cvpr080_recomputed/per_gt_boundary_joined.csv --out /tmp/cvpr080_readout
```

控制统计需要现有NumPy，使用本机已捆绑运行时，不安装依赖：

```bash
/Users/xk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 analysis/week1_short_relation_diag_080/run_controls.py --inputs /tmp/cvpr080_recomputed --out /tmp/cvpr080_controls
```

39项合成测试在捆绑运行时通过。results_v1精确运行源保存在source_snapshot_v1；随后run.py仅修补缺失共存值的join健壮性（现有数据无缺失，不改变结果）。独立reviewer逐条重算全部N、时间端点、汇总及关键bootstrap，结果一致。可用`python3 analysis/week1_short_relation_diag_080/make_report.py`从已保存结果重建一页判决。

所有结果限于已有验证数据的诊断，不是独立泛化评估。正式机制结论、性能改进及新方法采用均未发生。
