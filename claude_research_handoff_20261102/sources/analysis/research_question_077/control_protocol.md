# 077：关系时长与实体共同支持时长的预设描述对照

本协议在运行真实比较前固定。只读取既有预测/标注，不运行模型、不改变预测、不以 GT 生成新的预测。结果属于 **Correlation**，不是训练收益、因果效应或独立泛化结果。

## 数据与字段

- 073 归档的 `per_gt_durations.csv`：30,142 条 VidOR validation GT，`relation_sec`、`joint_local_continuous_sec`、双模型 Top100 命中、C、F。
- 073 归档的 `per_gt_features.csv`：相同 GT。`size` 的原始分箱 `<1% / 1–5% / >=5%`；`motion` 的原始分箱 `<.02 / .02–.1 / >=.1`。这些导出没有连续面积/速度数值，不能把分箱中点冒充测量值。大小缺失 10 条、速度缺失 411 条，涉及相应控制的比较完整案例删除，并报告删除数。
- 共存仅使用包含当前关系区间的 **GT 连续共同支持** `joint_local_continuous_sec`，包括标注生成框，不能解释成已人工确认持续可见；不用实体各自总寿命，也不用关系标注跨度 `pair_span_frames`。
- `per_gt_cascade.csv` 只有 4,739 条，是局部阶段诊断，不能补成全量 GT 的输入状态。全量 F 使用原归档 F（13,294 条），它仅代表 VrdONE 已有候选轨迹几何可行，不是 GT 输入，也不代表 BIG 输入可行。
- 按 `video_name + gt_index` 一对一连接，核对主客体 ID/类别、谓词、区间、组别、C/F、时长，任何冲突报错。双模型命中必须与 A/B/C/D 一致。记录源文件及脚本 SHA256。

## 两个主问题

1. **同一种关系，短时是否仍更易漏检？** 暴露：关系 `<3s` 对 `>=3s`。主设计精确匹配类别三元组，并控制大小档、速度档、GT 共存档 `<3 / 3–10 / 10–20 / >=20s`。
2. **关系时长接近时，实体长时间共存是否有额外差距？** 只取 `<3s` 关系；暴露：GT 连续共同支持 `>=10s` 对 `<10s`。主设计精确谓词，加关系时长 **0.25s** 细档、大小档、速度档。

两者预设主群体均为 F；全量 GT 为必报并列分析，不能在 F 未支持时改取全量作为主结论。短关系主比较和共存主比较回答不同问题，不在二者之间择优隐去负结果。

支持性设计：精确谓词；精确类别三元组；谓词+大小+运动；同视频+同 ordered pair+同谓词。共存严格比较额外含关系 0.25s 档；同一实体对的共同支持若不变化，该层自然不可比较，不人为制造对照。另报长期共存内的长短关系对照，以及更细类别三元组共存对照，标为探索性。

**没有像素遮挡/关系清晰度标签，不宣称控制了遮挡、视觉证据充分度或标注歧义。** 同谓词控制亦不能排除同谓词内部的视觉难度差异。

## 指标和估计量

- 主终点：VrdONE Top100 **漏检率差**，暴露组减对照组。
- 必报并列终点：VrdONE 命中、BIG 命中、C 共同漏检，逐项整数分子/分母。
- 每设计报告原总体原始率、完整协变量原始率、相同共同支持群体原始率、标准化率与差值，以及两组/视频/层的支持数量和覆盖。前后差来自不同人群时不能解释成“消除的原因占比”。
- 共同支持：每个层两组各至少 1 条；不是每层精度保证。层权重 `w_h=(n_exposed,h+n_control,h)/N_supported`，两组使用相同权重。输出分层表和逐 GT 所属层，能够重算。
- 标准化分子是加权的分数计数，**不是实际恢复实例数**；不把它写成新模型收益。
- 固定随机种子 20261015；默认 2,000 次正权视频 cluster weighted bootstrap。每次为每个视频独立生成 Exp(1) 正权重，同视频全部 GT 共用该权重；在固定共同支持层上重算层内率及 pooled 权重。全部原共同支持层、双侧样本均保留，不因稀疏层被抽空而更换比较人群。报告支持保留质量（应恒为1）。这是视频权重 bootstrap，不是整数视频有放回重抽；置信区间仍是条件描述性不确定性，不能保证每个稀疏层都有充分精度。

## 判定规则（冻结，不据结果改阈值）

- 描述关联支持：主终点点差 `>=3 pp`，且 95% 视频 bootstrap 区间下界 `>0`。
- 上界 `<3 pp`：该共同支持人群不支持预设至少 3 pp 效应；不能推断所有条件下绝对无差异。
- 区间跨零、上界仍 `>=3 pp`：INCONCLUSIVE（精度不足），不能把 p>0.05 当否定。
- 任一比较侧独立视频 `<20`、或共同支持 GT 总数 `<60`：INCONCLUSIVE_SUPPORT_OR_BOOTSTRAP_STABILITY。每侧至少20视频的门槛已由 Commander 在初次读取真实效果前冻结于 PROTOCOL.md；冻结 controls.py 最初误只检查视频并集，现由独立 verify_controls.py 依据 membership 应用每侧门槛，保留原判断及更正判断，不改原始 summary。代码另检查正权 bootstrap 支持保留质量，理论应恒为1；异常保留质量不能作为有效区间使用。
- 报告所有预设设计；区间是边际区间，不是多重比较调整后的确认性检验。上述支持仍是关联，无法证明模型历史/上下文压制机制。

## 运行

```bash
python controls.py --self-test --out /tmp/rq077-unused
python controls.py \
  --root /root/rebinding_project/VrdONE \
  --features /root/autodl-tmp/VrdONE-data/commander_event_support_073/code_v1/inputs/per_gt_features.csv \
  --durations /root/autodl-tmp/VrdONE-data/commander_event_support_073/code_v1/inputs/per_gt_durations.csv \
  --out /root/autodl-tmp/VrdONE-data/research_question_077/controls_v1
```

输出：`summary.json`、`comparisons.csv`、`strata.csv`、`per_gt_controls.csv`、`per_gt_support_membership.csv`、`predicate_duration_counts.csv`、`output_manifest.json`。拒绝覆盖已有输出目录。只依赖 Python 标准库和 NumPy；不用 GPU。
