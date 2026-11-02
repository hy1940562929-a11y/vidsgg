# 082：原生预测导出的本地核验与汇总

**当前仅代码准备完成；没有082真实导入结果，也没有创建Evidence。** 服务器CPU exporter由`handoff/outgoing/VIDOR-WEEK1-NATIVE-EXPORT-082/`提供。本目录不连接服务器、不运行模型或训练、不反序列化模型/特征pickle。

## 收到完整导出之后

将导出目录整体复制到本地，再运行（把`/absolute/path/to/export`换成真实目录）：

```bash
cd /Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/week1_native_readout_082
python3 import_native.py \
  --export-dir /absolute/path/to/export \
  --out /Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/week1_native_readout_082/imported_fullval_v1
```

默认读取080的`results_v1/per_gt_slices.csv`和本地082交接包；可通过`--metadata`、`--package-root`显式定位同一冻结版本。输出目录必须不存在。核验失败不会生成诊断表。

只核验明确完成的局部技术烟测时可以加`--allow-partial-smoke`。它仍拒绝RUNNING、FAILED或未完成输出；导入清单为`PARTIAL_DIAGNOSTIC_ONLY`、`complete=false`，不能称为全量结果。正式运行不要加该选项。

## 核验内容

1. 完整导出状态、835视频清单、30,142唯一GT、832个含GT视频；每Top50/100/200恰有全部GT一次。视频无预测或无GT不从视频清单删除。
2. 所有导出文件SHA和大小、固定schema、exporter代码版本、历史源SHA、metadata和inventory回执；GT身份、类别、谓词、时间、FPS、F和旧命中标记逐字段对齐。
3. `actual_pairs.csv`中的实际有序pair、角色eligible并集、`proposal_binding.csv`的绑定阈值和类别。不会用两个独立bound实体的笛卡尔积代替实际pair。
4. `saved_candidates.csv`的全部原Top200候选（含未绑定项）、原rank和分数次序、pair索引和类别；独立重建每K内按身份/谓词选择的首个候选。选择不读取GT关系时间。
5. 预测半开区间、tIoU、时长比、时长秒数、完全不重叠标记全部重算；无预测时tIoU为0，但区间、时长比、disjoint和来源索引必须为NA。
6. rank≤K，Top50已有选择时Top100/200不能改变，后续新增必须符合原rank；逐预测复用次数独立重算。
7. 视频回执、源SHA回执和exporter总体分子分母/时间指标与本地结果核对。

轻量导出没有原始全部框，因此本地不能第二次从像素框重算实体映射IoU；该环节由冻结exporter在服务器上对完整轨迹计算，本地核验其分数、阈值、身份、实际pair与历史来源。不能把这个几何映射叫作正确GT实体输入。

## 输出含义

- `diag_summary.csv`：四种切片定义×Top50/100/200×all、mapped_actual_pair、native_F。含四个主切片、L-S、UNKNOWN及ALL。零样本群体保留N=0、比率NA。
- `readout_audit.csv`：无预测、实际pair映射覆盖、完全错时段、错时段伸长/重叠伸长、同一预测被多个GT复用。所有比例带分子分母。
- `binding_status_audit.csv`：proposal层面各种映射状态及分母。
- `official_historical_top100.csv`：只按新切片重新分组原VrdONE/BIG Top100命中标记，明确不是新评估的R@100。
- `manifest.json`：输入/代码SHA、完整或局部范围、核验状态、输出SHA。

这里的时间读出允许一条预测服务于同身份同谓词的多个GT事件，因此**tIoU命中率不是官方一对一R@K**；mapped_actual_pair仅表示可由实际输入pair构成几何映射，仍包含轨迹和时间读出问题。不得把错时段大时长比直接解释为边界延长。

## 合成测试

```bash
python3 -m unittest test_import -v
```

测试在本目录创建并清理临时合成材料，不读取服务器结果、不创造研究结果。覆盖篡改、重复/缺失GT、半成品/伪全量、独立实际pair核验、rank选择、无预测语义、tIoU重算和不可覆盖输出。
