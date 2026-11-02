# VrdONE训练/验证划分与未见关系核查

日期：2026-10-15。按用户追加问题读取当前AutoDL代码，并直接统计现有完整训练标注zip与验证标注；没有训练模型。

## 代码怎么划分

`configs/vidor.yaml`第82–89行：训练配置split=training；命名为test_dataset_config的评估配置实际split=validation。`dataloaders/vidor.py`第20–21行只接受这两个split；第50行、第86–95行从ann_dir/split逐视频JSON枚举。不进行关系实例层面的随机训练/测试拆分。训练shuffle和随机采样起点不改变视频归属。

实际读取训练7,000视频、267,210条关系标注，验证835视频、30,142条关系；视频ID交叉0。当前研究的30,142条属于验证集，不能称未曝光的最终官方test结果。

## “没见过关系”的不同含义

| 层次 | 实际结果 |
|---|---|
| 谓词类别，如watch/hold | 训练50种、验证50种；未见验证谓词0/50，受影响实例0/30,142 |
| 类别三元组，如cat-watch-fish | 训练6,258种、验证2,410种；验证中未见组合12.24%（295/2,410） |
| 属于未见组合的验证关系实例 | 2.13%（641/30,142） |
| 短关系中的未见组合实例 | 2.00%（233/11,621） |
| 新视频、实体实例和发生时间 | 视频划分不同；这是正常的泛化内容，不意味着谓词未见 |

具体未见组合例子：`baby-point_to-toy`有14条验证实例；`cat-watch-fish`有6条；`child-feed-cat`有6条。这里类别都属于固定词表，但组合在训练原始标注中未出现。模型并非只能输出见过的三元组，原VrdONE仍命中了其中11条。

## 复用原Top100预测的描述统计

| 验证集分组 | VrdONE命中 | BIG命中 |
|---|---:|---:|
| 训练标注出现过的类别三元组，29,501条 | 14.52%（4,285/29,501） | 10.01%（2,954/29,501） |
| 训练标注未出现的类别三元组，641条 | 1.72%（11/641） | 0.94%（6/641） |

这是两个已有预测集合的GT命中比例，不是分类accuracy，也不是共享输入的受控模型对照。未见组合与长尾、大小、轨迹质量等因素混合，不以此证明未见组合的独立因果效应。

未见类别三元组涉及的短关系只有233条，所以仅以“训练原标注未见该组合”不足以解释大部分短关系漏检。

## 原标注出现不等于每个实例实际进入训练

以下为实际loader过滤，不能忽略：

- 默认cut_max_preds=True时，实体对的关系实例数超过9会跳过该对（vidor.py:371）。
- stride=4采样后主体特征不足2点会跳过该对（:390）。
- 关系映射到采样网格后起止相同，会丢弃该关系（:428）。
- 后续truncate_feats还会裁剪（:445）。
- min_frames_th=15在该loader标为not used，不能据配置名宣称训练删除了所有不足15帧关系。

本次统计回答**完整训练标注里是否出现**，不声称已重建作者每次训练的实际采样曝光。固定50类输出也不意味着每类训练充分。

我们075的小样本训练继承作者checkpoint，仅微调mask投影；32个微调视频不等于模型从未见过其余官方训练视频。代码还检查train、adapter_dev与评估视频不交叉。已反复使用的validation仍不能重新叫独立泛化测试。

## 可复核文件

- 实际代码快照：`handoff/incoming/VIDOR-RESEARCH-QUESTION-077/source_snapshot_vrdone/`。
- 真实运行结果：`handoff/incoming/VIDOR-RESEARCH-QUESTION-077/remote_v1/split_coverage_v1/summary.json`、`per_gt_coverage.csv`。
- 脚本：`analysis/research_question_077/split_coverage.py`。
- 来源：`commander_training_support_040b/data/training-annotation.zip`，当前VidOR validation JSON与073逐GT表。7,000训练标注逐个读取；30,142验证GT索引和类别三元组逐条一致。源文件SHA已保存。
