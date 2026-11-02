# M011可见正区间监督：下一最小对照准备

状态：**未执行、未验证**。只准备修复自建M011的单项监督缺口，不是新架构、正式主方向或CVPR创新。031评分复验已经失败；本准备不改变其结论，也不根据029读出探针批准新头训练。

## 已有依据

M011对mask覆盖全部输入窗的匹配目标将两端标为unknown，partial endpoint NLL为零，该query也不进入unmatched KL。017导入记录及Commander重计：训练222/563属于此组，其中212/222在采样/crop后完整保留；短目标62/200属于此组。开发组为50/169。详见 `analysis/training_sampling_017/M011_SUPERVISION_FOLLOWUP.md`，不是整个原VrdONE训练的普遍结论。

## 唯一变化与数学边界

保留原known flags、partial NLL、未匹配KL、架构和推理解码。仅对原both_unknown匹配行增加可见GT正点的平均负对数覆盖概率，再按实例平均，系数固定1，不扫描权重。

原窗内联合端点分布为 P(s,e) ∝ exp(S_s+E_e)·1[s≤e]。已见正点t提供关系在t存在的信息，因此

`log P(s≤t≤e) = logcumsumexp(S)[t] + reverse_logcumsumexp(E)[t] − logZ`。

先用小矩阵枚举核对概率和梯度。这个软覆盖项没有把窗外真实端点声明为已知。模型仍只允许窗内端点，其最优窗内输出自然趋向覆盖可见正窗；若使用整段硬覆盖事件，则全窗正例退化为首点/末点完整端点监督，不能包装成新的窗外删失似然。

## 固定配对设计

两臂：local原损失、local原损失加软覆盖。从相同seed11初始state各训练2000步，固定样本、批次顺序、匹配、AdamW、学习率和梯度裁剪；参数数目仍35682。准确说法是末层零初始化，其余随机参数也须相同。

不从旧local终点续训；旧pooled checkpoint只重评价作为低成本参照，无需本轮重训pooled。该设计不能主张局部视觉比pooled更有效；要提出该主张才需要完整交叉对照。

可复用：
- 原runner：`analysis/vrdone_architecture_011/remote/software_snapshot_v2/analysis/vrdone_architecture_011/prototype/train_interval.py`。
- 训练代码同目录 `training.py`，联合归一化同目录 `structured_interval.py::joint_log_partition`。
- 原AutoDL缓存：`/root/autodl-tmp/VrdONE-data/commander_architecture_010/export_v1/{train,adapter_dev}/`。训练88批、350pair、563目标；开发23批、89pair、169目标；执行前再核manifest和批次SHA。
- 固定匹配SHA：`49c164a0b73007dcc3a6b945a62813606c018e15eba09390b9675dbb8d87df3d`。
- 旧初始化全state SHA：`8912e0072a5d63fa61ed2dab595504d54bdab6fb6ccae614629213cec6c44684`。
- pooled checkpoint：`analysis/vrdone_architecture_011/remote/training_results_v2/training_v2/pooled/interval_last.pt`。

旧local训练实测87.25秒，仅作为小头训练预算参照，非当前耗时保证。基座不重新前向、不训练；现有环境，不改共享数据或旧checkpoint。

## 结果前待正式冻结的推进／停止门槛

先两臂训练和全部开发集评价，暂不启动全验证集推理。

1. 新项只作用固定训练222条、开发50条；匹配、初始化、旧损失、样本与批次顺序对账。否则技术失败。
2. 50条的平均可见tIoU不提高，或改善实例数不多于退化实例数：停止“此缺口是可修复重要退化来源”的解释。
3. 即使50条提高，若全部169条平均tIoU或tIoU≥0.5通过数下降，停止该修复版本。若总体还不达到旧pooled或同环境0步anchor的平均tIoU及tIoU>=0.5通过数，仅记录局部修复，不扩大训练。
4. 通过后才做实际检测评价，同时看mAP、Top100、所有增失及C恢复；开发集tIoU提升不能替代检测收益。

历史开发tIoU参照：local0.598849，pooled0.691597，初始anchor0.701495。不能用历史数值替代本轮同环境对照；也不应反复用169个开发目标调参后声称独立泛化。

下一步仅进行代码/数据可复用性核查、有限实现及协议审查；正式冻结后才运行这两臂。没有主方向或主方法采用决策。
