# H101：长共存网格上，均值归约的点式 mask focal 是否稀释短事件正支持？

2026-10-21。**DESIGN_ONLY → 交 E101 执行**。**NOT_VERIFIED_CAUSE**。Grace/supervisor 在 Codex `usage_limit_exceeded` 后接管撰写。

用户硬交付仍未完成：①实验验证的完整原因；②针对该原因、文献对齐的架构。本文件是一条完整、可操纵的因果假设，不是已验证根因。

## 1. 必须继承的停止结论（不得重开）

| 实验 | 出口 | 对新假设的约束 |
|---|---|---|
| E091 | `NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE`；几何辅助 CDF/W1 主配方未达 +5pp | **不复活** A/B/S 几何辅助项；不得把本轮写成“再试一次几何距离” |
| E097 | `REJECT / SCOPE_CAPACITY_CONFLICT`；旧 F/原 H 下 oracle≤149/334 | 不在旧单槽容量下要求多数净恢复；沿用已验收原生 9 槽 |
| E098 | `REJECT / UNDERREADING_NOT_ESTABLISHED`；严重低读取 47/334≈14% | 不复活联合行列 attention 归一化；不预设“没读到” |
| E099 | `REJECT / INTERFERENCE_NOT_ESTABLISHED`；分类梯度路由 C0 短组收益 CI 跨零 | 不投影分类梯度；不把负内积当伤害证明 |
| E100 | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE`；路径联合匹配 T−N 334 净增 +6/0/−2 | **不重开跨层分配一致性**；层间目标一致不是本轮操纵 |

086–089 仍成立：残余是**区间形成失败**（谓词符合仍大量失败）；融合损害、拆 hull 主补救、换 query 主因、长语义压制均未建立。见 [090 综合](../mechanism_synthesis_090/VERDICT_UPDATE.md)。

## 2. 恰好一条新原因假设 H101

> **H101：在长共存实体对的有效时间网格上，原生 mask focal 对每个已匹配事件做时间维 `mean` 归约，使短 GT 的少数正时间点在该事件损失中被大量负点稀释；共享 mask 头因此对短区间边界与紧致度的有效梯度不足。在保持 Hungarian 身份、Dice、分类损失、容量、推理 hull 不变时，把每个事件的正时间点总质量归一成单位质量（duration-normalized positive focal），应比原生均值归约、以及故意偏袒长正支持的对照，更稳定地改善残余短区间命中，并保护长关系。**

完整待检链条：

| 环节 | 内容 | 身份 |
|---|---|---|
| 输入结构 | 同 pair 长网格上短 GT 正点数 ≪ 有效长度 | 成员定义下的源码事实；稀释幅度待测 |
| 可操纵原因 | 点式 focal 的 `mean(1)` 归约对短正支持不公平 | **本轮唯一干预** |
| 中间机制 | 固定诊断 query 上短组局部分数/AUROC 弱于长组 | 待测 |
| 输出 | 残余短区间漏检或过宽 | 既有 334 残余为观察；因果主导性待合同 |
| 特异预测 | 同视频同谓词短−长增益差为正 | 必须由 §5 检验 |

**不是：** 跨层换事件（H100）、分类梯度打架（H099）、注意力读取总量（H098）、外加几何距离项（H091）、拆 hull（088）、换 query（089）。

源码锚点：[`losses.py`](../cause_arch_099_e099/source_snapshot/losses.py) 中 `masked_sigmoid_focal_loss` 在 valid 掩码后执行 `loss.mean(1).sum() / num_masks`；Dice 已是逐事件归一。本假设针对 **focal 事件内正支持稀释**，不否认 Dice 已有的集合归一。

## 3. 与 090 单位质量草案的关系

[090 M090-UNIT-MASS](../mechanism_synthesis_090/disruptive_proposal.md) 同时改监督形状与读出（分布 + 矩解码）。E091 已否定“在二值 mask 上叠加几何辅助”主配方。H101 **只改 focal 正点质量归一**，保留二值目标、Dice、原生 hull 推理，以便单独证伪“均值稀释”这一环；若 VERIFIED，再考虑是否升级到完整单位质量表示（那是后续架构，不是本轮并行第二假设）。

## 4. 唯一升级实验

只设计并执行 [E101-DURATION-EQUAL-FOCAL](EXPERIMENT_CONTRACT.md)：N 原生 / T 单位正质量 focal / X 长偏置 focal / C0 零更新。全部互斥出口写在合同内。

**自检：一条 H101、一个 E101、不重开 H100、不并行第二假设。**
