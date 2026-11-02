# E091缓存导入清单

日期：2026-10-20｜093本地交付附件。

用途：服务器可用后，先判断原E091能否恢复。**这是材料清单，不是新的实验任务或启动指令；本轮不连接远程、不导出、不运行模型。**

来源：[092缺项记录](../../analysis/cause_arch_092/STATUS.md)、[091执行合同§5](../../analysis/cause_arch_091/CAUSE_STATEMENT.md)、[复审要求](../../analysis/cause_arch_091/REVIEW.md)。

## 1. 先定位原材料

- [ ] 优先核对历史目录 `/root/autodl-tmp/VrdONE-data/commander_mech_minifalsify_087/results_v1/` 中的 `fresh_raw/train_*.pt`、`fresh_raw/adapter_dev_*.pt` 及评估原始文件。目录来自历史回执，当前存在性未知。
- [ ] 检查 `train_fresh_layers.pt` / `adapter_dev_fresh_layers.pt` 的实际字段；不能凭文件名认定包含query输入、独立事件target和匹配。
- [ ] 先对照原32训练、8开发视频清单，不重选视频；评价仍为原400/163与GT27可复用成员。
- [ ] 如果旧文件已经足够，保持原文件名和内容，通过manifest映射；不为统一文件名而改变数值。

推荐返回根目录：

`handoff/incoming/VIDOR-E091-GEOMETRY-CONTRACT-092/cache_v1/`

092提到的 `handoff/outgoing/VIDOR-E091-GEOMETRY-CONTRACT-092/` 仅是原任务包路径建议；093没有创建或派发第二份任务包。

## 2. 必备文件与字段

下列名称为建议逻辑布局，不表示文件已存在。

| 产物 | 必备内容 |
|---|---|
| `train/native_cache.pt` | 固定训练成员的同次前向：`hs`（原末层mask embedding输入）、`f`（mask时序输入）、`valid`、原生`cls/masks` logits、逐事件`preds/targets`、`metadata` |
| `dev/native_cache.pt` | 开发集同字段完整材料；本地单个`adapter_dev_0022.pt`不能替代 |
| `eval/native_cache.pt` | 固定评价pair/query的同源特征和logits、GT27可复用成员、长组保护材料；不能只有最终分数 |
| `frozen_matching.json` | 一次基线Hungarian分配：split、video、pair、event、query、target索引、层号，以及所依据的缓存/初始化SHA；每个target对应独立事件 |
| `initial_mask_embed.pt` | 原预训练分支初始化；参数名称、形状、dtype及父checkpoint SHA；不能使用续训后分支代替 |
| `manifest.json` | 相对路径→SHA256→来源；数据划分、固定成员ID、tensor schema、实际支持和缺失原因、坐标与采样信息 |
| `forward_consistency.json` | 初始化分支重算输出与同源原生输出的核验回执：实际误差、采用的既定容差、离散输出是否一致、冻结参数清单 |

`metadata`至少能追溯：video ID、subject/object身份、原pair_idx、query层/索引、event ID/谓词、原帧坐标、fps、valid范围、stride/offset、裁剪及随机采样记录。时间单位及区间闭开约定必须显式记录，不凭数组长度猜。

参考原字段实现：[043 Capture/project/reconstruct](../../analysis/native_mask_training_043/run.py)、[087 fresh_forward](../../analysis/mech_minifalsify_087/run_probe.py)。保留原tensor维度和层轴说明；只训练最后时间分支，不把早期层混进训练。

## 3. 文件与SHA验收

- [ ] 每个原文件计算SHA256，记录源路径、字节数和导入相对路径；导入后再次计算并一致。
- [ ] 原协议文件、源代码、配置、父checkpoint和初始化权重都有来源SHA；不只给压缩包一个总SHA。
- [ ] 历史32训练视频/350pair/563GT、8开发视频/89pair/169GT与固定清单对齐；这些是历史计数，不能直接冒充最终有效损失支持数。
- [ ] 训练、开发、评价视频无交叉；不在已曝光400或GT27中重新划训练集。
- [ ] 原400/163成员、pair_idx及F query不变；固定H准入381/400，其余19条仍留主分母。
- [ ] 缺失、零正采样点、裁剪和有效监督支持逐条入账；不能按效果删成员。

## 4. 匹配和数值门槛

- [ ] GT是逐事件target，不是同谓词多个事件的union；087的50类probe不能替代。
- [ ] 一次基线匹配与完整缓存、权重、配置绑定；A/B/S共享，训练中不重匹配。
- [ ] 初始化重算原生mask/class输出及离散结果满足既定一致性要求；记录实际误差，不改容差追求通过。
- [ ] encoder、时序输入、query、分类、轨迹和匹配冻结；允许更新的参数只属于原末层mask embedding。
- [ ] 所有输出先封存再评价；推理不使用GT时间选query、分量、区间或参数。
- [ ] 能同时提供零更新Z(P)及C0(C)；不能只拿训练后的A/S作参照。

这里列的是将来的材料验收要求；093没有执行前向或替这些检查签发PASS。

## 5. 什么情况下必须停

| 情况 | 状态与动作 |
|---|---|
| 完整训练/开发/评价缓存不可用 | `NOT_RUN_MISSING_CACHE`；保存缺项，不用分数伪训练 |
| 有材料但来源、匹配、数值或容量不一致 | `TECHNICAL_FAILURE`；保留错误，不放宽协议 |
| 全部门槛通过 | 材料可供原E091继续，**不等于实验已完成或获益已验证**；是否执行依当时用户指令，不自动启动 |

本清单不新增训练臂、不另设消融、不改RQ、不采纳M091、不排队094。完整实验和统计出口仍由091 §5唯一规定。
