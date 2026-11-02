# 092 材料门槛判定

**NOT_RUN_MISSING_CACHE**

任务：仅执行E091-GEOMETRY-CONTRACT。已核对091 §5、几何投影P与复审合同；本轮在训练材料门槛停止。训练步数0，新增模型前向0，GPU任务0，无新预测或性能数字。

## 本轮实际核查

- 本地工作区文件清单未找到完整`train_cache.pt`、`adapter_dev_cache.pt`、`train_fresh_layers.pt`或`adapter_dev_fresh_layers.pt`；实际路径清单见[material_audit.json](material_audit.json)。这是当前工作区范围内的核查，不声称扫描整台电脑或证明远程文件不存在。
- 087的[导入回执](../mech_minifalsify_087/import_receipt.json)明确将训练/开发特征缓存和大部分`fresh_raw`保留在远程。本地仅有`preflight_v1/fresh_raw/adapter_dev_0022.pt`这一原生小批次文件，不覆盖32训练/8开发视频。
- 本地087分数NPZ实查字段为`frames/fine/fused/position/native_masks/native_classes`。它们不能替代原mask embedding的同源query输入和时序特征。087探针checkpoint也不是原VrdONE分支初始化。
- 043/078的历史训练审计JSON在本地，但对应全套可训练缓存未导入；审计通过记录不能替代张量。079旧层缓存还存在历史一致性失败，不能据此绕过本轮数值门槛。
- `/private/tmp`没有AutoDL复用socket。对用户最近给出的`connect.bjb1.seetacloud.com:16241`只做一次非交互网络可达性检查，返回`Connection refused`，未认证、未读取远程文件。此前沙箱内DNS失败另经许可在沙箱外确认了连接拒绝。**远程缓存存在与否仍未知，不能写成远程已检查且缺失。**未连接学校服务器，未请求密码。

## 精确缺项：需要交付的材料

以下是建议导入产物名，不表示这些文件已经生成；允许保持旧文件名，但manifest必须建立一一对应。

| 建议产物 | 必须包含的内容 | 当前缺口 |
|---|---|---|
| `train/native_cache.pt` | 固定32视频/350pair/563历史GT的同次前向：末层mask embedding输入`hs`、mask分支时序输入`f`、valid mask、原生分类与mask logits、逐事件targets/predicates、pair/event ID、原帧坐标及采样/裁剪记录 | 全套训练张量本地未导入；有效支持数须核验，不能直接套历史GT数 |
| `dev/native_cache.pt` | 固定8视频/89pair/169历史GT的同字段缓存 | 只有单批次preflight原始文件，不是完整开发集 |
| `frozen_matching.json` | 同一基线、同一缓存的一次Hungarian结果，query→独立事件target映射；不能用同谓词union替代 | 没有与完整092可用缓存绑定的可验证匹配合同；078审计中的匹配索引仅是复用线索 |
| `initial_mask_embed.pt` | 原预训练末层mask embedding初始化与参数名/形状/父checkpoint SHA | 本地有历史训练后分支及049初始化线索，但没有完成与092同源缓存的初始化闭合核验；不得用训练后权重顶替 |
| `eval/native_cache.pt` | 原400/163固定pair_idx和F query、GT27全部可复用读出、长组保护支持；与训练同模型的`hs/f/valid/logits/coordinates` | 已有分数和GT材料，但缺完整可重算末层分支的评估特征 |
| `manifest.json` + `forward_consistency.json` | 32/8成员与源SHA、与评估无交叉、target有效支持账、tensor shapes/dtypes、坐标单位、初始化重算原生logits/分类/离散输出一致性回执、冻结参数清单 | 旧清单/回执可复用核对，但新材料未齐，不能签发092通过回执 |

字段依据：`analysis/native_mask_training_043/run.py`的Capture/project/reconstruct（`hs/f/valid/cls/masks/preds/targets/metadata`），及`analysis/mech_minifalsify_087/run_probe.py`的fresh_forward。缺少同源输入时无法只训练原末层分支。

## 建议handoff位置（仅建议，未派发、未排队）

- 任务包：`handoff/outgoing/VIDOR-E091-GEOMETRY-CONTRACT-092/`
- 材料返回：`handoff/incoming/VIDOR-E091-GEOMETRY-CONTRACT-092/cache_v1/`
- 优先检查历史远程目录：`/root/autodl-tmp/VrdONE-data/commander_mech_minifalsify_087/results_v1/`中的`fresh_raw/train_*.pt`、`fresh_raw/adapter_dev_*.pt`及评估raw。路径来自导入回执；本轮无法确认当前主机是否仍有该目录。`*_fresh_layers.pt`须检查字段，不能默认含原query/目标/匹配。
- 只补上述材料后才能按冻结合同继续；本轮不重导出、不新增训练臂、不放宽容差、不改成员。

## 合同与结论边界自检

保留A/B/S、末层mask embedding、500步、seed90/91/92、P/C和Z/C0；未执行任何一臂。没有因数据缺失运行分数伪训练或替代实验，没有产生092效果结论。

091复审已经把AF改为NR净恢复对比。即使未来NR下界超过0.5，也不能由此认定完整或主要根因；保持`VERIFIED_INTERVENTION_EFFECT`的限定解释。本轮更没有任何因果升级。

**已停止，等待用户下一指令。RQ不变、未采纳M091、无093队列。**
