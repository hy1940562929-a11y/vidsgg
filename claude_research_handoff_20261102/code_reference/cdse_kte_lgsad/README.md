# CDSE / KTE / LGSAD：完整候选实现

状态：**UNVERIFIED_RESEARCH_DESIGN**。这是对本次讨论方案的代码化，不代表方法有效、新颖性已成立或达到投稿标准。所有随包测试都是 CPU 合成张量的软件检查；没有运行真实训练、验证集评估、GPU 实验或远程命令。

## 文件

| 文件 | 用途 |
|---|---|
| `relation_enhancement.py` | 三个模块、时间窗参考、逐谓词时刻/边界监督、同对同谓词排序损失、可选扰动一致性损失、软评分融合、文本缓存检查 |
| `vrdone_adapter.py` | 接入原 `MaskVRD`：高分辨率 query 读取、FPN 残差融合、类别/mask/边界软评分、原匹配与损失接口 |
| `integration_example.py` | 在已有训练代码中构建模型的完整函数；不启动训练 |
| `prepare_text_anchors.py` | 用已有 CLIP 安装和本地 checkpoint 生成真实文本向量缓存 |
| `config.example.json` | 候选参数；`seconds_per_step` 必须按真实数据填写 |
| `smoke_test.py` | 合成输入前向、反向及有限数检查 |
| `test_enhancement.py` | 核心模块回归检查 |
| `test_vrdone_adapter.py` | 对真实导入的 VrdONE 源码运行随机权重的接口检查 |
| `VALIDATION.json` | 本次软件检查记录，不是实验 Evidence |
| `SOURCE_MANIFEST.json` | 检查使用的原模型源码和交付文件哈希 |

代码包不包含原 VrdONE 全仓库、模型权重、CLIP 权重或数据集；它提供的是本方案的完整实现及原模型适配器，运行原任务仍使用你已有的 VrdONE 工程和数据接口。

## 实现与讨论方案的对应

### CDSE

- 从原 backbone 的 `feats[0]`（BSA 融合后、下采样前的全分辨率特征）读取 `e_t`。
- 逐谓词预测 `coarse_logits[B,P,T]`，各谓词独立 sigmoid；不是一个全局“是否有任何关系”分数。
- 每个目标时刻从前后 `context_seconds` 内选择目标谓词分数低于阈值的有效位置，排除中心邻域，按 `1-score` 池化参照。
- 用前缀和实现，按谓词分块；不构造 `[T,T]` 或 `[P,T,T]` 矩阵。
- 选择权重停止梯度；找不到足够候选时该处 CDSE 残差严格为零。
- 输出保留原特征，新增可学习有界残差，不执行投影删除或直接相消替换。
- `context_enabled=False` 为默认值。先训练逐时刻预测头，再用验证数据评估/校准背景选择，确认后调用 `model.set_context_enabled(True)`。测试时不读 GT 选择背景。
- 低分仅为模型的预测；不能据此声称选中的位置必然是真背景。平衡 BCE 训练后的分数也不自动成为已校准概率。

### KTE

- 使用主体 ROI、客体 ROI 和相对框几何的时间差分，按真实采样间隔换算每秒变化量。
- 缺失或补齐位置之间不计算差分；所有时间卷积步长为 1。
- 门控按目标谓词受到逐时刻关系监督。它学习的是关系相关变化，不是通过大差分直接认定发生了交互。
- 额外高分辨率 memory 由关系 query 读取，并残差接入原 FPN mask 特征，避免只修 mask 而分类路完全读不到新分支。
- 同一实体对/谓词的“有运动但该关系未发生”应在**已确认标签**中作为负例。可选增强一致性接口接受已对齐、标签不变的扰动输入，代码不会假定任意相机运动增强都保持标签。
- 相对几何和 ROI 差分不保证消除相机运动、检测抖动；它们也无法恢复原采样中根本没观测到的帧。

### LGSAD

- 冻结/缓存真实文本编码器输出，学习视觉与文本的投影及局部条件预测头。
- 支持固定谓词文本 `[P,E]`；独立核心还支持调用方提供的实体对条件文本 `[B,P,E]`。默认原模型适配器使用固定谓词文本，未自动添加实体类别模板。
- 视觉局部特征与目标谓词文本共同产生独立关系成立分数和 start/end 分数；起止位置来自视觉与训练时间标注，不从静态文本中凭空读取。
- 原模型 query 预测的类别权重与上述逐谓词分数进行有界**加性**融合；没有把原 mask 与一个小于 1 的门控直接相乘，因此没有“只能缩小、不能补峰”的形式限制。
- start/end 分数用于原预测区间的软评分，不直接裁边；没有实现或宣称“文本阈值就能精准剪掉误报”。最终区间仍由原模型规则产生。

## 一次运行 CPU 软件检查

在已有 PyTorch 环境中，从本目录执行：

```bash
python smoke_test.py
python -m unittest test_enhancement -v
```

本机检查使用 `/Users/xk/miniconda3/envs/ml/bin/python`，Python 3.11 / PyTorch 2.2.2；未安装或修改任何依赖。

原模型接口检查需指定**已有**源码目录（目录下应有 `models/maskvrd.py`）：

```bash
VRDONE_ROOT=/absolute/path/to/existing/vrdone python -m unittest test_vrdone_adapter -v
```

这些检查不读取真实数据或预训练权重。`smoke_test.py` 会明确标注 `synthetic=true`、`research_result=false`；其中随机文本向量只用于张量运算测试，不是 CLIP 的替代实现。

## 张量和标签约定

独立核心输入：

```python
outputs = enhancement(
    pair_features,       # [B,T,C_pair]，BSA 输出，未进一步下采样
    subject_features,    # [B,T,C_roi]
    object_features,     # [B,T,C_roi]
    relative_geometry,  # [B,T,C_geom]，原规范化相对框特征
    valid_mask,          # bool [B,T]
    seconds_per_step,    # 正数或 [B]；真实特征采样间隔
    context_enabled=False,
)
```

独立核心 `P` 轴为 0-based；原 VrdONE 输入标签为 `1..P`，背景为 `0`。文本缓存第 0 行对应原模型类别 1。输入/GT 必须位于**同一个特征时间网格**，不允许把原始帧 mask 直接当作特征步 mask。

`seconds_per_step = 原始帧采样步长 / 视频实际FPS`。例如只有在 stride=4 且 FPS=30 时才是 `4/30`；512 是特征步长度，不自动等于512原始帧。本适配器假定数据集采样间隔一致，若 FPS/采样间隔混用，应扩展原数据分桶/预处理接口，不能套用单一常量。

### 逐时刻目标与损失

```python
targets = build_dense_targets(
    class_ids=preds_list,       # list[[N_i]]，原类别ID 1..P
    instance_masks=masks_list, # list[[N_i,T_i]]，二值 mask
    valid_mask=valid_mask,
    num_predicates=P,
    known_masks=known_list,    # list[bool[P,T_i]]，未知位置 False
)
losses = EnhancementCriterion()(outputs, targets)
total_auxiliary_loss = torch.stack(list(losses.values())).sum()
```

- 同谓词多个实例的正 mask 取并集，所有同谓词正区间都排除出负例；允许同一时刻多个谓词为正。
- 边界目标保留各实例起止，不只取全视频首末点；窗口裁断、相邻未知位置不强行生成确定边界监督。
- 默认不把所有未标注位置当负例。应提供 `predicate_known_list`；只有检查了标注完整性后，才能显式设置 `assume_exhaustive_annotations=True`。
- 原匹配损失 + `enh_coarse`、`enh_relation`、`enh_start`、`enh_end`、`enh_motion_gate`、`enh_rank`。这些返回值已经乘了各自权重；不要再次重复加权。
- 排序损失使用同对同谓词的全部已知正例与已知负例，负例聚合是 log-mean-exp，不因共存窗口更长而自动增加惩罚。
- 可选一致性项只在调用方提供标签/时间对齐不变的增强时启用；绝不使用测试标注计算训练损失或选择背景。

## 准备真实文本向量

用原数据集类别表生成 `predicates.json`：必须是原前景类别按 ID 1..P 排列的名称列表，不能自行改顺序。代码没有猜测 VidOR 的类别顺序，也没有附一份伪造的类别表。

在已有 OpenAI CLIP 环境中使用一个**本地 checkpoint 文件**：

```bash
python prepare_text_anchors.py \
  --predicates /absolute/path/predicates.json \
  --checkpoint /absolute/path/local_clip_checkpoint.pt \
  --output /absolute/path/new_predicate_text_anchors.pt \
  --template 'a subject {predicate} an object'
```

支持重复 `--template` 后取归一化平均。脚本只用 CPU；不会下载模型、安装 CLIP、覆盖已有输出。缓存记录 checkpoint 哈希、模板、类别顺序和 `synthetic=false`。

本机未安装 CLIP，也未提供真实 CLIP checkpoint；因此真实文本向量导出未在本次执行。主模块接受缓存向量，无需把文本编码器放进每步训练。

## 接入你现有的原模型训练代码

把本目录加入已有工程的 `PYTHONPATH`，调用：

```python
from integration_example import build_model

model = build_model(
    vrdone_root=existing_vrdone_root,
    model_config=model_config,             # 原配置 model_config 字典
    native_state_dict=original_model_state_dict,
    anchor_path=real_anchor_cache_path,
    predicate_names=native_foreground_names,
    seconds_per_step=actual_seconds_per_feature_step,
    device=your_existing_device,           # 由使用者原有代码决定，不自动抢GPU
    assume_exhaustive_annotations=False,
)
model.train()
loss_dict = model(training_batch)          # 原字段 + predicate_known_list
loss_dict['total_loss'].backward()
```

`build_model` 是完整函数，参数接你的已有加载代码；上面变量代表原工程的真实配置、数据和权重，未把随机初始化冒充预训练模型。调用后再创建 optimizer，使新增参数进入优化器。推理照原接口 `model.eval(); model._config_eval(inference_config); model(video_batch)`。

可选 nuisance 路径：`training_batch` 添加 `so_features_aug_list`，并显式设置 `augmentation_preserves_relations=True`；增强列表必须与原样本同顺序、同长度、同实体对、同关系标注。不要将随机剪时、对象交换等标签可能变化的输入作为一致性增强。

### 原模型兼容性处理

- 按已导入的 `MaskVRD` 接口实现，标准视觉输入和 `with_clip_feature=True` 的布局均有接口检查。
- 初始新增残差强度为0，固定原权重且 eval 模式时，新旧预测应完全一致；检查使用随机权重证明的是软件兼容，不是性能等价实验。
- 原有 query 数、前景类别、背景类、匹配策略和下游阈值/首末正点成区间/最短长度/TopK/排序预算保持由原实现控制。仍可能发生多个正段被首末点包在一起；代码没有声称彻底解决该问题。
- 原模型每个 query 仍共享一条时间 mask；本实现按类别概率混合逐谓词证据，没有新增每个 query、每个谓词各自独立的实例 mask。
- 检查用快照中，`forward_training` 传入 `loss` 的位置顺序与方法签名不一致；另有 TorchScript 损失将 `num_masks` 推断为 Tensor，而调用处传 float。本适配器用关键字调用和同值标量 Tensor 修正这两个兼容问题，不改原快照文件。**`enabled=False` 的基线对照也共用相同修正，不得把修复带来的变化算作研究贡献。**
- 同一GT网格上的未补齐 mask 只补零后交给原 matcher，不插值、不改变事件跨度。
- 关闭分支/上下文 warm-up 会产生未使用的参数；使用 DDP 时需 `find_unused_parameters=True` 或在该阶段明确冻结未使用模块。
- 原模型训练脚本、数据集 loader、远程运行器、完整GPU实验命令均未改写。真实实验仍通过人工 handoff。

## 建议的受控比较

1. `enabled=False`：原预测分支，同样兼容修复。
2. 分别关闭 `use_cdse` / `use_kte` / `use_language`。关闭语言时使用可学习类别 ID 向量作为控制，不宣称实现了跨模态对齐。
3. 在独立核心中传入固定置乱的文本向量作为语义条件对照，类别标签与输出轴顺序保持原样。
4. 普通局部差分、同参数高分辨率分支、正确/错位上下文等机制对照应采用同等训练/输出预算；本包不自动开展这些实验。
5. 分开报告短关系、持续关系、长共存下短关系、总体指标，以及精确率/召回率变化。损失下降、分支有梯度、或概率被缩小，都不等于机制已验证。

## 来源与边界

原框架：[VrdONE](https://arxiv.org/abs/2408.09408)、[作者代码](https://github.com/lucaspk512/vrdone)。差分和细粒度对齐分别有 [TDN](https://arxiv.org/abs/2012.10071)、[LocVTP](https://arxiv.org/abs/2207.10362) 等先例。本实现是本次候选设计，不是这些论文的官方复现，也不是三个首创模块的证明。

未包含任何正式实验 Evidence ID、数据集效果数字、投稿就绪声明或隐式远程执行。
