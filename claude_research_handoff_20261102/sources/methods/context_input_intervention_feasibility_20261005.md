# 固定目标输入的上下文干预：源码可行性核验

2026-10-05，Explorer只读核查，Commander归档。结论：源码层面可实现，但尚未执行无操作前向准入，不是已存在的科研结果。021 INCONCLUSIVE不提供上下文压制的正证据。

## 输入与干预入口

原VidOR._val_getitem拼接2069维时序输入：主体ROI1024、客体ROI1024、pair空间5、主体框/运动8、客体框/运动8。CLIP关闭。切片分别为0:1024、1024:2048、2048:2053、2053:2061、2061:2069。raw神经路径只读so_features_list；cat_ids、cat_scores、框、轨迹持续期及offset用于构造或后处理，sids控制pair数量/顺序。

最小hook在loader之后、backbone之前：保持完整pair列表、顺序和batch，preprocessing之后clone已padding的X，仅替换指定pair的目标GT外有效列；原M原样传入，目标内全部2069维及padding逐元素锁定。不能先修改框再重新计算loader，因为相邻框差分会改变GT内边界运动通道。

采样点保持原pair_start+offset+4*i。short补到512；long补到原分片最大长度的max_div_factor倍数。必须保持原200-pair分片和short/long映射，不能抽单条long pair另跑后声称padding不变。

## 位置与末端读出

015有效配置use_abs_pe=false、use_rel_pe=false。卷积、window及下采样仍有位置结构。固定入口不保证目标内部表征不变，后者正是待测响应。

最后有效logit为eᵀf，其中e=最后decoder层mask_embed(hs)，B×9×256；f=neck输出mask_features，B×256×Tpad。padding logit另填−10。e不是裸hs或初始learned query embedding。

可侧边记录neck输出、predictor.mask_embed输出和predictor结果；deep_supervision时mask_embed按L×B还原并取末层，再按原short/long映射还原pair。forward_test只收三个固定键，不能假设新增e/f会自动导出。点积重构及hook无操作不改输出都必须实际检查。

## 现成缓存的限制与准入

015 run_probe.save_input明确剔除so_features_list。019仅从保存的mask/class logits生成统计，没有保存2069维入口向量或e/f。因此需要从原GT ROI/proposal重建输入并新forward，不能直接在现成019结果中改上下文。

下一步最多先做技术准入：冻结来源与pair映射，原流程重建输入，无操作forward重现015 raw；加记录hook再次前向，核对输出不变和有效位置eᵀf重构残差。准入通过后才能设计并冻结具体上下文替换、匹配对照和kill criterion。本页未启动前向、训练或新实验。

缓存ROI来自带reference/memory/global机制的MEGA；固定缓存不等于固定完全局部的像素证据。非自然替换造成的分布偏移、query语义变化、仅段外激活改变导致hull机械缩短，仍需独立控制。代数路径分解不是共享上游的独立因果贡献。

## 源码定位

- analysis/vrdone_architecture_010/training_preflight/source_snapshot/dataloaders/vidor.py:672（通道与采样）
- analysis/vrdone_architecture_010/training_preflight/source_snapshot/models/maskvrd.py:161、201、363（模型入口、raw路径、padding）
- analysis/vrdone_architecture_010/new_container_preflight/source_snapshot/utils/misc.py:193（相邻框差分）
- analysis/vrdone_architecture_010/training_preflight/source_snapshot/models/blocks.py:91（mask传播）
- analysis/vrdone_architecture_010/training_preflight/source_snapshot/models/predictor.py:92、100（decoder、eᵀf）
- analysis/vrdone_architecture_010/training_preflight/source_snapshot/models/local_transformer.py:956（时序pos_embed）
- handoff/incoming/VIDOR-TEMPORAL-SUPPORT-015/remote/paired27_reproduction/effective_config.json:18（PE关闭）
- analysis/gt_input_mask_probe_20260929/run_probe.py:92、173（删去入口缓存、重建gi）
- analysis/temporal_alignment_019/diagnose_alignment.py:196（只读既有logits）
- analysis/vrdone_architecture_010/cache_builder/export_frozen.py:217（已有f/logit导出范式）
- analysis/gt_input_mask_probe_20260929/source_snapshot/feature_extractor_vidor.py:20（MEGA上下文）
