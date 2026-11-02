# 缺数据时的恢复清单

本地轻量包不含native_cache、GPU资源、训练checkpoint。先只检查用户给定的合法材料位置，不扫描他人目录，不自动SSH。

恢复后逐项核验：
1. 原始checkpoint（预训练初始化而非先前续训），SHA、配置、类别表。
2. train/dev/eval官方视频ID及角色，特征来自GT轨迹还是native proposal，不能混用。
3. 同源query特征、mask branch输入、原生logits、valid/output_mask、时间坐标、完整事件target、冻结匹配及初始化。若要训encoder/decoder还需其真实输入或原模型，不能靠mask分数NPZ替代。
4. frozen_matching.json、initial_mask_embed.pt、manifest.json、forward_consistency.json及每份缓存字段映射。细节参照随包094检查和 `sources/docs/deliverables/CACHE_IMPORT_CHECKLIST.md`。
5. 原生重建同输入/同布局/同dtype/同padding，max abs≤4e−6；类别排序、二值mask和valid一致。记录实际误差/批次/非有限值，不静默改容差。
6. 目标谓词训练正/负支持、dev可识别短/长与视频数；无支持事件保留在总账，不能偷偷删除。
7. 官方原预测、原始vIoU/evaluator、逐GT命中与最终候选，才能开展公平正式比较。

缺项精确写MISSING:<字段/文件>及需用户提供路径；仍可交付代码和TBD论文，不可编造结果。恢复路径历史建议：`handoff/incoming/VIDOR-E091-GEOMETRY-CONTRACT-092/cache_v1/`。该名称不保证本包附带它的张量。
