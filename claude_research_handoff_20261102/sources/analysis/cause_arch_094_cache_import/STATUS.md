# 094 E091材料导入与验收

**CACHE_GATE_PASS** — 2026-10-20。仅材料通过；A/B/S训练步数为0，E091评价未运行。

## 材料范围

- 本地根目录：`/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-E091-GEOMETRY-CONTRACT-092/cache_v1`。
- 远程组装目录：`/root/autodl-tmp/VrdONE-data/commander_cache_import_094/cache_v4`；原087 raw和预训练checkpoint未覆盖。
- 已逐文件核验 1821 个manifest负载文件，共 1,372,319,607 字节。另封存manifest自身SHA与本地原协议快照。
- 分片保留原生`f/hs/valid/cls/masks`的数值和dtype。训练/开发附独立事件`preds/targets`；评价时间target由原GT27标注、锁定清单与坐标连接，完整标注存于`sources/annotations/`，未用50类union probe替代。

| 必备项 | 路径／映射 | SHA256 | 字段 |
|---|---|---|---|
| 训练native cache | `cache_v1/train/`，88个分片；[train索引](train_file_index.json) | `2d48a0b2780691a47da97d72c8bc4c2e880d83bd1c037af961c05b9d50888faf`（逐文件SHA索引自身） | 齐全 |
| 开发native cache | `cache_v1/adapter_dev/`，23个分片；[adapter_dev索引](adapter_dev_file_index.json) | `9a62a3b6919c6e8597b05aa7adf99b407f77d61d7cca6e48fda056ada17e5c3f`（逐文件SHA索引自身） | 齐全 |
| 评价native cache | `cache_v1/eval/`，1614个分片；[eval索引](eval_file_index.json) | `80cd780d7b29f00b5219060518d48fee38dbd53ba880fd848354d15ef7cc0c2d`（逐文件SHA索引自身） | 齐全 |
| frozen_matching.json | `cache_v1/frozen_matching.json` | `00f033838b3503ce0b73b216163d6c9e94310b5341a63c849b89ac4fd22a0209` | 齐全 |
| initial_mask_embed.pt | `cache_v1/initial_mask_embed.pt` | `f5a3ae513a285bb0832eac122ee13142601438a98c44616444fd76ba049973fb` | 齐全 |
| manifest.json | `cache_v1/manifest.json` | `2613d524a528528212cb28c1d2a543431d7fc9b08cfc9d61f324a6aebd2f2269` | 齐全 |
| forward_consistency.json | `cache_v1/forward_consistency.json` | `71759efafd4df231ccaf0dfd9ec0c355fb3757290e134dcc765d7875bf28c839` | 齐全 |
| tensor_schema.json | `cache_v1/tensor_schema.json` | `2ba8fa53a9ee46c56fe94ddcee97cf7715a03834a949d12312bf21f353f896f8` | 齐全 |
| event_support.csv | `cache_v1/event_support.csv` | `23a5b6fd7f63e6114c932dbc2549676a68d7c87ef204832823b7b93b76df0de2` | 齐全 |

分片的每个SHA、源路径、字节数见三个索引及原manifest；不是仅校验压缩包。
逐项机器回执：[material_audit.json](material_audit.json)。

## 支持账与固定协议

| 分组 | 视频 | 实体对 | 独立事件target | 至少2个正点 | 仅1个正点 | 零正点 |
|---|---:|---:|---:|---:|---:|---:|
| train | 32 | 350 | 563 | 561 | 2 | 0 |
| adapter_dev | 8 | 89 | 169 | 168 | 1 | 0 |

上述是本缓存实际支持，不能推广为整个数据集无采样缺失。裁剪、网格起止与GT边界差异、采样种子、offset、随机抽样记录和`censored_positive_cells`逐pair/event保留在`pair_sampling_support.json`与`event_support.csv`。网格端点不等于GT端点不自动解释为裁剪。

- train/dev/eval视频互不重叠；评价保留27视频、2573条原锁定GT，全部有固定pair缓存。
- 评价native cache共1614个pair，原锁定GT使用其中890个pair；保留全部1614，不依效果选pair。
- 原400/163成员、固定pair_idx和F query保持；原H准入381/400，其余19/400仍留主分母。
- 关系≥3秒且共存≥10秒、可复用固定读出的保护组为1101条；仅封存支持，不报告新性能。
- 439个训练/开发pair匹配记录覆盖732/732事件，每target恰好一次，同pair query唯一，目标数不超过9。匹配绑定初始化及每个缓存SHA，未来A/B/S共用。
- 原共享mask embedding初始化来自预训练EMA checkpoint，非继续训练权重；未来若执行E091只将其更新作用于decoder层3，早期层输出保持缓存。

## 数值验收

1725/1725批缓存（88 train + 23 dev + 1614 eval）重算mask/class分支与同源原生输出的最大绝对误差均为0；既定容差保持4e-6，mask二值、分类top1/top6一致。匹配使用的重算与缓存张量完全相同，故Hungarian输入一致；匹配覆盖另经CPU核验。

本轮用GPU做冻结mask/class分支重算验收；未重跑encoder，未新导出视频前向，未执行optimizer，未训练任何一臂。不能把缓存PASS写成机制或方法有效。

## 异常与脚本修正均保留

1. v1误用raw logit跨query排名；改为089既定的全51类softmax概率，旧query不变。
2. v2额外要求全部2573条旧F等于当前argmax；原091只要求保留F，089排序一致性范围为400。v4保留全GT旧F，仅对原400执行排序检查。组外1条发生微小排名漂移：`1161_8966147395/GT3/pair2`，旧q0概率0.3675785877732299，q1概率0.3675795173808527；仍使用q0，未删样本。
3. v3将成员表字符串GT ID与整数比较，造成主组连接为空。v4按历史089做规范化，恢复原400/163；没有改成员或放宽数值门槛。

旧脚本、失败日志、修正说明保存在本目录，远程v1/v2/v3输出未删除。上述为验收实现错误，不是模型效果负结果，也不掩盖组外排序漂移。

## 来源与执行边界

原协议快照与SHA见[protocol_sources.json](protocol_sources.json)。远程实际使用043来源合同SHA与本地副本一致：`dd4be25bd4f35a394c387d0905ef747d0a3c051f7f62e3d0931244caa6525bd9`。本地文档批量上传被自动审批拒绝（未单独授权上传）；改在本地封存，无相关文档外传。

状态：仅`CACHE_GATE_PASS`；092当时的`NOT_RUN_MISSING_CACHE`历史记录不改写。RQ不变，M091未采纳，E091未启动，不自动排队下一项。**094到此停止。**
