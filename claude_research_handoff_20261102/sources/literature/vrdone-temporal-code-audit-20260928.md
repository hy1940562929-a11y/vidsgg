# VrdONE 时间离散化与过滤：公开源码核查

日期：2026-09-28。Explorer只读核查公开`main`固定commit `74981850ae719ac3cb9f407f0bb1e7a9ea04388d`，Commander整理；没有运行模型。这不是用户AutoDL实际运行版本的证明。

## 确定的公开代码行为

| 环节 | 规则 | 固定来源 |
|---|---|---|
| VidOR默认 | `feat_stride=4`、`pred_min_frames=5`、最终`n_max_pair=200` | [config](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/configs/vidor.yaml#L116-L121) |
| 训练GT | 相位随机0..3，GT左右边界用ceil映射到采样网格；l>=r删除，r-l=1保留 | [loader](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/dataloaders/vidor.py#L420-L457) |
| 推理边界 | mask概率>0.5，取首末正点；start=first*stride+offset，end=last*stride+offset+1；原帧跨度<5删除 | [decoder](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/models/maskvrd.py#L281-L301) |
| 实际TopK | 官方shell传6；直接运行eval.py的CLI默认10且覆盖YAML，须核对真实命令 | [shell](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/scripts/eval_vidor_ckpt.sh#L5-L10)、[eval](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/eval.py#L35-L54) |
| 最终排序 | 在上述过滤之后，按subject/predicate/object分类分数的算术均值取全视频Top200 | [sort](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/models/maskvrd.py#L319-L328) |

**代码逻辑推导**：默认输出区间长度为`4*(last-first)+1`。单个正采样点还原为1帧，必被默认5帧门槛过滤；训练监督却允许单点GT mask。这里存在训练目标与最终保留输出之间的支持范围差异，是一个可直接追踪的机制候选。它可能是有意的精度/召回取舍，尚不能称作实现bug，也不意味着所有单点mask都应保留。

多个不连续正段会被首末正点包成一个跨度；是否伤害具体短GT必须读取raw mask。FPN内部上采样恢复到特征网格，最终解码没有先将mask插值到原始帧再确定边界。[FPN](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/models/fpns.py#L229-L257)

## 必须区分的其它规则

- `min_frames_th:15`等部分YAML字段被当前loader标为not used，不能宣布它们导致GT被过滤。[loader L23–27](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/dataloaders/vidor.py#L23-L27)
- 验证阶段共同轨迹长度<5、采样后不足2点过滤的是**对象对输入**，不是该对上每条GT关系的时长。[L659–685](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/dataloaders/vidor.py#L659-L685)
- 验证相位默认0，但原点是对象对共同轨迹起点；不能直接用全视频frame_id模4判定是否采到GT。
- `max_so_pair:200`是batch大小，不是pair剪枝；训练某pair关系实例>9会跳过整对。这些规则和全视频输出Top200不是一回事。[batch](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/models/maskvrd.py#L202-L209)、[training](https://github.com/lucaspk512/vrdone/blob/74981850ae719ac3cb9f407f0bb1e7a9ea04388d/dataloaders/vidor.py#L366-L391)

## 本项目已完成的本地检查

复现脚本：`handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_resolution_audit.py`，结果`duration_resolution_results.json`。只读已导入实例表，不运行模型。

| GT<1秒内的帧长度 | GT数 | Top100匹配 | 全部保存结果匹配 |
|---|---:|---:|---:|
| 1–4 | 230 | 0 | 0 |
| 5–8 | 549 | 0 | 0 |
| 9–16 | 1,230 | 3 | 5 |
| 17–32 | 1,158 | 13 | 21 |

所有5,177个**成功匹配**的预测，长度均为4n+1，最小13帧；这与公开解码规则相容。成功子集不包含失败预测，不能据此宣布全输出没有5/9帧预测、推理门槛是13帧或量化已被确证为漏检原因。

1–8帧共779条零匹配只是需要解释的范围；GT帧数不等于预测帧数，8帧GT也不必是单点监督。单点输出被删这一逻辑事实不能解释全部779条，下一步须记录真实pair起点、采样索引、mask正点数、候选类别与删除位置。

本轮新分桶为公开stride线索驱动的探索性分析，未当作独立验证集。缺少真实配置和过滤前缓存时，机制保持UNVERIFIED。
