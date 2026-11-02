# 085 实际结果独立复核

2026-10-20。审查者 `/root/context085_review` 只读复核，Commander 保存回传正文。审查对象为 `handoff/incoming/VIDOR-LONGSHORT-CONTEXT-085/remote_20261020/{code_v3,plan_v3,infer_v3}`、日志及本地报告。

**Decision：NEED_MORE_EVIDENCE（机制）；数值及文件完整性审计通过。** 未达到预定支持门槛，也未证明替换有效削弱长关系。不能宣称机制成立或被证伪。

## 已独立核验

- 28/28 NPZ文件SHA、28/28输出数组摘要哈希一致；7/7 baseline/noop逐元素一致。
- 原数组重算28臂指标、14组baseline差值、7组配对差值、视频等权和固定seed bootstrap一致，概率跨平台误差约1e-8。
- baseline、editL、editN锁定query及任意query时间命中上界均0/7；两干预臂新增/损失均0/0。
- 执行脚本、两个073 helper、model、predictor、dataloader均找到计划SHA匹配的本地副本。effective config重新序列化导致字节不同，但内容与原SHA匹配配置完全相同。
- plan SHA为 `9e6dc4230d64cfb8667adb0f551f6bb8799cd7dc45dd2ec207c00987e25887a3`，与summary一致；执行脚本SHA为 `8d00b39ac0d2cbcbcdf32041c8ebe761fe3e43c53db7daa73fdd36dc320ac6bf`。
- 导入可行候选的hash排序、视频配额、选中集合重算一致。455筛选排除+11可行=466初筛记录；11可行−4配额排除=7执行。466不应写成去重后的独立事件。

## 主结果

| 指标 | 值 |
|---|---:|
| 视频等权editL−editN | +0.003201677 |
| 视频bootstrap95%区间 | [−0.006746331, +0.022314668] |
| 视频等权editL−baseline | +0.005869024 |
| 事件等权editL−editN | −0.002371038 |
| 视频支持 | 3，预定至少10 |
| supported | false |

## 问题

**R-001 MAJOR：机制UNVERIFIED。** editL后原L的长query mask概率仅2/7下降、5/7上升，事件均值变化+0.004025587；长类别概率仅3/7下降。唯一正向视频0049_3151715056的长mask也上升，不能解释为长关系减弱释放短关系。不得删除操纵失败样本或改用分类概率下降替代mask检查。

**R-002 MAJOR：总体支持或证伪UNVERIFIED。** 3视频、5有向实体pair，6个watch、1个speak_to。每视频4/2/1例，配对均值约−0.006746、−0.005963、+0.022315。正向总均值由唯一正向视频贡献，事件等权反号。主效应低于0.01、区间跨零、视频不足。三视频bootstrap只宜描述，机制为INCONCLUSIVE，不是机制不存在，也不是执行失败。

**R-003 MINOR：输入、权重不变量仅由运行断言支持。** 已检查ROI-only、保护区、geometry、原生preprocessing、decoder配置和参数哈希断言及28条审计记录。未重新加载原始特征/checkpoint，也未独立核验全部27视频标注、特征源。不得声称Reviewer重建验证全部输入。不影响输出复算。

**R-004 MINOR：外推范围。** 329条因缺少N/donor窗口排除，严格27视频不能代表全val。报告中的085数字、未决判决及不扩训练边界与审计一致。082/080/079/073历史论述不在本次复核范围；新颖性NOVELTY_UNCERTAIN。

R-001/R-002阻止机制升级，不阻止登记执行完成。无需为通过此次数值审计追加实验。进一步机制研究需可靠操纵及足够视频支持，保留原输入、noop、匹配对照、冻结query、目标保护、双向新增/损失；未满足前不能进入基于已确认机制的方法训练。
