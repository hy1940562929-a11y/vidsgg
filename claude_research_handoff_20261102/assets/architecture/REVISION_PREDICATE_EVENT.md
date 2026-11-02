# 架构图修订说明

日期：2026-10-30。状态：**DESIGN_ONLY / UNVERIFIED**。

修订以用户当前编辑过的 `architecture_reference_style_font_matched.svg` 为源；原件保存为 `architecture_reference_style_font_matched_before_predicate_event.svg`。保留源图的两张输入图片、Tracker、Local Temporal Encoder、图例、Arial/Times 字体组合、浅奶油/浅青底色及桃色/紫色/绿色模块配色。

## 三处方法设计对应

1. **逐谓词时间证据图**：每个主体—客体实例对产生 `E_so[T,C]`，`C=50`，独立 sigmoid。矩阵格子仅表示张量结构，不表示实测激活值。同谓词所有合法事件区间的并集为正例；未知标注不自动当负例。该证据头与项目 M010 的共享 `1×1` 谓词读出存在直接关联，不能单独宣称新颖。
2. **query—谓词条件化时间读取**：`w_kp = MLP([q_k; e_p; Wq(q_k) ⊙ Wp(e_p)])`；`r_kp(t) = logit(E_so(t,p)) + w_kp^T h(t)`。谓词与实例共同改变特征读取方向，而非只加时间常数。实现时应保留原始 evidence logits，无需对 sigmoid 结果数值反求 logit。推理条件来自模型类别候选，不使用 GT 谓词。
3. **事件集合学习与解码**：联合评价区间和事件集合，在固定最终预测预算下选择输出；重复事件保持独立，多谓词可重叠。`L_set` 表示集合匹配/监督目标，`L_keep` 表示有效持续关系的保护目标；具体损失、搜索规则和预算须在实现前冻结。图不承诺任何总体指标或长关系零退化。

## 相对原图的内容调整

- 用“原生关系假设＋谓词嵌入”替换“其它关系驱动语义更新”作为主要输入，避免将长关系压制短关系这一未验证解释画成既定机制。
- 用显式的谓词证据张量及条件化读取替换泛化的 Predicate/Instance Evidence 黑盒。
- 用联合事件集合解码替换独立 Legal Argmax 与未经验证的固定三轮 repair 表述。
- 输出采用事件向量卡片，未标记真实命中、时间长度或性能增益。
- 没有新增柱状图、折线图、实测热图或装饰图标。模块、连线、公式与标签均为 SVG 矢量对象；仅保留输入实物图片。

## 文件

- 用户指定路径：`architecture_reference_style_font_matched.svg`（同步修订版）。
- 独立修订版：`architecture_reference_style_predicate_event_v2.svg`。
- 预览：`architecture_reference_style_predicate_event_v2_preview.png`。
- 重建脚本：`build_predicate_event_revision.py`；始终读取保存的源图备份。
- 验收回执：`predicate_event_revision_receipt.json`。

## 验收范围

已检查 XML 可解析、ID 唯一、引用可解析、两张输入图内嵌、文字不超出画布；已渲染并查看全图与放大区域，修正重叠标签和穿字连线。此验收仅针对图形与设计表达，不是代码实现或科研实验验证。
