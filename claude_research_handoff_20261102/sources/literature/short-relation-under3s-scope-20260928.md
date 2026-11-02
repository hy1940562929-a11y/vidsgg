# 小于三秒的短关系：定义与文献覆盖

2026-09-28。Direction复核，Commander整合。范围：7篇核心工作；另补查2篇近年工作的相关全文章节，1篇出版方预览。数量表示本次核查范围，不是领域论文总数。没有新模型实验。

## 定义判断

`0<GT duration<3s`可以作为本研究的操作性分组。**未核实它是领域标准，也未证实3秒是模型机制断点。** 此阈值在本项目已有数据探索后提出，不能称为预注册或事前独立确定；可在后续实验中固定使用，同时完整保留分箱和阈值敏感性。关系时长、输入clip长度、视频长度及采样窗口不能互相替代。

## 7篇核心工作

| 证据类别 | 论文 | 已核查内容 | 支持明确GT<3s定义？ |
|---|---|---|---|
| 直接时长失败分析，1篇 | [Diagnosing Errors，BMVC 2021](https://isis-data.science.uva.nl/cgmsnoek/pub/chen-diagnosing-bmvc2021.pdf) | §3.2 Fig.5：S=(0,10]秒，M=(10,20]秒，L>20秒。VidOR两个方法的短组FNR为90%/88%，默认Top200。原图聚合口径与本项目microRecall不可直接换算。 | 否，短组上界10秒 |
| 分时长改善，1篇 | [Social Fabric，ICCV 2021](https://arxiv.org/pdf/2108.08363) | §6 Fig.5：short/medium/long的P@1改善，非RelDet Recall；未查明分组阈值。 | 未核实 |
| 相关时间建模/定位，5篇之一 | [VRD-STGC，CVPR 2020](https://pkumyd.github.io/paper/CVPR2020_VideoVRD.pdf) | §3.1多尺度滑窗最小30帧，Fig.2为谓词平均时长；不是GT短长分组。 | 否 |
| 同上 | [TSPN，2021 v1](https://arxiv.org/pdf/2107.07154v1) | §IV-E(c) Table VI比较4/8/16/32 sectors，实际时间长度随对象对共存区间变化。 | 否 |
| 同上 | [BIG，CVPR 2022](https://arxiv.org/pdf/2112.04222) | §3.3、§4.3 Table6多实例grounding、1/5/10 bins及单次/多次发生关系。 | 否 |
| 同上 | [HCM，2023](https://arxiv.org/pdf/2307.08984) | §IV-A的30帧clip/15帧重叠属于输入分段；Tables I/II分析上下文与关联。 | 否 |
| 同上 | [VrdONE，MM 2024](https://arxiv.org/html/2408.09408v2) | Introduction、Fig.3、§3.4关注短暂/持续关系，采用多尺度和关系mask定位。 | 未查到明确3秒分组召回 |

核心7篇中，已核实明确采用GT<3秒作为short定义的为0篇。不能说7篇均专门研究3秒内关系，也不能从0篇推出整个领域不存在该定义。先前核查的整体效果和局限见`short-relation-review-20260928.md`。

## 近年补查与误引排除

- [DiffVSGG，CVPR 2025](https://arxiv.org/pdf/2503.13957)：附录§2 Table1为ImageNet-VidVRD总体RelDet/RelTag结果，所查结果不提供3秒分组依据。
- [SVG2/TraSeR，2026 v3](https://arxiv.org/html/2602.23543v3)：§4.2、Appendix D的4秒窗口/1fps为表示窗口；§5使用GT panoptic trajectories和开放词汇协议，不是当前同协议短关系修复结果。
- [VSTRD，Neurocomputing 2025](https://www.sciencedirect.com/science/article/pii/S0925231225019794)：只核出版方预览，目标偏长段关系断裂；全文短组结果UNVERIFIED，不计完整核查。
- [TSPN 2026正式版](https://www.sciencedirect.com/science/article/pii/S0957417425031185)不与2021预印本重复计数。
- [VidSTG补充材料，CVPR 2020 §1](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zhang_Where_Does_It_CVPR_2020_supplemental.pdf)中的3秒是**视频长度过滤**，另外排除关系<0.5秒或不足视频长度2%。这是另一任务的构建规则，不能引用为本研究的GT<3秒定义。

检索组合包括VidOR/VidVRD与3 seconds、three seconds、2 seconds、5 seconds、short duration、short-lived、temporal granularity、2024–2026。Social Fabric阈值及部分付费全文仍有缺口。

## 对研究范围的影响

扩展到3秒会同时纳入更多已知时间定位/变长关系问题，因此相关工作覆盖更广，不能凭改阈值建立新颖性。当前合理问题是：在控制谓词、输入轨迹和预算后，哪些具体环节导致该时长范围内的关系漏检，能否提出可复核且可推广的改进。先完成C组画像和成功对照，尚不进入方法设计。

原先公开VrdONE代码的单点mask/最短预测区间线索仅针对特定极短输出情况，不能因为GT分组扩大到3秒就外推为整个3秒组的统一原因。若亚秒与1–3秒主要原因不同，应保留分别诊断和结果。
