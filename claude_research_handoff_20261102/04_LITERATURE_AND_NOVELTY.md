# 04 文献对齐与创新边界

本轮网络核查日期2026-11-02。F=本项目历史笔记核过关键全文/源码；A=本轮官方摘要核查；本表的设计解读不是本项目复现结果。091历史26项完整表随包，保留原F/A/P分级，不伪称本轮读完所有全文。对以下近邻的差异是待验证研究切口，不是“他人不能解决”的证明。

|近邻|主来源/阅读层级|已有设计|本项目要回答的额外问题|
|---|---|---|---|
|VrdONE, MM2024|[论文全文](https://arxiv.org/html/2408.09408v2)，F；本轮核附录D|实例query+类别/mask、BSA、多尺度；已试边界回归和语义分割|为何某些短事件仍没有匹配区间，改进能否保总体与长关系|
|BIG, CVPR2022|[论文](https://arxiv.org/abs/2112.04222)，A/F历史|分类后多实例grounding、二部图|不能宣称首次支持同pair多事件；需公平输入与输出预算|
|SegPVSG, ICML2026|[PMLR](https://proceedings.mlr.press/v306/li26bx.html)，A/F历史|TFN先定位显著区间再识别；RGVA生成关系视频片段增强|在谓词感知、短/长分层、共存条件下与TFN同预算比较，而非仅改名称|
|VSGN, ICCV2021|[原文](https://arxiv.org/abs/2011.14598)，A/F历史|短片段放大拼接、跨尺度图金字塔|短事件本身不是新问题；跨尺度不等于共存条件机制|
|ActionFormer, ECCV2022|[原文](https://arxiv.org/abs/2202.07925)，A|多尺度/局部attention、逐时刻分类和边界|作为任务适配强基线；不能仅改一维中心宽度称创新|
|TriDet, CVPR2023|[原文](https://arxiv.org/abs/2303.07347)，A/F历史|相对边界概率分布、SGP跨粒度|分布边界本身有先例|
|BAM-DETR, ECCV2024|[原文](https://arxiv.org/abs/2312.00083)，A|区间内anchor、左右边界双路径、质量排序|不能把边界专用query或质量排序单列首次|
|DenoiseLoc, Boundary-Denoising, 2023预印本|[原文](https://arxiv.org/abs/2304.02934)，A/F历史|扰动GT span再学习边界去噪|普通区间修复/去噪已有；需要谓词/共存特异比较|
|TadTR, TIP2022|[原文](https://arxiv.org/abs/2106.10271)，F历史|可变形时间query、迭代修正、局部质量评分|普通IoU/质量头有直接先例|
|TE-TAD, CVPR2024|[原文](https://arxiv.org/abs/2404.02405)，A|真实时间坐标、随视频长度自适应query|时间尺度表达/容量改动不能与证据机制混淆|
|VSLNet, ACL2020|[官方](https://aclanthology.org/2020.acl-main.585/)，A/F历史|query高亮、条件起止span、联合合法端点|区间高亮+端点监督已有；语言query与关系query不同但非自动原创|
|GlobalPointer, 2022预印本|[原文](https://arxiv.org/abs/2208.03054)，F历史|类别条件起止对联合评分，多标签span|[T,T,C]区间矩阵本身不是新方法|
|NeuralNetwork-Viterbi, CVPR2018|[原文](https://arxiv.org/abs/1805.06875)，历史核查|类别条件时长先验联合序列解码|按谓词时长先验与联合解码不是首次|
|Self-DETR, ICCV2023|[原文](https://arxiv.org/abs/2308.10570)，A|从cross-attention恢复self-attention多样性|attention collapse不能未经测量就套到VrdONE|
|TBT-Former, 2025预印本|[摘要](https://arxiv.org/abs/2512.01298)，A，仅确认预印本|ActionFormer改进、跨尺度FPN、边界分布|新近先例；不能称已确认CVPR/ICCV接收或据摘要断言缺陷|

## 必须纠正的SegPVSG误读

原文是时序稀疏与长尾，确实覆盖显著片段定位。不能称它只作“宏观有无”、固定1–2秒切片、结构上不能亚秒定位；没有证据。RGVA是生成视频片段增强，不能写成简单GAN合成特征。历史源码核查发现Gaussian作用位置有论文/实现差异，比较时记录实际版本；输出合并并非一定把所有不连续片段hull。作者代码 https://github.com/ticatt/SegPVSG 。

## 核查的新信息：VrdONE已经做过普通边界对照

附录D表10：boundary regression / semantic mask / instance mask三种变体；作者报告检测mAP分别11.28 / 11.57 / 11.86（×100单位），R@100分别13.85 / 13.78 / 14.21。这些是文献原文数值，非本项目重新测评，也不等于我们新增起止分布+时长的完全同一实现。其意义是“把mask改成边界”不是未探索空白。

## 数据集与迁移

VidOR仍是直接基线主场。独立迁移优先ImageNet-VidVRD（800train/200test、35实体/132谓词）：[官方](https://xdshang.github.io/docs/imagenet-vidvrd.html)。VrdONE报告其关系时长按15帧倍数量化，不能用它直接认证VidOR亚秒效果。

PVSG/OpenPVSG是panoptic任务：[代码](https://github.com/LilyDaytoy/OpenPVSG)。历史原文核查400视频、5FPS，289源VidOR/55 EpicKitchen/56 Ego4D；迁移前核对视频ID重叠，不能当完全独立第二数据集。跨任务适配不可直接拼mAP排名。

## 未完成的新颖性判断

“谓词条件证据验证＋区间外扩/收缩解释”值得核查，但对比学习、span能量、完整性评分、反事实上下文不变性都各有先例。Claude需查明最邻近方法的精确公式与训练协议，写差异表；不能承诺research gap大或CVPR录用。宁可缩小且证明一个实质差异，也不组合四个成熟模块各算一项首次。
