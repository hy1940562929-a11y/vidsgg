# M011相关架构的已核查先例

检索日期2026-09-29。来源为论文、CVF官方页面和作者机构。以下是文献机制，不是本项目复现结果，也不是穷尽性创新性审查。

| 工作与来源 | 已做的机制 | 对本方案的约束 |
|---|---|---|
| [TriDet，CVPR2023](https://arxiv.org/abs/2303.07347) | 相对边界概率分布、时序粒度建模 | 起止概率分布不能称首次；应证明实例条件的额外收益 |
| [TCANet，CVPR2021](https://arxiv.org/abs/2103.13141) | 局部与全局上下文，边界/内部信息，渐进区间细化 | 邻域上下文与边界修正已有直接先例 |
| [BMN，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.html) | 起止边界匹配与proposal置信图 | 区间联合建模不构成独占创新 |
| [SoLa，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Soft-Landing_Strategy_for_Alleviating_the_Task_Discrepancy_Problem_in_Temporal_CVPR_2023_paper.html) | 冻结分类编码器上的轻量适配，帧间相似性训练 | 冻结特征与定位任务不匹配是已知研究问题；本项目尚未测定其贡献 |
| [ASRF，WACV2021](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html) | 分类与边界概率分支，边界辅助分段 | 显式结束/边界建模有先例；不能说论文等同于我们的半马尔可夫关系模型 |
| [Segmental Spatiotemporal CNN，ECCV2016](https://arxiv.org/abs/1602.02995) | 时空CNN与半马尔可夫动作分段 | 事件状态/分段结构并非全新；引用修订版，其记录了数学修正 |
| [BIG](https://arxiv.org/abs/2112.04222) | 分类后定位，时间二分图，关系三元组条件的多实例定位 | 关系条件定位是VidVRD内部直接先例，不能仅与通用TAL比较 |
| [VrdONE](https://arxiv.org/html/2408.09408v2) | 联合关系分类与时间mask的单阶段框架 | 当前工作改动应准确定位到冻结实例区间读出，非首个联合关系定位 |
| [TPT，IEEE TMM26(2024)](https://research.polyu.edu.hk/en/publications/end-to-end-video-scene-graph-generation-with-temporal-propagation/) | 对象query传播与关系时序一致性，统一检测/跟踪/关系 | 跨对象身份传播有先例；VidHOI/ActionGenome结果不能当VidOR对照 |
| [PALS，2026预印本](https://arxiv.org/abs/2603.21559) | pair affinity排序与上下文、关系感知匹配 | 输入pair质量与噪声上下文已被关注；弱监督ActionGenome不等于本项目设定，未确认正式发表 |

本轮不引入上述工作的benchmark数字进行横向比较，因为任务、数据和输入协议不同。A/B/C的机制、预测及最小验证见 `methods/vidor-architecture-011-candidates.md`。
