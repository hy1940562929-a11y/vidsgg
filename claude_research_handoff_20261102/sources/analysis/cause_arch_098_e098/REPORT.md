# E098-JOINT-READING 执行报告

**执行结果：REJECT / UNDERREADING_NOT_ESTABLISHED。**

按冻结合同执行到训练前停止门槛。原生重建与新容量门槛通过，但“334残余中多数事件存在严重低读取”未成立。因此没有启动N/J训练，没有干预后效果数字。

## 1. 本次实际执行范围

| 环节 | 实际状态 |
|---|---|
| AutoDL连接与GPU | 可用；NVIDIA GeForce RTX 4090 D。初次沙箱内连接受限，授权环境下连接成功，不作为远程连接失败。 |
| 协议冻结 | [PROTOCOL.md](PROTOCOL.md)，[源SHA](input_freeze.json)，[执行代码SHA](software_freeze.json) |
| 原生输入与初始化 | 找到087的88个train、23个dev、1614个eval原始输入，以及原预训练EMA checkpoint；没有用分数NPZ替代decoder输入。 |
| N原生数值重建 | **1725/1725批通过；最大绝对误差0**，低于既定4e−6；有限值、mask二值、类别Top1/Top6一致。GPU冻结前向约66.13秒，更新0步。 |
| 材料身份补验 | 冻结manifest及1821个负载文件SHA一致；训练32视频/350pair/563事件、开发8视频/89pair/169事件及完整target索引/类别检查通过；评价成员2573及400/163/334/1101保持。 |
| 9-slot容量门槛 | PASS，见下表。两边统一解除旧F/原H历史准入作为主读出的限制。 |
| 原低读取现象门槛 | **FAIL：47/334，95%CI [9.95%,18.55%]，CI下界没有超过50%。** |
| N/J × seed98/99/100 | **训练0/6；每臂更新0步。** J算子、500步预算、撤回/恢复、训练后AUROC与保护组效果均NOT_RUN。 |

CPU身份、容量与读取准入脚本用时约3.24秒。耗时是脚本回执，不包含本轮准备和文档时间。完整分母和源代码均保留；未修改容差、成员、阈值或query数。

## 2. A路线确实移除了旧诊断容量墙，但不是模型改进

读出：每个固定pair的9个原有query，各自按当前分数H解码（sigmoid>0.5、stride4末点+1、min5），最多一个区间/槽。GT只在输出生成后用于匹配或oracle容量审计。

| 固定集合 | GT数 | 新协议C0一对一时间命中 | U9容量上界 | U9−C0 |
|---|---:|---:|---:|---:|
| 主组 | 400 | 46/400 | 396/400 | 350/400 |
| 亚秒组 | 163 | 0/163 | 163/163 | 163/163 |
| 原残余组 | 334 | 29/334 | 331/334 | 302/334 |
| 长保护组 | 1101 | 540/1101 | 815/1101 | 275/1101 |
| 全部诊断GT | 2573 | 989/2573 | 1591/2573 | 602/2573 |

因此334组具有正的可达缺口，400组最大视频等权改善也超过预设5pp，容量准入通过。

**这些不是新模型效果。** C0未更新参数；334的29次命中来自统一更换诊断输出/匹配合同，不能与E097旧F的0/334相减后称为架构收益。U9是GT oracle容量，不是模型恢复。

表内各集合独立一对一匹配、独立优化上界，不得跨行相加或认为能同时实现全部上界。这里不要求谓词和轨迹vIoU正确，不是官方mAP或Recall@100。容量门槛只需最大匹配数，最优解tie不改变计数；本轮没有发布C0逐GT新增/损失身份来冒充完整效果评价。

来源：[capacity_audit.json](results_v1/admission_v1/capacity_audit.json)、[逐pair容量账](results_v1/admission_v1/capacity_per_pair.csv)。

## 3. 触发停止的决定性数字

按合同取原预训练模型**最后一个decoder cross-attention层、全部8个heads平均、dropout前**权重。R是GT时间获得的9-query合计读取量，除以均匀时间读取应获得的量。R=1表示与自身时长相称，预设R<0.5表示低读取。没有选择层、head或GT最佳query。

| 集合 | R<0.5事件数 / GT数 | 比例（附分子/分母） | 视频整簇bootstrap 95%CI | R中位数 |
|---|---:|---:|---:|---:|
| 主组 | 60/400 | 15.00%（60/400） | [11.25%,19.00%] | 0.6762 |
| 亚秒组 | 28/163 | 17.18%（28/163） | [9.33%,27.78%] | 0.6610 |
| **原334残余** | **47/334** | **14.07%（47/334）** | **[9.95%,18.55%]** | **0.6635** |

334组覆盖23个视频，334/334目标都完整落在固定pair段内。因此这组低读取统计不是因为GT落在所选段外而被记成0。

合同准入要求：334组低读取比例的95%CI下界>0.50。实际下界0.0994757，上界0.1854642也低于0.50。**未达到“系统性严重低读取影响多数残余”的事前定义，按合同停止训练。**

CI来自seed98、10,000次视频整簇重采样，每次重算事件分子/分母；不是把334事件当独立视频，也不是分子/分母固定的二项区间。CI端点是bootstrap分位数，不能伪造单一整数分子。

原始分数：[per_gt_reading.csv](results_v1/admission_v1/per_gt_reading.csv)，汇总：[underreading_audit.json](results_v1/admission_v1/underreading_audit.json)。脚本保留全组与长组分布，但其中分别有919/2573、279/1101的GT未完全处于其历史所选pair内，不能拿它们与400组直接比较后推导长短读取机制。

## 4. 坐标与材料核验依据

原输入采样stride4；配置`backbone_arch=[2,2,3]`、`scale_factor=2`，主分支三次下采样均为居中kernel3/stride2/pad1，所以coarse key中心间隔为4×2³=32原视频帧。decoder的key卷积kernel3/stride1/pad1不再移动中心。四层decoder反复使用同一memory/mask，不逐层改变时间网格。

实际源码和配置已封存：[backbones.py](results_v1/source_snapshot/backbones.py)、[blocks.py](results_v1/source_snapshot/blocks.py)、[local_transformer.py](results_v1/source_snapshot/local_transformer.py)、[predictor.py](results_v1/source_snapshot/predictor.py)、[effective_config.json](results_v1/source_snapshot/effective_config.json)。不是仅凭脚本注释假设32帧。

读取审计核对890个所选pair的key有效前缀、padding零权重、attention行和、中心在段内和正长度时间单元；每条GT按其与单元的交叠积分，不要求短GT包含一个key中心。首尾单元裁到固定pair边界。[逐pair坐标回执](results_v1/admission_v1/key_coordinate_audit.csv)。

原生重建逐批回执：[rebuild_batches.jsonl](results_v1/preflight_v1/rebuild_batches.jsonl)；[初始化/源码/配置回执](results_v1/preflight_v1/material_receipt.json)；[完整材料身份回执](results_v1/admission_v1/material_identity.json)。

`local_transformer.py`不在094旧source列表中，不能声称它有094历史SHA。本轮在原生重建前记录实际全部models源码SHA，CPU准入时复核未变，并用1725批同输入输出重建确认接线；这一范围写入材料回执。

## 5. 判决范围

支持的结论是：**H098所规定的“严重低读取覆盖多数残余”前提没有建立，停止当前联合时间读取主原因配方，不启动J训练。**

不支持以下说法：

- “J训练过而无效”：没有训练任何N/J模型。
- “注意力读取分配不存在问题”：只检查预设层、head聚合及R<0.5的操作性定义。
- “短关系证据已经充分”：读取系数不是视觉语义信息量；R不低不代表mask一定可定位。
- “完整根因已经找到”或“长关系语义压制已证伪”：本轮均不成立。

E091仍为`NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE`，几何主配方未复活；E097仍为`REJECT / SCOPE_CAPACITY_CONFLICT`且未训练；本次没有把任何旧假设升格。M098不采纳，RQ不改，不增加query、不改R门槛、不换attention层，不排队下一项。

## 6. 复现与审查

实际执行代码：[preflight.py](preflight.py)、[admission.py](admission.py)；运行日志：[preflight.log](preflight.log)、[admission.log](admission.log)。原始回执已逐文件SHA核验导入：[import_receipt.json](import_receipt.json)。

合成检查确认min5合法网格及一对一容量（单预测不能命中两条GT，两槽可分别匹配相同时间的两个事件）。本地另对导入CSV重计47/334、对逐批回执重计1725个零误差批次；[selfcheck.json](selfcheck.json)。

独立Reviewer已静态复核N数值预检及容量/R积分/bootstrap实现为`PASS_WITH_CONCERNS`。其要求的manifest逐项SHA、完整target检查与实际坐标来源均已保存；该设计/实现审查不改变科学停止出口。

结果复核：独立Reviewer重新核验16个导入SHA、1725批重建、实际坐标源码、逐GT的47/334及视频bootstrap区间、逐pair的U9=331/C0=29，给出**PASS**。确认科学出口为`REJECT / UNDERREADING_NOT_ESTABLISHED`，无文字修订要求；本轮到此停止。
