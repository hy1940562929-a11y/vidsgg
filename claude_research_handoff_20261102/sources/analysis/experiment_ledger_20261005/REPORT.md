# 第一阶段：VidOR / VrdONE / BIG 实验总账

核验日期：2026-10-05。只使用项目本地实际文件；未把聊天概括或文献数字当作实验事实。未运行新模型、训练、远程任务，未修改历史材料或正式state。本报告不选择主方向，不设计新模型。

登记42条实验、派生分析及未完成事项。其中16个实际方法干预条件：8个无训练推理条件、6个M010训练臂、2个M011训练臂；另有GT输入双臂oracle诊断。分析条目共享原始数据，不能算作独立复现次数。

完整逐项字段在[ledger.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/experiment_ledger_20261005/ledger.json)；包括假设、改动、固定项、范围/划分、baseline、指标、gain/loss/C恢复、分组/全量结果、可支持/不可支持结论、证据和实现路径。

## 1. 核验范围与证据等级

| 类型 | 本总账中的含义 |
|---|---|
| Observation | 已保存数据/结果直接显示的数值或事实 |
| Correlation | 原始或分层控制后的关联；共同支持和分母必须一并保留 |
| Diagnosis | GT辅助的候选存在性、oracle、匹配容量或错误路径检查 |
| Intervention | 实际更改解码/评分/模型并评估，必须同时给新增和损失 |
| Causal evidence | 本轮可确认固定条件下某操作如何改变具体候选；尚不能认证历史记忆、上下文压制或重复事件竞争等机制 |
| Hypothesis / UNVERIFIED | 未完成实验，或本地没有足够返回材料 |

已独立重算逐GT分组、排名命中、收益损失、目标分组和已导入逐视频AP；已打开对应实现、配置、训练日志及划分，并核对新增头权重文件。完整835输入框轨迹/预测轨迹以及GT27全部raw/input未全部导入，本轮没有从头重跑全部几何匹配或oracle搜索。远程几何诊断与本地聚合复算分开登记。

## 2. 固定定义及数据范围

- A=BIG与VrdONE原Top100都命中；B=仅VrdONE命中；C=都漏检；D=仅BIG命中。新方法的C恢复始终使用这个固定C。
- F=008 v2中VrdONE实际合格输入实体对存在几何可匹配的某个连续区间。这是oracle条件，不是模型实际输出，也不是BIG输入条件。
- 正式检测：精确三元组、主体与客体管道vIoU均达到0.5、置信度排序贪心一对一。R@K分母为该范围全部GT。
- mAP以下一律用0–1小数。完整835视频清单有832个非零GT视频，mAP取832个视频AP的平均；recall分母30142。不能给mAP编造TP/GT分子。
- 各特征组可重叠；分组恢复数不能横向相加。严格<3秒按该视频fps计算，90帧通常并不小于3秒。

| 数据范围 | 用途 | 重叠/选择情况 |
|---|---|---|
| validation835 / 30142 GT | baseline、总体诊断、R4全量 | 832视频有关系；缺预测键视频的10条GT仍保留 |
| 固定32 / 1209 GT / C1047 | 008小试、009筛选、M010、M011 | 同一批验证视频反复使用；非独立测试 |
| 扩展64 / 2601 GT | 冻结R4扩展 | 与32不重叠，但全验证集此前已做诊断；两者包含于835 |
| training32：原标注1237 GT，缓存350 pair/563 GT | M010、M011训练 | 官方training；不与dev或validation交叉 |
| adapter-dev8：原标注827 GT，缓存89 pair/169 GT | 新增头开发诊断 | 对新增头留出；原预训练基座可能见过这些training视频 |
| strict27 / 2573 GT / C2270 | GT输入掩码诊断 | 来自切换筛选及已有GT特征，非随机，也非独立测试 |

训练正例仅覆盖26/50个谓词。M010 seed10，M011 seed11；均冻结基座、固定小缓存、最终步权重。M010 2000步从初始化重新训练，不是200步续训。M010动态Hungarian；M011固定原匹配。

## 3. baseline配对与主要观察/诊断

| 读取预算 | A | B | C | D | VrdONE / BIG命中（分母30142） |
|---|---:|---:|---:|---:|---|
| all_saved | 3055 | 2122 | 23866 | 1099 | 5177 / 4154 |
| top50 | 1869 | 1467 | 26342 | 464 | 3336 / 2333 |
| top100 | 2487 | 1809 | 25373 | 473 | 4296 / 2960 |
| top200 | 2919 | 2258 | 24371 | 594 | 5177 / 3513 |

BIG来自发布的epoch70 grounding hit-info，VrdONE来自重建输入的epoch3预测；不是共享输入和训练条件下的受控模型比较。all_saved时保存预算也不相等。

| 条目 | 直接核实的结果 | 能支持 / 不能支持 |
|---|---|---|
| 005/006时长画像 | <1秒C3151/3167；1–3秒7475/8454；3–10秒10296/12193；≥10秒4451/6328 | 短组漏检相关；不能据此定位时间模块缺陷 |
| 005保存尾部 | VrdONE漏检25846；其中保存排名101–200有881，无保存命中24965 | 尾部存在可读命中；不是增加模型能力 |
| 006成功匹配长度 | 5177个已成功保存匹配的预测，最短13帧、长度满足4n+1 | 仅成功匹配子集；不是全部预测/被过滤候选的最小长度 |
| 007/r2控制 | 同视频同有向pair同谓词的短长共同支持9470 GT，C标准化差仍约+7.133个百分点 | 关联；时长仍非独立因果。40组盲审未填 |
| 008保存候选oracle | 时间裁剪3707；仅谓词770；两者1967；独立尾部兼容882；Top100兼容但分配失败8 | GT辅助诊断桶，不是实际恢复数量 |
| 008输入v1/v2 | v1 GT交集裁剪可行13143；v2任意连续区间可行13294，差151 | 定义放宽产生变化，不是新模型恢复151 |
| 008输入v2全量 | F13294/30142；C内F8545/25373，不可行16828/25373 | 固定当前输入的限制；不能分开归责检测/分类/跟踪，也不能套到BIG |
| 008原始query32 | 1070漏检=722输入不可行+182无匹配raw hull+28同query谓词不在Top6+138排名/分配 | 原始输出的诊断路径；不是互斥因果贡献 |
| 008固定C32 | C1047=721输入不可行+180无匹配mask+26谓词/query联合不足+120排名/分配 | C中的不同可观测路径；不能都叫时间失败 |
| 009大小/速度 | 小实体C4917/5242；高速C9342/10485 | GT几何代理与共同失败关联；不直接测跟踪错误 |
| 009标准化 | 高速F内差-0.788pp（-91.8551/11657×100）；小实体F内+3.595pp（381.4350/10610×100） | 分层结局C，不等于VrdONE单模型miss；加权分子不是样本数 |
| 009重复标注 | 6097个多记录键、18063条GT；合并重叠/精确相邻后无变化 | 多段记录确实分离；不等于人工核验的重复动作 |
| 013后续段 | 原始C10727/11966；控制负载后+4.303pp（998.6473/23207×100），加重复次数后+1.612pp（185.8820/11528×100） | 支持该共同支持内的关联；未证实缺峰/记忆 |
| 013同键严格比较 | 1189键/322视频；首段漏852/1189、后段1304/1675；键等权差+2.058pp（24.4738/1189×100） | 同键、同一时长箱、双侧F；不等于精确连续变量匹配 |
| 013客体接续 | C1309/1411；负载控制+1.803pp（172.7930/9584×100），同视频-2.030pp（-29.4702/1452×100） | 接续代理方向不稳定；不能当真实切换或旧边残留 |
| 013其余特征 | 多谓词、同类不同客体、反向关系、短间隔、相对标注跨度、负载均已登记，保留负差/反向结果 | 全部数值和支持分母见ledger中013-STRUCTURE；未进行干预 |
| 实体/关系时长 | 短关系11621；双实体累计均≥3秒10230、当前连续均≥3秒5870、共同连续≥3秒5311 | 三种定义不同；GT支持时长不是人工可见时长 |

上述统计全部来自validation，不含新训练或自动恢复；因此“新增、损失、C恢复”不适用。没有填写的视觉标签不能推测为实体交互/遮挡/空间变换。

## 4. 实际推理与训练干预总表

所有gain/loss/C恢复均为Top100。除64和835明确标注的两行外，评价均为同一32视频/1209GT。mAP为小数；R@50/100/200以命中数和行内GT分母表达。

| 条件 | GT分母 | mAP基线→处理 | 命中@50 / @100 / @200 | 新增 | 损失 | C恢复 | 判断 |
|---|---:|---|---|---:|---:|---:|---|
| 008-SPLIT | 1209 | 0.068242328→0.068221372 | 103 / 140 / 188 | 2 | 1 | 2 | 不确定：命中+1但mAP下降 |
| 008-RIGHT | 1209 | 0.068242328→0.068140673 | 103 / 139 / 186 | 2 | 2 | 1 | 负/无净增：新增2损失2且mAP下降 |
| 009-R1-32 | 1209 | 0.068242328→0.068247507 | 103 / 144 / 183 | 6 | 1 | 4 | 不确定：局部正向点估计；未扩大 |
| 009-R2-32 | 1209 | 0.068242328→0.071665924 | 104 / 135 / 175 | 6 | 10 | 4 | 混合：mAP升但Top100净-4 |
| 009-R3-32 | 1209 | 0.068242328→0.068447097 | 103 / 139 / 187 | 0 | 0 | 0 | Top100无恢复；mAP及R200略升 |
| 009-R4-32 | 1209 | 0.068242328→0.068814871 | 104 / 142 / 188 | 4 | 1 | 3 | 不确定：局部正向点估计；候选由此筛选 |
| 009-R4-64 | 2601 | 0.118774654→0.119765818 | 301 / 393 / 492 | 11 | 6 | 10 | 不确定：新增11损失6的正点估计 |
| 009-R4-835 | 30142 | 0.117370619→0.116791628 | 3313 / 4266 / 5122 | 78 | 108 | 64 | 负结果：mAP下降；新增78、损失108、净-30 |
| 010-200-class_only | 1209 | 0.068242328→0.068237896 | 103 / 139 / 186 | 0 | 0 | 0 | 无召回变化 |
| 010-200-mask_only | 1209 | 0.068242328→0.068242328 | 103 / 139 / 186 | 0 | 0 | 0 | 无召回变化 |
| 010-200-full | 1209 | 0.068242328→0.068237896 | 103 / 139 / 186 | 0 | 0 | 0 | 无召回变化 |
| 010-2000-class_only | 1209 | 0.068242328→0.069102302 | 104 / 140 / 188 | 3 | 2 | 2 | 不确定：局部+1；仅排序变化，时间分支未提供独立收益 |
| 010-2000-mask_only | 1209 | 0.068242328→0.068242328 | 103 / 139 / 186 | 0 | 0 | 0 | 无召回变化 |
| 010-2000-full | 1209 | 0.068242328→0.069088833 | 104 / 140 / 188 | 3 | 2 | 2 | 不确定：局部+1；仅排序变化，时间分支未提供独立收益 |
| 011-local | 1209 | 0.068242328→0.059238376 | 88 / 121 / 154 | 11 | 29 | 9 | 负结果：新增11损失29，且弱于pooled |
| 011-pooled | 1209 | 0.068242328→0.069160056 | 104 / 140 / 184 | 4 | 3 | 4 | 不确定：单seed局部+1，R200下降2 |

方法改动：008-SPLIT按正mask连通段拆分；008-RIGHT把末点+1改为+4；R1算术均值改几何均值；R2乘跨度内mask平均概率（含内部负点）；R3同pair/predicate时间NMS；R4阈值0.5→0.75。所有条件继续保留min5和最终输出上限。拆分同时改变候选数与同分排序；右补齐仍不能让单正点通过min5。

M010：逐时间谓词证据残差，class_only仅分类门、mask_only仅mask门、full两门；200与2000为独立从相同初始化训练。M011：固定分类/Top6的联合起止点头，local使用逐位置视觉，pooled用时间均值替换该视觉项，但仍保留mask、位置、前景/背景信息。

## 5. 目标组收益与全体收益对照

单元格为新增/损失；组的GT分母在括号中。008的大小/运动/重复分组是本轮以原per-GT、005和v2特征重新聚合，未进行新模型推理。

| 条件 | <3秒 | <1秒 | 小实体<1% | 高速≥0.1/s | 重复≥3 |
|---|---|---|---|---|---|
| 008-SPLIT | 1/0（454） | 0/0（132） | 0/1（168） | 2/0（473） | 0/0（452） |
| 008-RIGHT | 0/1（454） | 0/0（132） | 0/0（168） | 1/1（473） | 0/1（452） |
| 009-R1-32 | 2/0（454） | 0/0（132） | 0/0（168） | 2/0（473） | 1/0（452） |
| 009-R2-32 | 3/0（454） | 0/0（132） | 0/0（168） | 3/1（473） | 1/3（452） |
| 009-R3-32 | 0/0（454） | 0/0（132） | 0/0（168） | 0/0（473） | 0/0（452） |
| 009-R4-32 | 0/0（454） | 0/0（132） | 0/0（168） | 3/0（473） | 1/0（452） |
| 009-R4-64 | 3/2（1006） | 0/0（260） | 0/0（354） | 3/0（807） | 4/1（714） |
| 009-R4-835 | 24/8（11621） | 1/0（3167） | 4/9（5242） | 19/23（10485） | 33/33（11207） |
| 010-200-class_only | 0/0（454） | 0/0（132） | 0/0（168） | 0/0（473） | 0/0（452） |
| 010-200-mask_only | 0/0（454） | 0/0（132） | 0/0（168） | 0/0（473） | 0/0（452） |
| 010-200-full | 0/0（454） | 0/0（132） | 0/0（168） | 0/0（473） | 0/0（452） |
| 010-2000-class_only | 1/1（454） | 0/0（132） | 0/1（168） | 2/1（473） | 1/1（452） |
| 010-2000-mask_only | 0/0（454） | 0/0（132） | 0/0（168） | 0/0（473） | 0/0（452） |
| 010-2000-full | 1/1（454） | 0/0（132） | 0/1（168） | 2/1（473） | 1/1（452） |
| 011-local | 1/17（454） | 0/0（132） | 4/3（168） | 6/23（473） | 2/9（452） |
| 011-pooled | 1/0（454） | 1/0（132） | 1/0（168） | 3/0（473） | 0/2（452） |

**全量R4**：不足3秒+24/-8，净+16（16/11621×100个百分点）；≥3秒+54/-100，净-46（-46/18521×100个百分点）。短关系也有8条原正确损失。全体+78/-108，净-30。

**M010**：class_only与full恢复集合完全相同；已导入10条前后候选见证的几何相同，属于已生成候选的排序变化。full没有超过class_only，不能将净+1写成时间模块收益。

**M011**：local恢复9条原C，但全体净-18；其中7条恢复见证纯时间IoU升、2条降。dev端点NLL以119条为分母，6.350968→6.167766；MAP tIoU以169条为分母，0.701495→0.598849。候选有效→无效45581指候选，不是45581条GT。pooled净+1，但R200从186降184。

**数据规模边界**：R1/R2/R3、008解码、M010、M011均没有新方法full835结果。R4是唯一执行到完整835的这些推理改动；结论为总体负结果。

## 6. GT输入诊断单独列账

| 条件/分层 | 原输入raw覆盖 | GT输入raw覆盖 | 新增/损失 | 二值mask达标 | GT hull tIoU达标 |
|---|---|---|---|---|---|
| overall（N=2573） | 683/2,573 | 1,678/2,573 | 1059/64 | 1,717/2,573 | 1,731/2,573 |
| C（N=2270） | 385/2,270 | 1,385/2,270 | 1054/54 | 1,424/2,270 | 1,433/2,270 |
| short_lt3（N=966） | 144/966 | 452/966 | 328/20 | 458/966 | 473/966 |
| duration_<1s（N=333） | 10/333 | 66/333 | 60/4 | 55/333 | 75/333 |
| short_joint_ge3（N=626） | 68/626 | 161/626 | 113/20 | 178/626 | 174/626 |
| short_joint_ge10（N=466） | 29/466 | 49/466 | 38/18 | 66/466 | 61/466 |

raw覆盖：GT辅助在原始query中选可匹配候选，忽略谓词分类与TopK。GT侧使用真身份pair；native侧使用类别相容及几何匹配。输入纠正同时改变框、类别、身份、时间支持、视觉/运动特征，并关闭GT轨迹去重（多保留114个段输入）。这些条件变化不能合并解释成mask head单独提升。

GT双臂没有训练，两臂fresh forward、原EMA、stride4/offset0、9query、阈值0.5。二值mask使用正采样点保持至下点；hull使用首末正点包围区间。这是不同指标。

同网格合法区间oracle覆盖2563/2573，而实际GT hull为1731/2573；网格可行但实际失败832条。它排除不了稀疏采样的信息损失，也未证明长上下文压制。

合并只谓词不同但身份/时间相同的关系后为2512个事件：一对一最大匹配二值1644/2512、hull1658/2512。仍是GT辅助容量，不是自动检测，也没有双臂一对一净增。

## 7. 数据冲突、口径冲突与材料缺口

| 编号 | 核验发现 | 总账处理 |
|---|---|---|
| C01 历史算术修正 | 6个字符串nan曾被分到高速：10491→10485，NA405→411 | 保留旧integrated_results_before_nan_correction；采用finite口径，不覆盖源CSV |
| C02 oracle定义变化 | input v1可行13143 vs v2 13294 | v1=GT交集裁剪，v2=任意连续区间；151不计模型恢复 |
| C03 匹配口径 | 005尾部实际命中881 vs 008尾部独立兼容882 | 882=880实际tail+2未分配；另8条Top100分配桶中1条落入tail，所以官方tail=881 |
| C04 分母差异 | 全体输入可行VrdONE漏8998 vs CF8545；小试不可行722 vs C不可行721 | 前一对差453条D；后一对差1条非C。均非算术错误 |
| C05 训练规模 | train原标注1237 vs缓存563；dev原标注827 vs缓存169 | 用途和过滤不同，分别记录；M011 NLL119 vs tIoU169也不可混用 |
| C06 指标不同 | raw vIoU1678、binary1717、hull tIoU1731；一对一事件分母2512 | 各自命名，不合称正确mask数或正式Recall |
| C07 实体时长 | 双实体累计长10230 vs连续长5870 vs共同连续长5311（均限定11621条短关系） | 三种定义均保留；不能择高值写故事 |
| C08 评估复用 | 008/009/M010/M011共享32；64不重叠但已暴露；835含前述组 | 不声称独立泛化；选择后的bootstrap未校正选择偏差 |
| C09 版本副本 | M011 final_results与evaluation_results一致；diagnostic v1/v2分析M010而非新增训练 | 只计2个M011训练臂；八个真实权重仍分别计数 |
| C10 浮点尾差 | baseline mAP 0.117370618806(float64) vs0.117370620612(官方float32 PR) | 差约1.8e-9，无命中差异；本表保留各文件原数值 |
| C11 机制口径 | pooled仍有mask/位置；M011部分端点似然不是完整窗外删失；KL不是全候选保留约束 | 按实际实现描述，不按模块名字扩大解释 |
| C12 实际状态 | 005旧outgoing未启动但incoming确实存在；006/007无对应远端返回 | 不凭task状态或文件名判断完成；本地统计与远端未完成分开 |
| C13 缺失返回 | strict44仅找到脚本，未找到切换候选CSV/指标结果；BIG绑定探针原始包未找到 | 不用聊天数字补账，不断言未做过；状态为本地材料不足 |
| C14 重现材料范围 | 8个新增头checkpoint本地存在；完整基座/全量原始框轨迹/GT27 raw未全部导入 | 本轮为结果/实现审计，不冒充从头复现实验 |
| C15 GT历史特征来源 | 提取代码和27视频完成日志存在，但原提取命令及权重哈希不足 | 影响完整重现；不能把后续输入对齐检查当历史来源完全认证 |
| C16 结果方向 | R4局部32/64正、全835负；M010 full不胜class；M011 local弱于pooled | 全部保留；个别C恢复不等于总体有效 |

## 8. 失败尝试与未验证事项

- 008首次pilot因捕获函数命名空间NameError退出；retry成功。前者是技术失败，无性能结论。
- 009 speed8_v5曾因相对输出路径FileNotFoundError退出；后续完整评估存在。性能优化版本不计新模型。
- GT旧缓存核验第26视频超过数值容差，失败保留；主结果采用两臂全部fresh forward。全部3913587个mask logit的numpy/CUDA阈值判断不一致0。
- M012：设计/实现/打包静态检查已做；v1缺helper、v2补齐；无训练、权重、检测指标，后续用户HOLD。
- PVSG REBIND-003：只有READY_FOR_REMOTE任务包，无返回结果。Direction/Reviewer文献任务不是科研实验结果。
- 006/007/007-r2远端包与人工视觉审查：没有返回完成证据。已有本地统计照常登记。
- strict44参与者切换与BIG绑定探针：当前项目材料不足，不加入已验证性能统计。

## 9. 逐项假设、改动、固定项与文件索引

所有记录完整字段见ledger.json；这里给可读索引。每行的结果、分组、gain/loss见前表及对应原始文件。

| ID | 性质 | 要验证什么 / 改了什么 | 能支持 / 不能支持 | 证据与实现 |
|---|---|---|---|---|
| 005-COMPARE | Observation | VrdONE新增与损失哪些GT；配对是否一致；不改模型；比较已有预测命中 | 支持：Top100 A2487/B1809/C25373/D473；VrdONE4296、BIG2960命中；不能：B不能等同VrdONE架构独有解决的语义类型；未恢复BIG原预测/导出器 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/summary.json)；[run.log](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/run.log)；实现：[analyze_import.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/analyze_import.py) |
| 005-PROFILE | Observation / Correlation | 短关系及某些谓词是否集中漏检；只重分组已有GT与命中 | 支持：<1s共同漏检3151/3167；1-3s7475/8454；3-10s10296/12193；>=10s4451/6328；不能：时长因果效应；检测/跟踪/边界的独立机制 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/results.json)；[duration_summary.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_summary.csv)；实现：[analyze_import.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/analyze_import.py) |
| 006-DURATION-RESOLUTION | Observation / Diagnosis | 亚秒关系是否与采样/预测时长有关；仅分析已匹配保存区间 | 支持：可以统计已命中候选的区间长度/分辨率；不能：幸存者样本不能证明漏检由stride或最短长度造成 | [duration_resolution_results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_resolution_results.json)；实现：[duration_resolution_audit.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_resolution_audit.py) |
| 007-PROFILE | Correlation | 共同失败中哪些结构特征更多；只统计/抽样，不改模型 | 支持：记录时长、谓词、关系段数、视频负载的分布和分母；不能：表中inspection候选不等于已看视频；交互/运动语义未核验 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/shared_failure_profile/results.json)；[inspection_sample.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/shared_failure_profile/inspection_sample.csv)；实现：[profile_shared_failures.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/profile_shared_failures.py) |
| 007-R2 | Correlation | 短长组差距能否由谓词/类别/重复性构成解释；在共同支持分层内标准化；生成盲审样本 | 支持：可比较指定分层内的短长组失败率；不能：共同支持群体不同；无运动/视觉人工审核；不是时长因果干预 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_controls/results.json)；[standardized_comparisons.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/duration_controls/standardized_comparisons.csv)；实现：[compare_duration_within_groups.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-PAIR-AUDIT-005/analysis/compare_duration_within_groups.py) |
| 008-SAVED-ORACLE | Diagnosis | 失败在已有预测中是否可用改谓词/裁时间解释；GT辅助匹配/裁时间，不产生可部署预测 | 支持：识别time_crop3707、predicate_only770、两者1967等诊断桶；不能：oracle可行不等于自动恢复；互斥桶不是永久单因故障 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/cpu_audit/summary.json)；[per_gt_diagnosis.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/cpu_audit/per_gt_diagnosis.csv)；实现：[audit_saved.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/audit_saved.py) |
| 008-INPUT-V1 | Diagnosis | 当前轨迹裁到GT交集是否达标；仅测试GT交集裁剪；后由v2任意连续区间搜索替代 | 支持：可裁剪达标13143；v2新增151条可行，属于oracle定义放宽；不能：151不是模型恢复；v1与v2运动lag支持规则也不同 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit/summary.json)；[per_gt_input_features.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit/per_gt_input_features.csv)；实现：[audit_input_pairs.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/audit_input_pairs.py) |
| 008-INPUT-V2 | Diagnosis | 固定轨迹上是否存在能匹配GT的任意连续区间；穷举oracle时间；v2复现loader合格pair筛选 | 支持：全体F13294；C中F8545、不可行16828；不能：不是同时可达召回；不能分解为检测、分类、跟踪各自损失；不是BIG输入诊断 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit_v2/summary.json)；[per_gt_input_features.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit_v2/per_gt_input_features.csv)；实现：[audit_input_pairs_v2.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/audit_input_pairs_v2.py) |
| 008-QUERY32 | Diagnosis | 输入可行后是无匹配mask、无正确谓词还是候选排序损失；读取9query原始mask与分类，不训练 | 支持：1070漏检=722输入不可行+182无匹配hull+28谓词不在同query Top6+138排序/分配；C1047中721+180+26+120；不能：诊断路径不等于互斥物理原因；不证明旧关系记忆或上下文压制 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/query_audit/summary.json)；[per_gt_query_diagnosis.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/query_audit/per_gt_query_diagnosis.csv)；实现：[audit_pilot_queries.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/audit_pilot_queries.py) |
| 008-SPLIT | Intervention | 拆开跨gap区间是否恢复关系；同一query的正mask分成多个连续输出 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：拆段同时改变候选数/排序；不能单独归因于gap建模 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/pilot_retry/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/pilot_retry/per_gt_outcomes.csv)；实现：[pilot_decode.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/pilot_decode.py) |
| 008-RIGHT | Intervention | 末端+1改为补满stride是否减少损失；解码末端补至下个采样单元并截于pair结束 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不能证明采样无影响；只检验该后处理 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/pilot_retry/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/pilot_retry/per_gt_outcomes.csv)；实现：[pilot_decode.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/code/pilot_decode.py) |
| 009-FEATURE-CONTROLS | Correlation | 原始高失败率是否在控制群体构成后仍存在；GT几何代理量分箱与共同支持标准化 | 支持：高速全体+5.816pp，F中-0.788pp；小实体全体+11.279pp，F中+3.595pp；重复同视频全体-0.660pp；不能：不能把GT速度当轨迹误差；F条件化是oracle筛选且可能有选择偏差；非因果效应 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/features/results.json)；[per_gt_input_features.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit_v2/per_gt_input_features.csv)；实现：[profile_common.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/features/profile_common.py) |
| 009-R1-32 | Intervention | 评分是否影响候选选择；关系与两实体分数算术平均→几何平均 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不能由已暴露小样本外推全量；与其他变体筛选存在选择效应 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-R2-32 | Intervention | mask置信度是否代表匹配质量；原分数乘首末正点跨度内mask概率均值（含内部负点） | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不能由已暴露小样本外推全量；与其他变体筛选存在选择效应 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-R3-32 | Intervention | 去重复能否释放正确候选名额；tIoU>0.7的同pair同谓词候选抑制 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不能由已暴露小样本外推全量；与其他变体筛选存在选择效应 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-R4-32 | Intervention | 更紧时间区间是否改善检测；sigmoid阈值0.5→0.75 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不能由已暴露小样本外推全量；与其他变体筛选存在选择效应 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/old32_v1/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-R4-64 | Intervention | 冻结R4是否在另一批视频净增；阈值0.5→0.75 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：不是未见测试集；不能替代完整benchmark | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/expanded64_v2/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/expanded64_v2/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-R4-835 | Intervention | 冻结R4能否全量净提升；阈值0.5→0.75 | 支持：仅支持该操作在该固定样本上的预测效果；新增与损失见逐GT表；不能：局部短组收益不等于全体有效；不能据此声称学习到边界机制 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/full835/evaluation/summary.json)；[per_gt_outcomes.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/full835/evaluation/per_gt_outcomes.csv)；实现：[run_inference.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/run_inference.py) |
| 009-WITNESS | Diagnosis of Intervention | 恢复/损失来自几何还是排名变化；事后跟踪同pair/query/predicate | 支持：阈值改变确实使部分同候选vIoU跨阈值；103条保留候选损失中mask均值升而匹配质量降；不能：这103条是按损失筛出的条件样本，不能当全体质量负相关；不是历史上下文因果证据 | [r4_same_query_witnesses.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/full835/evaluation/r4_same_query_witnesses.json)；[boundary_extent_audit.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-COMMON-FAILURE-009/remote/full835/evaluation/boundary_extent_audit.json)；实现：[audit_r4_witnesses.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/experiment/audit_r4_witnesses.py) |
| 009-ANNOTATION | Observation / Correlation | 短关系分箱/重复多段是否是标注或fps边界假象；按整数帧与真实fps核分箱；同键合并重叠/相邻段 | 支持：6097多记录键、18063关系合并后仍分离；90帧记录1906条均>3秒；不能：分离标注不等于已核验重复动作；没跑合并GT后的新评价 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/annotation_structure/results.json)；实现：[audit.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_009/annotation_structure/audit.py) |
| 010-200-class_only | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；仅训练分类残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/class_only/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/class_only/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 010-200-mask_only | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；仅训练时间mask残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/mask_only/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/mask_only/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 010-200-full | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；训练分类与谓词时间证据联合残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/full/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps200/full/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 010-2000-class_only | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；仅训练分类残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/class_only/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/class_only/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 010-2000-mask_only | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；仅训练时间mask残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/mask_only/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/mask_only/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 010-2000-full | Intervention | 谓词时间证据耦合是否超过单独分类/时间分支；训练分类与谓词时间证据联合残差分支 | 支持：200步三臂命中不变；2000步class/full同样+3/-2，相关见证几何未变；mask_only无命中改变；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/full/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/new_container_preflight/evaluation_results_v1/steps2000/full/outcomes/summary.json)；实现：[train_adapter.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_010/integration/train_adapter.py) |
| 011-local | Intervention | 局部视觉驱动的区间解码是否优于同参数pooled控制；训练query条件化联合起止点头；局部时间视觉特征 | 支持：local确有9条C几何恢复，但总体下降；pooled仅正点估计，paired bootstrap包含0；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/final_results_v2/evaluation_v2/local/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/final_results_v2/evaluation_v2/local/outcomes/summary.json)；实现：[train_interval.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/prototype/train_interval.py) |
| 011-pooled | Intervention | 局部视觉驱动的区间解码是否优于同参数pooled控制；训练query条件化联合起止点头；以时间均值替换局部视觉，保留mask/位置等信息 | 支持：local确有9条C几何恢复，但总体下降；pooled仅正点估计，paired bootstrap包含0；不能：不能外推full835；验证32重复使用；GT-pair小缓存与检测轨迹评价输入不同；单seed未证实稳定收益 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/final_results_v2/evaluation_v2/pooled/official/summary.json)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/final_results_v2/evaluation_v2/pooled/outcomes/summary.json)；实现：[train_interval.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/prototype/train_interval.py) |
| 011-TRAIN-DIAG | Diagnosis of Intervention | 训练似然与区间质量是否一致；解码为何丢候选；对已完成训练权重及原始解码做诊断 | 支持：局部头NLL降低而dev MAP tIoU降低；大量单点/短区间被min5规则过滤；不能：候选45581不是GT损失；不能把整个负结果只归因输入域差；没有对损失函数做独立因果消融 | [adapter_dev_before.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/training_results_v2/training_v2/local/adapter_dev_before.json)；[adapter_dev_after.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/training_results_v2/training_v2/local/adapter_dev_after.json)；实现：[training.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/prototype/training.py) |
| 010-EVIDENCE-DIAG | Diagnosis | M010时间证据是否可分、是否改动原mask；读取M010 full既有权重在冻结train/dev缓存的逐时刻证据；时间打乱控制 | 支持：可核查证据可分性、阈值翻转位置与训练/开发差距；不能：目录名含011不代表M011模型结果；AUROC或dense BCE不能替代检测效果 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/remote/diagnostic_results_v2/evidence_diagnostic_v2/summary.json)；实现：[diagnose_evidence.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_architecture_011/diagnose_evidence.py) |
| 013-STRUCTURE | Correlation | 非显眼特征是否具有稳定条件差异；构建接续、并存、相对跨度、后续段等代理量；标准化 | 支持：后续段加重复控制后仍小正差；不同客体接续同视频差值反转；视频输出预算形成容量约束；不能：接续候选不是真实切换；相对标注跨度不是物体共存时长；未证实记忆/冲突机制 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/results.json)；[per_gt_features.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/per_gt_features.csv)；实现：[analyze.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/analyze.py) |
| 013-WITHIN-KEY | Correlation | 后续段较难是否只是跨视频/谓词构成差异；同键比较；增加相同时长档和双侧F限制 | 支持：严格子组首段漏852/1189、后段1304/1675；键等权差为24.4738095/1189×100个百分点；不能：不能用两组合并率差替代键等权差；同一时长档不等于相同时长；无不确定性区间/机制干预 | [results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/results.json)；[per_gt_features.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/per_gt_features.csv)；实现：[analyze.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/vrdone_common_failure_013/analyze.py) |
| DURATION-ENTITY | Observation / Correlation | 关系短是否也意味着实体短；官方GT累计/当前连续/共同连续时长分别统计 | 支持：短关系11621；两实体累计均长10230，但当前连续均长5870，共同连续>=3为5311；不能：GT支持含自动框，不是人工核验物理可见性；不能反推出预测轨迹错误类型 | [per_gt_durations.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/entity_relation_duration_20260929/per_gt_durations.csv)；[results.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/entity_relation_duration_20260929/results.json)；实现：[analyze.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/entity_relation_duration_20260929/analyze.py) |
| GT-PAIRED27 | Diagnosis / Oracle-input intervention | 纠正上游输入后能多产生多少可匹配raw mask；native输入→GT框/类/身份pair/重新ROI特征；GT关闭轨迹去重 | 支持：GT输入覆盖1678，新增1059、损失64；短<3组144→452、+328/-20；C覆盖385→1385、+1054/-54；不能：不能把oracle query挑选当自动Recall；不能分离框/特征/配对/去重的独立贡献；不能外推835 | [summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/summary.json)；[per_gt.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/per_gt.csv)；实现：[run_probe.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/gt_input_mask_probe_20260929/run_probe.py) |
| GT-TIME27 | Diagnosis | GT输入后是否仍缺少短时间mask；网格是否可表达；GT辅助选最优query/穷举合法网格区间 | 支持：亚秒二值55/333，区间75/333；短关系且共同>=3秒二值178/626；网格可行但实际区间失败832；不能：二值mask与hull tIoU不可互换；网格可表达不证明采样保留足够视觉信息；尚未证明长上下文稀释 | [per_gt.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/per_gt.csv)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/summary.json)；实现：[evaluate_masks.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/gt_input_mask_probe_20260929/evaluate_masks.py) |
| GT-ONE-TO-ONE | Diagnosis | GT独立挑query是否重复使用同一掩码；合并只谓词不同而身份/时间相同的GT；最大匹配 | 支持：二值一对一1644/2512；hull时间1658/2512；不能：GT最大匹配不是模型实际选择；没有native与GT一对一增益 | [unique_time_events.csv](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/unique_time_events.csv)；[summary.json](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/summary.json)；实现：[evaluate_masks.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/gt_input_mask_probe_20260929/evaluate_masks.py) |
| 012-UNRUN | Hypothesis | 保留原区间并学习修正收益；无可纳入本总账的完成干预 | 支持：设计/代码已准备，未训练/未评估；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | [vidor-architecture-012-mathematical-design.md](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/methods/vidor-architecture-012-mathematical-design.md)；[EXECUTION_PROGRESS.md](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/tasks/VIDOR-ARCHITECTURE-012/EXECUTION_PROGRESS.md) |
| 003-NO-RETURN | Hypothesis / planned diagnosis | 统计真实参与者替换；无可纳入本总账的完成干预 | 支持：READY_FOR_REMOTE；无incoming结果；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | [task.yaml](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/outgoing/VIDSGG-REBIND-003/task.yaml) |
| 006-NO-RETURN | Hypothesis / planned diagnosis | 定位短关系失败阶段；无可纳入本总账的完成干预 | 支持：NOT_STARTED/NOT_HANDED_OFF；部分问题后来由008处理；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | [task.yaml](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/outgoing/VIDOR-SHORT-AUDIT-006/task.yaml) |
| 007-VISUAL-NO-RETURN | Planned observation | 核验实体交互/运动语义；无可纳入本总账的完成干预 | 支持：本地抽样完成；无人工填写返回；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | [task.yaml](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/outgoing/VIDOR-FAILURE-PROFILE-007/task.yaml)；[task.yaml](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/handoff/outgoing/VIDOR-FAILURE-PROFILE-007-r2/task.yaml) |
| STRICT44-NO-RESULT | UNVERIFIED | 旧边残留/新边建立延迟；无可纳入本总账的完成干预 | 支持：本地有脚本；未找到strict_candidates.csv、strict44_raw_metrics.csv或对应结果日志；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | [batch_raw_strict44.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/gt_input_mask_probe_20260929/source_snapshot/batch_raw_strict44.py)；[run_gt_s00001.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/gt_input_mask_probe_20260929/source_snapshot/run_gt_s00001.py) |
| BIG-BINDING-NO-RESULT | UNVERIFIED | 实体绑定候选是否被过早丢弃；无可纳入本总账的完成干预 | 支持：当前项目未发现对应结果/脚本；不使用聊天数字作为实验证据；不能：任何恢复数量、性能提升或机制结论均不能由这些材料单独支持 | 当前项目未找到结果文件 |

第一阶段到此。方向比较、论文压力测试、主备方向与最小决定性实验均未在本轮开展。

复核脚本：[recompute.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/experiment_ledger_20261005/recompute.py)；结构化汇编：[build_ledger.py](/Users/xk/Documents/ChatGPT/cvpr-agent-lab/analysis/experiment_ledger_20261005/build_ledger.py)。所有新输出仅保存在本审计目录，原结果与state保持原样。
