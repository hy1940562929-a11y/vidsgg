#!/usr/bin/env python3
"""Assemble a local, provenance-preserving research handoff. No model execution."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = []
MISSING = []
COPY_REQUESTS = []


def write(name, text):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + '\n', encoding='utf-8')


def copy_source(src, dest=None, role='reference_only', limit=12_000_000):
    COPY_REQUESTS.append((src, dest, role, limit))


def perform_copy(src, dest=None, role='reference_only', limit=12_000_000):
    src = Path(src)
    if not src.is_absolute():
        src = ROOT / src
    try:
        rel = str(src.relative_to(ROOT))
    except ValueError:
        rel = str(src)
    if not src.is_file():
        MISSING.append({'source': rel, 'reason': 'not found locally'})
        return
    if src.stat().st_size > limit:
        MISSING.append({'source': rel, 'reason': 'excluded by explicit lightweight-package size limit', 'bytes': src.stat().st_size})
        return
    dest = dest or ('sources/' + rel)
    target = OUT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    target.write_bytes(data)
    MANIFEST.append({'source': rel, 'package_path': dest, 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest(), 'role': role})


PROMPT = r"""
# 给 Claude 的完整执行提示词

你是本项目的研究负责人、算法实现者和论文作者。请使用同一份证据链完成“问题核验 → 文献对齐 → 一套方案 → 完整代码 → CVPR论文初稿”，不要只给建议或散落的代码片段。不要向我承诺一次运行就能找到真实根因、获得提升或保证录用；把能完成的设计、实现、验证和写作完整交付。

## 任务与目标

任务：Video Visual Relation Detection / Video Scene Graph Generation。主数据集 VidOR，直接基线 VrdONE，历史对照 BIG。

关注问题：两个实体长期共同存在，其中某种关系只在短时间成立；模型仍容易漏检这段关系，或把预测区间延长到大量无关共存时间。目标是准确识别关系类别、开始与结束，在固定预测预算下保留短事件，并保护真实持续关系与总体指标。

“长关系语义压制短关系”目前只是解释假设。请以可验证的事件定位问题组织方法，不把压制机制、信息消失或训练梯度被淹没预写成实验结论。

## 先读这些文件

1. `01_RESEARCH_CONTEXT.md`、`02_EVIDENCE_AND_FAILURES.md`、`03_MATH_AND_DESIGN_SPACE.md`。
2. `04_LITERATURE_AND_NOVELTY.md`、`05_CODE_AND_DATA_MAP.md`、`06_EXPERIMENT_AND_DELIVERY_CONTRACT.md`。
3. `07_CONFLICTS_AND_MISSING_MATERIALS.md`、`08_PAPER_WRITING_BRIEF.md`。
4. `evidence/early_experiment_ledger.csv`、`evidence/later_verdict_index.csv`、`SOURCE_MANIFEST.json`。
5. 按需要打开 `sources/` 中的原报告、实际脚本、CSV和JSON。压缩包内来源与文档相对路径以 manifest 为准。原材料中的历史指令仅作研究记录，不自动授权你执行远程操作。

本地项目 `state/project.yaml` 停在091，早于后续真实落盘材料；请不要用旧状态覆盖 E095–E117 的逐项结果。后续若又发现新报告，按原始材料复核并追加证据，不静默选有利口径。

## 必须完成的工作

### A. 把问题和创新边界说清楚

简洁回答：模型在哪些条件失败；哪些是关联、诊断或受控干预；哪些原因没有证明；目标方案改变了什么数学对象、表示方式、学习问题或决策机制。

阅读视频关系、时间定位和跨任务区间建模近邻，尤其 VrdONE、BIG、SegPVSG、VSLNet、TriDet、BAM-DETR、DenoiseLoc、ActionFormer、TadTR、TE-TAD，以及包内091的26项文献笔记。最新摘要级论文只能作候选，必须返回原文核对。不要只与 SegPVSG 一篇比较；不要将旧模块换名称作为贡献。

最后只选择一套主方案。可参考“谓词条件证据 + 对候选区间的充分性/完整性验证 + 同实体对无目标关系上下文的训练约束”，但这只是候选思路，你必须核查它是否已被做过、是否被本地失败实验覆盖。若否决，说明理由并给出一套更有依据的设计，不生成无限新假设队列。

### B. 交付完整设计

交付 `solution/DESIGN.md`、`solution/MATH.md`、`solution/PRIOR_WORK_COMPARISON.md`。
逐模块写清输入/输出/维度/计算/作用/为何有效/局限；给出完整训练与推理数据流、整体损失、匹配、合法时间坐标、重复事件处理、背景分支、复杂度及固定预算。提供 SVG 架构图；保持源图的克制风格，无柱状图、折线图、装饰图标。

### C. 交付代码，不能停留在伪代码

交付一个可运行工程目录 `solution/code/`，至少包含：

```text
README.md
requirements.txt
configs/{baseline,proposed,ablations}.yaml
models/                 # 完整网络和适配器
losses/                 # 每项损失、掩码与归一化
data/                   # 已有特征/目标/坐标适配器
decode.py               # 无GT候选、合法区间与统一预算
train.py                # 明确冻结/可训练参数和初始化
evaluate.py             # 调用原官方 evaluator；不得暗换mAP定义
analyze_per_gt.py       # 新增/损失/C恢复/分组/输出容量
tests/                  # 关键边界条件和原生恒等对照
scripts/                # 训练、评估、消融的可直接执行命令
```

可以合理调整目录，但文件必须完整。不要用 `pass`、伪模型、随机特征结果或“自行补充原模型接口”替代核心实现。基座源快照不完整处，明确指出缺项；代码应支持 `--native-root` / `--config` / `--checkpoint`，不要假定包内含权重与数据。

每个输出标注状态：IMPLEMENTED / SOFTWARE_TESTED / REAL_DATA_EVALUATED。只做过CPU合成测试就写软件测试通过。真实实验未运行时提供人可执行的任务包和命令，不伪造日志、权重、seed均值、显著性或mAP。

### D. 完整论文初稿，使用官方CVPR2027 LaTeX

以 `official_cvpr2027/` 为模板来源，`paper_starter/` 为可编译的起点。交付独立完整 `paper/`：main.tex、各节、references.bib、图、表、补充材料和编译说明；环境允许时实际编译、检查PDF。

正文按8页目标组织，参考文献及LLM Usage Statement依官方规则处理。不通过压缩行距、改字号/页边距凑页数。匿名稿删除作者身份与本地/服务器路径。

现在先完成全文写作：Introduction / Related Work / Problem / Method / Experiments / Ablations / Limitations / Conclusion 都写成连贯论文文字，方法与实际代码一致。

数据处理约定：
- 已核验结果保留原值、原协议及来源。
- 尚未做的实验，完整写好目的、设置和表格，数值单元格用 `[NOT RUN]` / `[TBD]`，结论句保留显式待填位置。摘要不声称尚不存在的提升。
- 如需演示排版，只能另存 `paper_layout_demo/`，每页及每张表标 `SYNTHETIC DEMONSTRATION — NOT EXPERIMENTAL RESULTS`；不能混入论文真实实验表、结论、证据账或给真实基线捏造分数。默认不需要这份模拟版。
- 可以写“若实验支持则...”的备选叙事，另存 `paper/conditional_story.md`；不能把它冒充结果段。

### E. 自审并修订后交付

至少做两轮有记录的自审：第一轮检查新颖性、证据/claim、因果范围、比较公平；第二轮检查代码接口、论文/公式一致、数据泄漏、评价、图表与编译。写 `delivery/REVIEW_ROUND_1.md`、`REVIEW_ROUND_2.md`、`CHANGELOG.md`、`CLAIM_EVIDENCE_MATRIX.csv`、`ACCEPTANCE.md`。

自审不能靠删负结果通过。不可修复的科学缺口写出清楚，不将稿件标为 submission-ready。

## 资源和执行边界

- 默认只用现有预提取RoI视觉/几何特征，不更换重型视频骨干；新依赖/新特征必须单列成本和公平对照。
- 本次资料交接不授权SSH、GPU或大规模训练。遵守运行目录实际 AGENTS.md。需要学校服务器实验时只生成 `handoff/outgoing/<new-task>/`，由用户人工交接；运行在服务器CLI时也需确认该环境已有的明确授权和空闲GPU，不抢占资源。
- 不覆盖checkpoint、不改共享数据/环境、不改历史合同/结果、不泄露任何密码或密钥。
- 不复活已停止配方去追同一结论；允许用普通起止头/TFN等作为透明基线，不能把换名后的旧配方宣称新方法。

## 交付方式

先完成内部核查再持续执行已授权工作，不停在“建议下一步”。最终交付：完整文件包 + 一页结论（做了什么、真实证据到哪、方案/代码状态、论文状态、唯一最重要缺项）。若工具或材料限制导致实验无法进行，仍完成可完成的代码与带显式占位符的论文；列精确 blocker，不编造。
"""

CONTEXT = r"""
# 01 研究背景与当前问题

整理日期：2026-11-02。本包是本地资料整理与Claude交接，没有启动模型、GPU、SSH或新研究实验，没有修改正式state。

## 任务与评价

输入：视频中的实体轨迹、逐帧RoI外观、绝对框几何、主客体相对几何；原始系统先检测/跟踪，再对有向实体对进行关系预测。
输出：关系实例 `(subject_instance, predicate, object_instance, [start,end))`；同一实体对可以有多个谓词、同一谓词可以有多个分离事件。
正式关系检测匹配要求：类别三元组准确，主体/客体预测管道分别与GT的vIoU≥0.5，按置信度一对一匹配。采用仓库官方 evaluator，不把时间tIoU诊断冒充正式检测。

VidOR：官方7000 train / 835 val / 2165 test，80实体类别、50谓词。本项目val有30,142条GT，832个非空GT视频。验证集已经被反复分析、调试，不能重命名成未见测试集。

## 研究问题的可证实表述

“在同一实体对持续存在的情况下，模型能否准确判断每一种关系何时成立、何时结束，尤其是长共存窗口中的短暂关系？如何在固定预测预算下提高这些事件的检出与边界精度，同时保护持续关系与总体检测性能？”

用户偏好的机制叙事是“长期共存中的长关系特征压过短关系证据”。当前实验只支持失败现象和若干条件关联，没有可靠独立操纵证明该因果路径。写论文可以把它作为假设，不能作为已证根因。

## 阈值不是同一套

|材料|短关系|长共存|注意|
|---|---|---|---|
|早期005–089|严格 `<3秒`；亚秒 `<1秒`|GT连续共同存在≥10秒|按真实fps；90帧不自动等于<3秒|
|TICK用户报告|`≤2秒`为主，也有≤3秒分析|报告内不同S_C定义待逐项绑定|250视频/5518短/11469长不是全val30142|
|最新数学表达|占空比 L/T 很小|连续共存窗口C|占空比与绝对时长共同报告，不混作相同切片|

GT实体寿命、两实体各自累计寿命、同时连续共存、预测轨迹片段、关系标注跨度，五者不等同。

## 当前主干的实际数据流

```text
原视频 → 冻结检测/跟踪与RoI特征
   → 主体特征+绝对几何 / 客体特征+绝对几何
   → BSA/SOS主客体时空交互 → 相对几何融合 e_so
   → 多尺度时间编码器 → Temporal FPN → mask_features h(t)
                         └→ 9个关系query解码 → z_k
   → 类别头 softmax(背景+50前景)
   → mask_embed(z_k) · h(t) → 时间logits → sigmoid>0.5
   → 首末正点hull → stride4/帧坐标映射 → min5 → 原排序/TopK
```

配置已核：`num_classes=50`、`num_queries=9`、`backbone_arch=[2,2,3]`、`max_seq_len=512`、窗口9、类别/focal/Dice系数2/2/5、特征stride4、谓词Top6、预测min5原始帧。注意训练loader的min15与推理min5不是同一参数。
原生query已经对应实例，不是整个实体对只有一条完全类别无关曲线；但同一query的Top6谓词复用该query的时间mask，这是一个具体接线事实。

## 已提出但未被采纳的方案

- TICK/TICK-Net：状态/事件双路径、残差差分、峰值+时长；时间诊断改善没有转化成公平谓词匹配提升。
- ORCA：正交去共存、谓词条件高斯门、对比防膨胀；原始设计文本包含未证实的根因和不准确的原生0.4阈值，不能照抄。
- EvidenceRepair：局部区间修复、共存/证据参考、候选质量；历史只读记忆显示候选崩塌和长关系损失，但本包未重新连接服务器核对数值。
- CDSE/KTE/LGSAD：完整候选代码及29项CPU软件检查已落盘，未有真实方法增益。
- 普通边界版：保留主干/分类/粗mask，增加query条件起止分布、训练集软时长先验、有序联合区间解码。仅绘图设计，尚未实现新实验；组件已有直接先例，适合作强控制。

## 用户限制与偏好

只用现有预提取特征；希望架构对准研究问题，拒绝纯改名、外挂换第9槽、扩大候选虚增召回、换重型骨干涨点。希望完整数学、模块说明、可执行代码、论文英文初稿。表达简单直接，图中只用必要实物图/视频/向量，模块用几何形状；不需要装饰曲线/柱状图。最终图用SVG，正文图字体与既有参考统一。
"""

EVIDENCE = r"""
# 02 实验事实、失败路线与证据范围

以下从本地原报告、JSON、CSV及用户报告整理。完整来源快照随包；本轮没有重新训练或重算完整模型。索引“已读报告”不等于独立重新复现实验。用户附件与记忆信息另标等级。

## A. 初始基线与共同失败

来源：`sources/analysis/experiment_ledger_20261005/{REPORT.md,ledger.json}`；原配对summary/instance_comparison随包。

|Top100原始分类|数量/全部30142|比例|
|---|---:|---:|
|A 两者命中|2487/30142|8.2509%|
|B 仅VrdONE命中|1809/30142|6.0016%|
|C 两者漏检|25373/30142|84.1782%|
|D 仅BIG命中|473/30142|1.5692%|

VrdONE命中4296/30142，BIG2960/30142。原VrdONE全val mAP=0.117370619（0–1）；R@100=4296/30142。BIG是发布hit缓存，输入/训练与重建VrdONE不是同源受控条件。

共同失败C：<1秒3151/3167；1–3秒7475/8454；3–10秒10296/12193；≥10秒4451/6328。它们是分层共同失败，不是VrdONE单模型原因分解。

当前VrdONE输入oracle可行F=13294/30142；C中F=8545/25373，输入不可行16828/25373。不能将这批全部归咎于关系头，也不能把“存在可行输入”说成模型实际选择正确。

## B. 几何/画像与控制

短关系<3秒11621/30142，C10626/11621；其中4160/10626输入可行。小实体C4917/5242，其中600/4917输入可行，4317/4917不可行。高速样例F条件下标准化关联约−0.788pp，不能预设高速是关系头主因。多段、后续段、切换代理对控制敏感，未建立记忆因果。

来源077：F共同支持5334条，短组标准化漏检高22.073pp；同视频/同有向pair/同谓词支持3060条，差12.385pp。短关系内长共存对照支持3577条，差15.526pp。更严格同视频/pair/predicate/细时长F仅69条，差−2.174pp、支持不足。全部是相关，不是拉长上下文的因果效应。

## C. 原生解码/小头：已尝试过

|实验|范围|核心结果|边界|
|---|---|---|---|
|008拆mask|32视频/1209GT|139→140；新增2/损失1，mAP略降|候选/排序变化，不能称总体有效|
|008右边界补齐|同上|新增2/损失2|未明显改善|
|009阈值0.5→0.75|全val30142GT|短<3新增24/损失8，长≥3新增54/损失100；总体净−30，mAP下降|收紧能局部救短，也会损失短和长|
|M010逐谓词证据|32/1209，200或2000步|2000步class/full净+1，恢复集合相同；mask-only无变化|收益来自排序，没证明时间证据有效|
|M011起止头|32/1209，2000步|local新增11/损失29；pooled新增4/损失3|端点NLL下降不保证最终区间改善|
|078联合区间|4739GT|588→577命中，新增4/损失15|普通联合起止已有本地负结果|

更多早期实验和未返回项见42条总账；把“未做”与“无提升”严格区分。

## D. 086–089锁定的残余现象

锁定群体：400条短<3秒、长共存≥10秒且原选连续实体段覆盖GT，23视频；亚秒163条。GT实体输入、固定诊断pair；不是随机val样本或官方RelDet主结果。

- 086：理想网格400/400能放下；默认任意query hull47/400；全阈值hull121/400。全阈值/全分量355/400是GT择优且扩容量的oracle，不能当方法收益。
- 087同容量线性probe：369/400有训练类别支持。fine/fused/position局部AUROC0.53972/0.52273/0.40973；fine−fused +0.01699，CI[-0.02644,0.06156]。判INCONCLUSIVE_NO_EXPANSION，没有支持明显融合损害。
- 088固定容量读出：H47/400，单分量C53/400；新增7/损失1。视频等权差+2.1835pp，CI[0.4168,4.4321]，不足预设5pp；NOT_SUPPORTED。H/C均优于对应容量null，不是“没有任何定位信息”。
- 089：347条存在前景Top1为目标的query，单分量334/347仍失败；F与T实际换query=0/347。支持残余区间缺陷，不证明“主因是选错query”，也不证明谓词分类已完全解决。

## E. 原因干预 E091–E117 的实际判决

完整逐项原VERDICT见 `sources/analysis/cause_arch_*/VERDICT.md`，机器索引 `evidence/later_verdict_index.csv`。以下每条仅限自己的预算/接口/数据/门槛，不能否定整类理论。

|实验|操纵|当前结论|
|---|---|---|
|E091|末层mask embedding，原loss+真实CDF/W1 vs原loss/伪几何，3seed×500|NOT_SUPPORTED；B−A+0.7743pp，CI[0,1.7060]，低于5pp；9/9完成|
|E097|局部状态对比|旧F+原H+单slot限制下oracle最多149/334，达不到多数167；容量门停止，训练0/9；不等于局部证据无效|
|E098|cross-attn行列平衡|严重低读取47/334，CI上界18.55%远低于多数；准入停止，训练0/6；不等于注意力没有问题|
|E099|分类梯度定向去冲突|修复Tensor接口后C0 88/88完成，P未稳定优于O/W；准入REJECT，正式0/9|
|E100|跨层事件分配N/T/X|扩dev16视频302事件，共同支持7视频23短37长；9/9完成，334净T−N=+6/0/−2，REJECT|
|E101|短事件focal正点单位化|9/9；334净T−N=−4/−4/−7；REJECT|
|E102|区外负点升权|9/9；净+6/+1/+7，但不稳胜错位X、eta不足；REJECT|
|E103|单位质量W1替换mask主监督|9/9；净−4/−13/−3；REJECT；不是E091同一干预|
|E104|跨层mask聚合|净+6/+2/−4；REJECT|
|E105|软质量分位区间替代hull|净−2/+5/+4；T与反转X命中完全相同；REJECT|
|E106|时间adapter|净+3/−5/+5；REJECT|
|E107|mask-feature adapter|净−7/−3/−7；REJECT|
|E108|短事件denoising辅助|净+2/0/−2；REJECT|
|E109|共存条件FiLM|净+3/+1/+7；幅度小且不稳胜X；REJECT|
|E110|时长匹配代价|净+3/+6/+7；eta约0.018不够且X可竞争；REJECT|
|E111|短先验query bank|所有臂Q=12；净+2/0/−4；REJECT|
|E112|短+共存过采样|小预算全正；5seed×2000后净+3/−4/−3/−1/+7；SCALE_REJECT|
|E113|Moment-DETR式中心宽度替代hull+辅助|净−24/−18/−17；REJECT/HARMFUL|
|E114|temporal-GIoU匹配|小预算全正；5seed×2000后净+4/+10/−8/0/+11，eta约0.011；SCALE_REJECT|
|E115|ActionFormer式TFPN-on-f|净+2/+3/−7；REJECT|
|E116|span辅助但仍hull|净+1/+8/+4；小且X不稳，REJECT；未见scale验证|
|E117|centerness/Gaussian mask-loss重加权|T334仅0/0/1，净−39/−40/−41；REJECT/HARMFUL|

后续目录存在E118，但本轮发现无REPORT/VERDICT，不写成已完成。不要自动接着训练它。

E095与后续E098原生9槽C0、F、P/C读出不同；同一334组C0可为0/29等，不能跨合同直接比提升。9query不等于全局最多9条谓词预测，因为一个query可有多个类别候选。

## F. TICK报告（用户提供，非本轮raw复算）

原文随包：`user_materials/TICK_EXPERIMENT_REPORT_USER_PROVIDED.txt`。该报告自己承认旧协议泄漏/候选膨胀，不能挑旧正数作主表。

- 250视频；短≤2秒5518条、长11469条；5配对seed100–104。
- A2/A4/Tversky未通过；A5旧路由+8.75pp存在训练评测重叠/时间only/候选更多。
- 干净训练180视频、Event GT213；K9 pred-aware短约−1.69pp、长约−5.81pp。
- B4时间only：N.q短11.29%、OER0.957；peak+median短17.93%、OER0.385，但长8.04%、谓词感知很低；这是时序读出诊断，不是整体成功。
- E1分类竞争、E2统一排序、E3重锚定均未带来干净端到端收益。
- 真GT区间Event谓词probe：207训练、734验证，feature+class MLP top1 18.0%、top5 59.7%，先验13.1%/39.5%；oracle定位也没有证明现有特征可以轻松分类。

所有此表数值的原始逐seedJSON/输出尚未随用户文本完整提供，不可给均值补造TP/GT分子，也不扩为全val官方mAP。

## G. 因果长语义操纵（用户提供）

221成功/250尝试，机制子组205，216视频；训练集均值方向操纵next_to语义。构造投影幅度通过，但独立query语义分数变化小且反向，gate=False；MANIPULATION_FAILED。weakening新增1/221、损失0不能证伪/证实长语义路径。原文在 `user_materials/CAUSAL_CONTEXT_USER_PROVIDED.txt`。

## H. 真实样例与图

VidOR视频1018_4199369046，GT实体#3/#4连续共存57.29秒；持续next_to及8段watch。实际最高分watch预测[220,1529)，目标[1440,1474)，fps29.97002997；时间IoU=34/1309，时长比1309/34=38.5。注意完整预测还跨过其他watch事件，不能画成单事件的简单外扩。图与case manifest/CSV随包。它是例子，不是总体机制证明。
"""

MATH = r"""
# 03 数学问题与可探索设计空间

## 定义

对同一有向实体对，连续共同存在区间 C=[0,T)，时序特征X={x_t}；关系p的事件集合 G_p=∪_r[s_r,e_r)。单事件长度L=e−s，占空比π=L/T。短关系可绝对短，也可相对共存窗口很短；两者分开。

联合输出 `(p_hat,s_hat,e_hat)=argmax P_theta(p,s,e|X)`；存在多个事件时为无序实例集合，允许跨谓词重叠，不强制所有谓词互斥。

时间诊断 tIoU=|I_hat∩I*|/|I_hat∪I*|。如果预测完整包含GT，只额外长Δ，则 tIoU=L/(L+Δ)，因此tIoU≥0.5要求Δ≤L。若等长整体平移δ，则 tIoU=max(0,L−|δ|)/(L+|δ|)。数学敏感性成立，不自动证明模型成因；正式检测还要求双实体vIoU和类别/匹配。

## 共存抑制：待检验因果假设

把输入分成固定局部目标证据U与远处无目标关系上下文B。时间分数ell_p(t)，局部区分间隔

M(U,B)=mean_{t∈I*}ell_p(t)−mean_{t∈N}ell_p(t)。

N是经合法标签界定的邻域负点，排除同pair同p的全部其他事件；未知不自动当负。若固定U、有效且匹配地改变B使M持续下降并伴随命中下降/边界膨胀，才支持这一操作范围的干扰路径。需独立语义操纵检查，不能仅投影构造通过。

## 供Claude评估的单一设计切口（UNVERIFIED，不是正式方法）

“谓词条件的区间证据验证”：在现有特征上，对每个候选区间同时验证内部充分性、外侧无目标证据、以及被裁去部分是否仍有同一事件证据。改变区间评分/证据归属，而不是一律缩短。

可思考统一得分 S(p,I|X)=S_support−λ S_unsupported_extension−μ S_omitted_event，但必须给出可计算、可训练且不依赖推理GT的定义。不要仅把三个任意加权分数命名成新理论；需要证明相对普通span/boundary、TFN、同pair对比基线的必要性。

四个创新位置：
1. 取证：谓词参与局部特征选择与同pair参考比较；单纯[T,C]线性分类已有先例/M010。
2. 成段：内部证据、外扩代价、收缩丢证据共同决定边界；避免全时间归一化导致长关系只能抢一个峰，也避免短段天然因均值高获优。
3. 事件：同一谓词多段独立解释；输出预算、同分排序、NMS和身份绑定透明。
4. 训练：固定局部证据，改变已核验无目标关系远上下文；事件区间应保持，真实关系持续证据改变时模型应响应。上下文拼接和位置改变需配对控制。

## 普通边界控制的数学形式（已有先例，非原创贡献）

u_kt=MLP([W_h h_t || W_z z_k || (W_h h_t)⊙(W_z z_k)])；
p_s^k(t)=softmax_t(w_s^T u_kt)，p_e同理；
S_kp(i,j)=log p_s^k(i)+log p_e^k(j)+λ log p_L(j−i|p)；
(i*,j*)=argmax_{legal i<j}S_kp(i,j)。

保留粗mask作为辅助，GT匹配实例端点CE；先验只从训练拟合、有完整时长支持，不把同一谓词硬编码短/长。一个query只输出一个连续段，不能声称自动解决内部空洞和多次事件。

## 理论上不能直接声称的结论

- 平均池化 toy model h_bar=b+πδ 可以说明某种汇聚的稀释，不证明真实VrdONE是线性低通，更不能把短关系等同高频运动。
- LayerNorm不沿时间滤波，不保证抑制恒定维度；特征正交投影不保证语义独立，可能删掉真实事件类别证据。
- 全局均值不自动等于“共存背景”，其残差不自动等于“纯事件”。
- sigmoid逐帧监督最优为条件概率；证据不确定、特征相似、标签差异都可能产生宽平台，不能仅凭宽平台认定损失几何为因。
- 起止各自argmax可能形成非法顺序；用合法联合区间解码。
- 对所有类共享同一条时间曲线、先固定粗窗再不可逆裁剪都可能限制定位，但需源码和受控比较证明本模型的实际影响。
- 两端/时长先验只改变输出变量，不创造缺失视觉信息；无条件缩短会伤真实长关系。
"""

LITERATURE = r"""
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
"""

CODEMAP = r"""
# 05 代码、数据、接口与复用地图

## 随包代码

`code_reference/native_snapshot/`：从本地导入的030 source_snapshot复制的模型/loader/helper evaluator；不是本轮从作者最新仓库拉取。`code_reference/native_snapshot/configs/vidor.yaml`等来自另一个已核对的服务器快照，manifest记录各自来源，不假定完整上游仓库/版本统一。

重点：
- `models/maskvrd.py`：MaskVRD，forward_training、forward_test、inference、bipartite_match、loss。
- `models/predictor.py`：MaskedTransformerPredictor；query mask embedding与FPN特征einsum。
- `models/backbones.py`、`fpns.py`、`local_transformer.py`、`blocks.py`：BSA/时间主干与时间mask传播。
- `dataloaders/{vidor,vidvrd}.py`、`category.py`：类别顺序、配对、区间坐标与标签。
- `VidVRD_helper/evaluation/visual_relation_detection.py`与`utils/evaluate.py`：官方关系评价及包装。

`code_reference/cdse_kte_lgsad/`：relation_enhancement.py、vrdone_adapter.py、integration_example.py、prepare_text_anchors.py、config、29项测试与历史VALIDATION.json。它是完整可参考实现，但研究状态UNVERIFIED；历史CPU随机权重/合成数据通过不证明精度。

`sources/analysis/cause_arch_*/`：各实验实际run/loss/adapter/admission/decode脚本和报告。请先读判决再复用，避免继承已经失败的训练配方。

## 必须核对的张量与接口

- 时间特征常为[B,C,T]，query常为[B,C,Q]或[L,B,Q,C]，按原predictor实际检查；不凭图猜维度。
- 前景50类+背景；当前native背景index0，TopK用`pred_probs[...,1:]`，不要错位类别ID。
- mask logits是原始实数，focal要求logits；不要对sigmoid输出再做with_logits。
- `_mask`、valid/output_mask、loss_mask用途不同；GT时间网格只做同网格右padding，不能随意插值改变事件长度。
- `num_masks`等TorchScript接口要求Tensor，不能传Python int/float。loss按关键字调用，normalizer数值/设备/类型保持。
- 原生复现检查固定同输入/布局/dtype/TF32和padding；历史门槛maxabs≤4e−6，不按结果放宽。两次浮点loss不可严格Python `==`，记录abs/rel差并使用已约定技术容差。
- 秒单位=真实采样stride/FPS；512是最大token数，不是fps或秒数。半开帧区间、最后token+1与+stride不能悄悄混用。
- shared predicate evidence GT用同pair同p事件并集；query实例mask target用每个独立实例。周边负点排除该谓词全部合法事件、未知标签掩蔽。
- 不能把GT谓词/GT短长/GT最佳分量作为部署条件；oracle仅诊断。

## 现有数据切分与风险

早期train32：官方training，缓存350pair/563GT，仅26/50谓词有正例；adapter-dev8：89pair/169GT；eval GT27：2573GT，反复曝光。E100-A1扩mechanism-dev16视频/302事件，共同支持7视频/23短/37长，原400/334/163等评价成员未改。其他TICK训练180、Event207/213来自另一套协议，不能混为一套数据。

原始native_cache与checkpoint不随轻量包复制。应按 `CACHE_AND_RESOURCE_CHECKLIST.md`核对本机可用性，缺数据时提交可执行准备脚本及精确字段，不能用087 score NPZ伪训原生分支。

## 历史服务器路径（仅定位，不是授权/在线事实）

用户曾给出学校私人目录：
`/home/user/xiongke/cvpr_lab/tick/`
`/home/user/xiongke/cvpr_migrate/rebinding_project/VrdONE/`
`.../VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth`
`/home/user/xiongke/cvpr_lab/tick/{train,val}_gt_features/`
`tick/{evidence_repair,orca,scwd,ls_compete,short_rel,causal_pl_context}/`

这些路径本轮没有远程核实。历史AutoDL端口、socket/GPU编号均不作为当前入口，不提供密码，不自动连接。真正训练由用户在合法环境交接执行。
"""

CONTRACT = r"""
# 06 Claude实现/实验/交付验收合同

本包要求一次完整推进方案、代码与论文初稿，不是自动无限实验任务。新设计不得靠改变旧合同追正数；需要新协议时预先登记一份新的比较合同，并保留负结果。

## 方案验收

只提出一套主架构；各模块能追溯到一个明确待解决限制。至少区分：谓词证据错误、时间区间错误、输入轨迹错误、过滤/排序错误。凡主张“某机制是根因”须有有效独立操纵、matched control与足够共同支持，方法增益本身不足以证明根因。

## 工程验收

1. 原生关闭新分支完全复现，记录模型/配置/数据SHA、数值布局、mask/类别及解码一致性。
2. 所有路径可配置；原权重不覆盖；依赖版本明确；训练和评价脚本可直接执行。
3. 软件测试覆盖前后向、finite梯度、变长padding、空目标、无正query、长关系、同pair同p多段、重叠不同谓词、真实FPS、边界帧、min5、固定预算、类别ID。
4. 真实数据最小检查记录使用的train/dev/eval视频和GT数量、是否原基座已见；不存在数据时准确NOT_RUN。
5. 各组新增/损失/C恢复附逐GT ID和预测见证；组重叠不求和。官方mAP和诊断time-match分开。

## 新方案最少比较矩阵（供冻结前完善，不是本轮已批准开跑）

|对照|目的|固定项|
|---|---|---|
|C0原checkpoint|基座参照|原数据/输入/evaluator|
|同预算原生续训|排除训练本身|初始化/步骤/批次/seed|
|普通query条件边界头|排除只换输出变量|相同特征/候选预算/训练量|
|TFN或清楚标注的TFN-style适配|与先定位后识别比较|同骨干/输入/任务评价；适配≠作者复现|
|选定主方法|检验实质新机制|上述匹配条件|
|机制matched control|排除额外参数、随机缩窗/候选增量|同容量/长度/训练预算|

不要在同一轮漫无边际新增十几个loss。主方法核心的必要消融预先确定，开发仅train/dev，最终冻结独立视频与额外数据集。

## 必报指标

- 正式mAP、R@50/100、按谓词正确且双管道匹配的短/长/全体检测。
- 固定<1、1–3、3–10、≥10秒及长共存条件组；若主张≤2秒另列，不换旧组。
- 时间tIoU0.3/0.5/0.7、端点误差、预测/GT长度比、过延展、无输出、错误位置；每项定义分母和匹配规则。
- 新增正确、损失原正确、固定原C恢复；真实输出数、非空query、背景剪枝、TopK竞争。
- 训练参数量/FLOPs/墙钟/GPU小时；相同条件多seed、按视频配对不确定性。
- 真实长关系与整体保护；目标组改善但总体下降不得判整体成功。

成功门槛在看新结果前冻结。历史eta=0.20、+5pp等仅属于旧原因合同，不能无解释继承为所有新方法标准。也不能看完数字再改成恰好能过的门槛。

## 文件级交付清单

- solution/{DESIGN.md,MATH.md,PRIOR_WORK_COMPARISON.md,architecture.svg}
- solution/code/：完整源文件/配置/训练/推理/评价/测试/复现命令
- experiments/{PROTOCOL.md,split_manifest.json,ablation_matrix.csv,resource_estimate.md}
- results/：仅实际跑出的日志/预测/summary/per_gt/SHA；无结果时README明确NOT_RUN
- paper/：官方CVPR LaTeX全文、BibTeX、可编辑图、表、补充、PDF若编译可用
- delivery/{REVIEW_ROUND_1.md,REVIEW_ROUND_2.md,CHANGELOG.md,CLAIM_EVIDENCE_MATRIX.csv,ACCEPTANCE.md}

不得把方案设计/CPU软件通过/旧论文稿/模拟表，混写成正式方法提升、因果完成或submission-ready。
"""

CONFLICTS = r"""
# 07 数据冲突、材料不足与不可直接继承项

1. **正式state过期**：project.yaml仍停091设计、remote inactive等；不能作为2026-11-02真实服务器状态或否定后来结果。本轮不更新正式state。
2. **E091历史版本冲突**：早期TECHNICAL_FAILURE与后来有效results_v2均保留；当前有效最终VERDICT是NOT_SUPPORTED。只引用旧失败会漏掉9/9的真实负结果。
3. **E100版本冲突**：原dev8/169支持不足、旧loss接口失败属于历史attempt；最终E100-A1扩dev16/302，9/9完成且科学REJECT。不能说仍只技术失败/训练0/9。
4. **E112/E114小预算正点被scale否定**：不能只选小预算UNDERPOWERED_POSITIVE；5seed×2000的scale结果也随包。
5. **官方RelDet/时间oracle不同**：400/334诊断、GT27掩码覆盖、TICK时间only、pred-aware、官方mAP不是同一指标，不纵向拼成一条性能曲线。
6. **同数不同口径**：086的47/400可跨同身份连续段，088的47/400固定pair/同源新分数；不能因为相同数字就认作同一实验。F/T/U和P/C也不得混用。
7. **短关系边界不同**：<3、≤3、≤2、<1，按实际fps定义；共存≥10不同于两个实体各自≥3。
8. **mAP口径**：项目官方实现是非空GT视频AP平均，不是随手改成类别mAP；不要用TP/GT解释AP或给AP配假分子。
9. **零结果与未测**：没训练/未评价记NOT_RUN，不写0；oracle上界不是实际新增；小错误/低读取门未过不是训练算法本身无效。
10. **TICK文本证据等级**：用户报告可用来追踪历史与规避重复，但缺完整原始JSON/逐seed预测。原文含过度因果表达，不能全盘继承。原实验分母、CI和百分比无法反推出整数TP时标“未提供”。
11. **ORCA设计并非事实**：原文阈值0.4与源码0.5冲突；把均值当共存背景、投影当纯瞬态、oracle闭合当语义充分均未成立。
12. **用户聊天新增统计**：整窗口基线610/731、1211/1332、0/5539、0/8724及State/Activity/Event网格异质性是用户聊天提供；本轮没有找到对应原CSV/脚本，另存USER_REPORTED，不升格核验结果。
13. **远程ER/ORCA复核**：记忆曾记录ER候选崩塌/长组损失，服务器原STATUS/JSON本轮未重新读取；仅作追索线索，不把记忆数字写论文。
14. **原材料不完整**：本包不含视频全量、数据盘、原生训练张量、checkpoint或真实学校服务器代码全量。代码快照与评价脚本可供实现，不能宣称已能无数据启动正式实验。
15. **独立泛化缺失**：GT27/400、32/64、val250等反复曝光；新增数据集实验未完成。PVSG源含VidOR需查ID重叠。
16. **最终图并非最终方法**：原用户路径SVG可能含后续手工修改；包内单独保留v3与当前canonical，均仅设计参考，不是模型版本标签。

Claude需要先输出简短证据冲突处理表。不得为实现“完整故事”补造结果；应通过新代码/新实验明确修正方案或缩小claim。
"""

WRITING = r"""
# 08 论文写作与CVPR模板说明

官方来源已下载并记录SHA：`official_template_receipt.json`，CVPR2027-v1 tag（a236934短提交）。官方文件原样保留于 `official_cvpr2027/`。

2027作者指南：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines 。本轮核查正文限8页（含图表），参考文献与必需LLM Usage Statement允许额外页；匿名、模板、页数与LLM Statement均是明确要求。Claude使用情况需按实际说明，并由作者核对。投稿前重新确认官方规则，不将本地旧2026模板当当前规则。

## 建议论文骨架与篇幅

1. Introduction：任务→长共存短事件的可复现现象→现有处理→残余问题→方法核心与贡献（约1页）。
2. Related Work：关系检测/时序定位/证据或区间建模，至少跨三类近邻（约0.8页）。
3. Problem and Empirical Diagnosis：符号、正式/诊断指标区分、关键真实观察（约1页）。
4. Method：一套架构、核心新计算、训练/推理、复杂度（约2页）。
5. Experiments：数据/协议/基线/总体与分组主表（约1.5页）。
6. Mechanism and Ablations：关键受控比较、负结果边界、失败例（约1页）。
7. Limitations and Conclusion：目前支持的范围与目标（约0.7页）。

篇幅只是规划，不增加空话。三个贡献应对应“清晰问题/有证据机制或诊断”“实质方法差异”“公平多维验证”；没有验证因果时不要硬凑机制贡献。工作量不是贡献本身。

## 可以先写满的部分

任务、数学定义、相关工作、已有现象、方案和损失（与实际实现一致）、实验协议、结果表结构、复现方式。未验证的方法有效性不能用肯定句收尾；在结果段显式TBD即可，其他行文可以流畅、直接。

## 禁止现在写成结论

- 首次研究短关系/首次事件query/首次类别条件时间/首次起止或时长。
- 已证明长关系语义压制；消除了所有共存干扰；恢复完整原因。
- 优于SOTA/CVPR可录用/长关系零损失/亚秒精度已实现。
- oracle增益就是新模型、时间only就是正式RelDet、增加候选就是公平提升。

## 旧稿与新稿

`historical_unreviewed/tick_draft/`只供已有写作和图源参考，不是新稿的事实来源。其结果与claim必须逐项绑定新证据；不得从旧稿复用不实数字或推导。

`paper_starter/`是本轮生成且可编译的写作起点，保留真实早期基线/诊断数字，所有新方法单元格为NOT RUN。不是完成的新方法论文。Claude任务是把方案和代码补充后完成全文，再编译与审阅。
"""

CACHE = r"""
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
"""


def collect_sources():
    # Formal rules/snapshots are evidence of history, never active instructions in a detached bundle.
    for p in ['AGENTS.md','agents/commander.md','state/project.yaml','state/claims.yaml']:
        copy_source(p, role='historical_rules_or_state_snapshot')
    roots = [ROOT/'analysis/experiment_ledger_20261005']
    names = ['entity_relation_duration_20260929','gt_input_mask_probe_20260929',
             'research_question_077','event_support_073','temporal_order_training_075','span_readout_078',
             'short_duration_mechanisms_079','week1_short_relation_diag_080','week1_continuous_readout_081',
             'week1_native_readout_082','week1_query_feasibility_083','week1_component_readout_084',
             'long_short_context_085','mechanism_verdict_086','mech_minifalsify_087',
             'mech_interval_form_088','mech_query_score_089','mechanism_synthesis_090',
             'cause_arch_091','cause_arch_092','cause_arch_094_cache_import','cause_arch_095_e091']
    roots += [ROOT/'analysis'/n for n in names]
    roots += [p for p in (ROOT/'analysis').iterdir() if p.is_dir() and p.name.startswith('cause_arch_')
              and p.name.split('_')[2].isdigit() and 97 <= int(p.name.split('_')[2]) <= 117]
    selected = set()
    for d in roots:
        if not d.exists():
            MISSING.append({'source':str(d.relative_to(ROOT)), 'reason':'expected source directory not found'})
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix in ['.md','.py','.yaml','.yml']:
                selected.add(f)
            if f.is_file() and f.name in ['ledger.json','results.json','config.json','status.json','execution_receipt.json',
                                         'review.json','derived_summary.json','counts.json','capacity_audit.json',
                                         'boundary_interpretation.json','intersection_manifest.json','intersection_members.json']:
                selected.add(f)
    for f in sorted(selected):
        copy_source(f)
    # Small important numerical tables and original summaries.
    specific = [
        'analysis/mech_interval_form_088/results_v1/summary.json',
        'analysis/mech_interval_form_088/results_v1/per_gt.csv',
        'analysis/mech_query_score_089/results_v1/summary.json',
        'analysis/mech_query_score_089/results_v1/per_gt.csv',
        'analysis/mech_minifalsify_087/support_primary400.csv',
        'analysis/mech_minifalsify_087/support_subsecond163.csv',
        'analysis/week1_short_relation_diag_080/results_v1/diag_summary.csv',
        'analysis/week1_short_relation_diag_080/controls_v1/controlled_summary.csv',
        'analysis/cause_arch_095_e091/results_v2/evaluation/summary.json',
        'analysis/cause_arch_095_e091/results_v2/evaluation/per_gt.csv',
        'analysis/cause_arch_100_e100/attempt_04_membership_expand/CONTRACT_AMENDMENT.md',
        'analysis/cause_arch_100_e100/attempt_04_membership_expand/MEMBERSHIP_FREEZE.md',
        'analysis/cause_arch_100_e100/attempt_04_membership_expand/final_statistics/summary.json',
        'analysis/cause_arch_100_e100/attempt_04_membership_expand/final_statistics/verdict.json',
        'handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/summary.json',
        'handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/group_counts.csv',
        'handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/instance_comparison.csv',
        'docs/deliverables/20261020_short_relation_cause_arch.md',
        'docs/deliverables/CACHE_IMPORT_CHECKLIST.md',
        'reviews/week1_diag_080_review.md','reviews/week1_native_readout_082_actual_review.md',
        'reviews/week1_query_feasibility_083_review.md','reviews/week1_component_readout_084_review.md',
        'reviews/short_duration_mechanisms_079_actual_review.md','reviews/long_short_context_085_actual_review.md',
        'reviews/vrdone-failure-008-review.md','reviews/vidor-duration-controls-007-r2-review.md',
    ]
    for p in specific:
        if ROOT/p not in selected:
            copy_source(p, role='numerical_or_review_source')
    for f in (ROOT/'literature').glob('*.md'):
        copy_source(f,role='historical_literature_note_not_new_reproduction')
    for f in (ROOT/'methods').glob('*.md'):
        copy_source(f,role='historical_method_note')
    native=ROOT/'handoff/incoming/VIDOR-NATIVE-FEATURE-CURRENT-030/full_v1/results_v1/source_snapshot'
    for f in native.rglob('*.py'):
        copy_source(f, 'code_reference/native_snapshot/'+str(f.relative_to(native)),role='native_source_snapshot')
    for p in ['configs/vidor.yaml','eval.py']:
        copy_source('analysis/vrdone_server_snapshot/'+p,'code_reference/native_snapshot/'+p,role='additional_snapshot_config_or_entrypoint')
    cand=ROOT/'output/cdse_kte_lgsad_20261029'
    for f in cand.iterdir():
        if f.is_file() and f.suffix in ['.py','.json','.md','.txt']:
            copy_source(f,'code_reference/cdse_kte_lgsad/'+f.name,role='candidate_software_only')
    fig=ROOT/'docs/figures/figure2_evidence_v4'
    for name in ['CAPTION_AND_SOURCES.md','case_manifest.json','case_gt.csv','case_predictions.csv',
                 'figure2_evidence_en.svg','figure2_evidence_en.pdf','figure2_evidence_en.png']:
        copy_source(fig/name,'assets/figure2/'+name,role='real_case_illustration')
    fig=ROOT/'output/relation_evidence_architecture_20261027/figure_revision'
    for name in ['REVISION_BOUNDARY_INTERVAL.md','REVISION_PREDICATE_EVENT.md',
                 'architecture_reference_style_boundary_interval_v3.svg',
                 'architecture_reference_style_font_matched.svg']:
        copy_source(fig/name,'assets/architecture/'+name,role='design_only_not_validated_method')
    for name in ['main.tex','references.bib','internal/review_audit.json']:
        copy_source('paper/tick_cvpr_draft_20261025/'+name,'historical_unreviewed/tick_draft/'+name,role='historical_manuscript_not_evidence')
    attachments = [
        ('/Users/xk/.codex/attachments/09565fac-0d19-459e-a29b-cb46748be5e7/已粘贴的文本.txt','TICK_EXPERIMENT_REPORT_USER_PROVIDED.txt'),
        ('/Users/xk/.codex/attachments/84918fcb-f218-4313-9fcc-3b41464db0af/已粘贴的文本.txt','CAUSAL_CONTEXT_USER_PROVIDED.txt'),
        ('/Users/xk/.codex/attachments/77c95a15-a454-4eff-ab13-1b2eb1017abb/已粘贴的文本.txt','EARLY_ABC_PROPOSALS_USER_PROVIDED.txt'),
        ('/Users/xk/Desktop/ORCA_NET_DESIGN.md','ORCA_NET_DESIGN_USER_PROVIDED.md'),
    ]
    for src,name in attachments:
        copy_source(src,'user_materials/'+name,role='user_provided_unverified_report_or_design')


def generate_indexes():
    d=json.loads((ROOT/'analysis/experiment_ledger_20261005/ledger.json').read_text())
    records=d['records']
    dst=OUT/'evidence';dst.mkdir(exist_ok=True)
    fields=['id','name','evidence_type','hypothesis','changed','fixed','data_scope','baseline',
            'main_metrics','gained_correct','lost_correct','C_recovered','target_feature_results',
            'whole_dataset_results','result_assessment','supported_conclusions',
            'unsupported_conclusions','evidence_paths','implementation_paths']
    with (dst/'early_experiment_ledger.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k,'NOT_RECORDED') for k in fields})
    rows=[]
    for p in sorted((ROOT/'analysis').glob('cause_arch_*/VERDICT.md')):
        n=p.parent.name.split('_')[2]
        if not n.isdigit() or not 91<=int(n)<=117: continue
        text=p.read_text()
        if '_e' not in p.parent.name and p.parent.name not in ['cause_arch_092','cause_arch_112_scale','cause_arch_114_scale']:
            continue
        labels=['NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE','VERIFIED_INTERVENTION_EFFECT','METHOD_ONLY_NOT_VERIFIED_CAUSE',
                'TECHNICAL_FAILURE','NOT_RUN_MISSING_CACHE','REJECT','SUPPORTED']
        # Historical failure words may occur below a newer current verdict.
        header='\n'.join(text.splitlines()[:8])
        tag=next((x for x in labels if re.search(r'(?<![A-Z_])'+x+r'(?![A-Z_])',header)),'SEE_SOURCE')
        sub=re.search(r'子因[：:]\s*\*?\*?([^\n*]+)',text)
        rows.append({'experiment_dir':p.parent.name,'label_reported':tag,'subreason':sub.group(1).strip() if sub else '',
                     'first_1500_chars':text[:1500], 'source':'sources/'+str(p.relative_to(ROOT)),
                     'audit_level':'read_current_verdict_not_full_independent_reproduction',
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    with (dst/'later_verdict_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (dst/'later_verdict_index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return len(records),len(rows)


def paper_starter():
    paper=OUT/'paper_starter'; paper.mkdir(exist_ok=True)
    for n in ['cvpr.sty','ieeenat_fullname.bst']:
        shutil.copyfile(OUT/'official_cvpr2027'/n,paper/n)
    write('paper_starter/main.tex',r'''
% Writing scaffold, not a completed or validated method paper.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\usepackage{times,graphicx,amsmath,amssymb,booktabs,xcolor}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\def\confName{CVPR}
\def\confYear{2027}
\def\paperID{00000}
\newcommand{\Missing}[1]{\textcolor{red}{\textbf{[#1]}}}
\newcommand{\NR}{\Missing{NOT RUN}}
\title{Brief Relation Events within Persistent Entity Pairs:\\A Research and Implementation Draft}
\author{Anonymous CVPR submission}
\begin{document}
\maketitle
\input{sections/00_abstract}
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_problem}
\input{sections/04_method}
\input{sections/05_experiments}
\input{sections/06_limitations}
{\small\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\section*{LLM Usage Statement}
Language models assisted with organization, drafting, and preparation of this
research scaffold. Authors must update this statement to reflect the actual
design, implementation, writing, and verification performed for the completed paper.
\end{document}
''')
    write('paper_starter/sections/00_abstract.tex',r'''
\begin{abstract}
Video relation detection associates entity pairs with predicates and temporal
intervals. Persistent entity tracks provide useful context, yet a brief
relationship may occupy only a small part of their co-presence. We study the
localization of such occurrences while preserving enduring relations and a fixed
prediction budget. Existing project diagnostics identify a substantial residual
interval-formation deficit, including failures after correct entity inputs are
provided. They do not identify a unique causal mechanism.
\Missing{METHOD: replace with the implemented and novelty-audited mechanism.}
\Missing{RESULTS: insert only after the matched evaluations are completed.}
\end{abstract}
''')
    write('paper_starter/sections/01_introduction.tex',r'''
\section{Introduction}
Video visual relation detection describes who interacts with whom, through which
predicate, and during which interval. Entity co-presence and relation occurrence
are distinct temporal quantities: two objects can remain visible while a
particular interaction appears only briefly. A useful detector must recover that
brief occurrence without fragmenting relationships that genuinely persist.

VrdONE jointly predicts predicates and instance masks, while BIG separates
classification from multi-instance temporal grounding~\cite{vrdone,big}.
SegPVSG explicitly localizes salient segments before recognition~\cite{segpvsg}.
These approaches establish temporal modeling as an existing capability, making a
generic claim of ``adding temporal localization'' insufficient.

Our starting observation comes from a fixed VidOR comparison. At Top100, both
BIG and VrdONE miss 25,373 of 30,142 annotated instances. In the subsecond group,
3,151 of 3,167 instances are jointly missed. This paired comparison describes
released systems under their existing inputs; it is not an architecture-only
controlled comparison. Subsequent fixed-entity diagnostics narrow the problem:
on a selected 400-instance persistent-pair short-event cohort, native hull
intervals cover 47 instances, while a fixed single-component rule covers 53.
Neither result is official predicate-aware relation detection.

We ask how to recover supported brief occurrences under an unchanged prediction
budget while retaining enduring relations. The central design must specify why
its selected boundaries follow predicate evidence, and its evaluation must
distinguish localization improvements from extra candidates or changed scores.
\Missing{Finalize contributions after method implementation and evaluation.}
''')
    write('paper_starter/sections/02_related_work.tex',r'''
\section{Related Work}
\paragraph{Video relation modeling.}
BIG formulates classification and temporal grounding through a bipartite graph,
and VrdONE predicts relation instances through one-dimensional instance
segmentation~\cite{big,vrdone}. SegPVSG addresses temporally sparse interactions
through temporal focusing and relation-centric video augmentation~\cite{segpvsg}.
Our comparison must therefore isolate a more specific contribution than sparse
event localization alone.

\paragraph{Temporal event localization.}
VSGN directly targets short actions using temporal magnification and cross-scale
graphs~\cite{vsgn}. ActionFormer and TriDet offer strong multi-scale temporal
detection and boundary modeling designs~\cite{actionformer,tridet}.
BAM-DETR uses boundary-oriented decoding, while DenoiseLoc learns interval
refinement through denoising~\cite{bam,denoiseloc}. These methods motivate explicit
controls for boundary prediction, refinement, and localization quality.

\paragraph{Conditional span prediction.}
VSLNet uses query-guided highlighting and span prediction~\cite{vslnet}.
Consequently, a conditioned start/end head alone is a close-precedent baseline.
\Missing{Compare the final core computation against the closest verified methods.}
''')
    write('paper_starter/sections/03_problem.tex',r'''
\section{Problem and Empirical Basis}
For an ordered entity pair, let $C=[0,T)$ denote a continuous co-presence interval,
and $X=\{x_t\}$ its observed features. A relation event has predicate $p$ and
interval $I^*=[s^*,e^*)$. Its temporal occupancy is
\begin{equation}
\pi=\frac{e^*-s^*}{T}.
\end{equation}
The detector predicts a set of predicate-labeled intervals, allowing simultaneous
predicates and multiple separated occurrences of one predicate. Inference does
not receive GT predicates, durations, or event boundaries.

For temporal diagnosis, $\operatorname{tIoU}(I,I^*)=|I\cap I^*|/|I\cup I^*|$.
If a prediction contains the entire event but adds duration $\Delta$, then
\begin{equation}
\operatorname{tIoU}=\frac{L}{L+\Delta},\qquad L=|I^*|.
\end{equation}
This establishes sensitivity to absolute boundary error for short events; it
does not establish why a particular model makes that error. Official relation
detection additionally requires the correct triplet and both entity tube matches.

\begin{table}[t]
\centering\small
\caption{Historical paired Top100 outcomes on 30,142 VidOR GT instances. These are
observations of two existing systems, not matched-input architecture ablations.}
\begin{tabular}{lr}
\toprule Outcome & Count / total \\
\midrule Both detected & 2,487 / 30,142 \\
VrdONE only & 1,809 / 30,142 \\
Neither detected & 25,373 / 30,142 \\
BIG only & 473 / 30,142 \\
\bottomrule
\end{tabular}
\end{table}

The fixed 400-instance diagnosis uses already-exposed videos and correct-entity
inputs. Its 47-to-53 interval coverage change is a limited readout result, not an
end-to-end method gain. The independent-query upper bounds and oracle choices
must remain separate from deployable inference.
''')
    write('paper_starter/sections/04_method.tex',r'''
\section{Method}
\Missing{DESIGN ONLY: this section must be replaced by the final implemented method.}

The current candidate design question is how a predicate-conditioned interval
can explain the observed evidence without including unsupported co-presence or
discarding valid event evidence. One abstract formulation is
\begin{equation}
S(p,I\mid X)=S_{\mathrm{support}}-\lambda S_{\mathrm{extension}}
-\mu S_{\mathrm{omission}}.
\end{equation}
This equation is a design requirement, not an implemented algorithm or a novelty
claim. The completed method must define all three terms, their normalization,
training targets, inference computation, and treatment of repeated occurrences.

\paragraph{Required architecture specification.}
\Missing{Define tensor shapes, predicate conditioning, query/event assignment,
legal coordinates, interval selection, confidence, and fixed-budget inference.}

\paragraph{Training objective.}
\Missing{Write only losses used in code. Specify native initialization, frozen
parameters, matching, unknown-label handling, and train-only statistics.}

\paragraph{Ordinary boundary control.}
A start/end baseline can select legal endpoints using
\begin{equation}
\arg\max_{i<j}\left[\log p_s(i)+\log p_e(j)+
\lambda_L\log p(L=j-i\mid p)\right].
\end{equation}
Conditioned boundaries and duration priors have close precedent. Their inclusion
does not itself establish the main contribution, and true long events must retain
support throughout the duration prior.
''')
    write('paper_starter/sections/05_experiments.tex',r'''
\section{Experiments}
\paragraph{Data and protocols.}
VidOR is the primary benchmark. Official train, development, and evaluation
video IDs must be sealed before new experiments. The previously examined 27- and
250-video diagnostic subsets are not independent held-out tests. ImageNet-VidVRD
can provide an additional benchmark, with its different annotation granularity
reported explicitly. A PVSG comparison requires task adaptation and a VidOR
source-video overlap audit.

\paragraph{Comparators.}
Compare the unchanged checkpoint, same-budget native continued training,
an ordinary boundary control, a clearly documented temporal-focusing adaptation,
and the proposed method. Use the same inputs, budget, evaluation implementation,
and matched seeds. A task-adapted control is not automatically an author-method
reproduction.

\begin{table}[t]
\centering\small
\caption{Planned official detection evaluation. New evaluations are not run.
Historical diagnostic numbers are deliberately not substituted here.}
\begin{tabular}{lccc}
\toprule Method & mAP & R@50 & R@100 \\
\midrule Frozen base & \NR & \NR & \NR \\
Native FT & \NR & \NR & \NR \\
Boundary & \NR & \NR & \NR \\
TFN-style & \NR & \NR & \NR \\
Proposed & \NR & \NR & \NR \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Event slices and gains/losses.}
Report $<1$, $1$--$3$, $3$--$10$, and $\geq10$ second groups, continuous
co-presence, interval occupancy, and both predicate-aware and time-only results.
Every recovery analysis includes newly detected instances, lost baseline
instances, fixed-original-C recovery, output counts, and long-event protection.
Ratios include their actual numerator and denominator; AP is not a TP/GT ratio.

\paragraph{Mechanism and ablations.}
\Missing{Pre-register the final core mechanism comparator and necessary ablations.
Do not repeat failed recipes under new names or choose validation thresholds
after seeing their results.}

\paragraph{Reproducibility.}
\Missing{Add actual environments, weights, splits, trainable parameters, seeds,
runtime, uncertainty, prediction seals, and code commit after execution.}
''')
    write('paper_starter/sections/06_limitations.tex',r'''
\section{Limitations and Conclusion}
The existing project evidence localizes a persistent short-interval deficit but
does not identify a unique causal explanation. Some event boundaries, especially
gaze relations, are also visually ambiguous. Frame-rate and annotation differences
limit direct cross-dataset comparisons.

\Missing{After implementation and evaluation, conclude only the supported method
and mechanism findings. Do not claim improved performance or a complete causal
account from this scaffold.}
''')
    write('paper_starter/references.bib',r'''
@misc{vrdone,
 author={Xinjie Jiang and Chenxi Zheng and Xuemiao Xu and Bangzhen Liu and Weiying Zheng and Huaidong Zhang and Shengfeng He},
 title={VrdONE: One-stage Video Visual Relation Detection},year={2024},howpublished={arXiv:2408.09408},url={https://arxiv.org/abs/2408.09408}}
@inproceedings{big,
 author={Kaifeng Gao and Long Chen and Yulei Niu and Jian Shao and Jun Xiao},
 title={Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs},
 booktitle={CVPR},year={2022},url={https://arxiv.org/abs/2112.04222}}
@inproceedings{segpvsg,
 author={Yikai Li and Quhui Ke and Jinglin Liang and Zhi-Yuan Zhang and Zhidi Lin and Shuangping Huang},
 title={SegPVSG: Panoptic Video Scene Graph Generation via Temporal Focusing and Generative Augmentation},
 booktitle={ICML},series={PMLR},volume={306},pages={67742--67759},year={2026},url={https://proceedings.mlr.press/v306/li26bx.html}}
@inproceedings{vsgn,
 author={Chen Zhao and Ali Thabet and Bernard Ghanem},title={Video Self-Stitching Graph Network for Temporal Action Localization},
 booktitle={ICCV},year={2021},url={https://arxiv.org/abs/2011.14598}}
@inproceedings{actionformer,
 author={Chenlin Zhang and Jianxin Wu and Yin Li},title={ActionFormer: Localizing Moments of Actions with Transformers},
 booktitle={ECCV},year={2022},url={https://arxiv.org/abs/2202.07925}}
@inproceedings{tridet,
 author={Dingfeng Shi and Yujie Zhong and Qiong Cao and Lin Ma and Jia Li and Dacheng Tao},
 title={TriDet: Temporal Action Detection with Relative Boundary Modeling},booktitle={CVPR},year={2023},url={https://arxiv.org/abs/2303.07347}}
@inproceedings{bam,
 author={Pilhyeon Lee and Hyeran Byun},title={BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos},
 booktitle={ECCV},year={2024},url={https://arxiv.org/abs/2312.00083}}
@misc{denoiseloc,
 author={Mengmeng Xu and Mattia Soldan and Jialin Gao and Shuming Liu and Juan-Manuel P{\'e}rez-R{\'u}a and Bernard Ghanem},
 title={Boundary-Denoising for Video Activity Localization},year={2023},howpublished={arXiv:2304.02934},url={https://arxiv.org/abs/2304.02934}}
@inproceedings{vslnet,
 author={Hao Zhang and Aixin Sun and Wei Jing and Joey Tianyi Zhou},
 title={Span-based Localizing Network for Natural Language Video Localization},
 booktitle={ACL},year={2020},url={https://aclanthology.org/2020.acl-main.585/}}
''')
    write('paper_starter/README.md',r'''
# CVPR 2027 writing scaffold

This is a compilable scaffold, not a completed method paper or experimental result.
Official style and bibliography format are unchanged. New results are explicitly NOT RUN.

Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

Claude must complete the method with the delivered implementation, replace the
placeholders after real evaluation, audit citations/claims, and organize the final
paper within the official page limit. Do not silently delete Missing markers to
simulate a completed experiment.
''')


def main():
    docs={'00_CLAUDE_PROMPT.md':PROMPT,'01_RESEARCH_CONTEXT.md':CONTEXT,
          '02_EVIDENCE_AND_FAILURES.md':EVIDENCE,'03_MATH_AND_DESIGN_SPACE.md':MATH,
          '04_LITERATURE_AND_NOVELTY.md':LITERATURE,'05_CODE_AND_DATA_MAP.md':CODEMAP,
          '06_EXPERIMENT_AND_DELIVERY_CONTRACT.md':CONTRACT,
          '07_CONFLICTS_AND_MISSING_MATERIALS.md':CONFLICTS,'08_PAPER_WRITING_BRIEF.md':WRITING,
          'CACHE_AND_RESOURCE_CHECKLIST.md':CACHE}
    for name,txt in docs.items():write(name,txt)
    print('Written core documents',flush=True)
    collect_sources()
    print('Source files queued:',len(COPY_REQUESTS),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(perform_copy,*args):args for args in COPY_REQUESTS}
        for n,job in enumerate(as_completed(jobs),1):
            try:
                job.result()
            except Exception as exc:
                MISSING.append({'source':str(jobs[job][0]),'reason':str(exc)})
            if n%100==0: print('Source files processed:',n,flush=True)
    MANIFEST.sort(key=lambda x:x['package_path'])
    MISSING.sort(key=lambda x:x['source'])
    a,b=generate_indexes()
    paper_starter()
    write('user_materials/CHAT_ONLY_CLAIMS_NOT_REVERIFIED.md',r'''
# 用户聊天补充，原始统计文件尚未核到

用户报告整共存窗口基线：
- 关系≤2s、共存≤3s：610/731，83.45%；共存≥10s：0/5539。
- 关系≤3s、共存≤3s：1211/1332，90.92%；共存≥10s：0/8724。
- 用户报告State存在30k+1帧网格，Activity/Event较逐帧精细；约32%/40%/15%需绑定切片/完整分母。

数学上若共存≥10且GT≤3、GT完全在共存内，则整窗口tIoU≤0.3，0通过是定义可推出的结果，不等于新的模型失败证据。上述比例可作寻找文件线索，不能在没拿到CSV/代码前升级核验。

历史只读记忆曾指出ER/ORCA没有通过收益门且有候选减少；本轮不重新远程读取，不把记忆数字作为论文证据。
''')
    write('historical_unreviewed/README.md','''
# Historical manuscript warning

These files are historical writing material. Their claims, result protocols,
and novelty statements have not been adopted for the new method. Consult the
evidence ledger and original reports before using any sentence or number.
''')
    write('README_START_HERE.md',f'''
# Claude研究交接包

日期：2026-11-02。用途：让Claude在同一份资料上完成方案、代码与CVPR论文初稿。

## 最简单用法

1. Claude支持项目/目录读取：解压整个包，粘贴 `00_CLAUDE_PROMPT.md`。
2. Claude只能聊天上传：先上传 `CLAUDE_SINGLE_MESSAGE.md`，再上传压缩包（若不支持解压，上传01–08及所需源文件）。
3. 避免一次把所有源码塞进对话上下文；先读核心文档，再按文件路径读取sources。

## 内容

- 完整提示词、问题背景、数学定义、近邻文献、代码入口、交付合同、冲突清单。
- 早期{a}条实验总账；后续{b}份判决索引（含历史/设计状态记录，按目录区分，不是{b}个独立成功实验）。
- 选定原报告、源码、配置、逐GT表、用户TICK/ORCA/因果实验原文。
- 真实VidOR Figure2及编辑SVG，普通边界方案参考图。
- 官方CVPR2027模板、带真实观察与显式NOT RUN的新论文写作骨架。

## 本次状态

只有资料整理/文件复制/模板下载/LaTeX与包验收。没有新模型实现、科研训练、GPU、SSH或新性能数字。论文骨架不等于论文完成；Claude后续仍需交付完整方法实现并用真实实验替换占位符。

本包不附训练缓存、checkpoint与全量视频；缺项与权限见07和CACHE清单。模拟数字不能进入真实论文结果。
''')
    write('CLAUDE_SINGLE_MESSAGE.md','\n\n---\n\n'.join([PROMPT,CONTEXT,EVIDENCE,MATH,LITERATURE,CODEMAP,CONTRACT,CONFLICTS,WRITING,CACHE]))
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'local packaging only; no research execution','files':MANIFEST,'missing_or_excluded':MISSING},ensure_ascii=False,indent=2))
    print(json.dumps({'source_files':len(MANIFEST),'missing_or_excluded':len(MISSING),'early_records':a,'later_verdicts':b},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
