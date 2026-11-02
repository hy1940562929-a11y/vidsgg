#!/usr/bin/env python3
"""Reproduce compact Chinese report from imported per-GT experiment records."""
import csv, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMP=ROOT/'handoff/incoming/VIDOR-GT-MASK-PROBE-20260929'
rows=list(csv.DictReader((IMP/'evaluation/per_gt.csv').open()))
summary=json.loads((IMP/'evaluation/summary.json').read_text())
entities={(r['video_name'],r['gt_index']):r for r in csv.DictReader((ROOT/'analysis/entity_relation_duration_20260929/per_gt_durations.csv').open(encoding='utf-8-sig'))}
assert len(rows)==summary['N_GT']==2573
def count(rs,k):return sum(r[k]=='1' for r in rs)
def pct(n,d):return f'{100*n/d:.2f}%（{n:,}/{d:,}）' if d else f'NA（{n}/0）'
def counts(rs):
    old=count(rs,'native_viou50_min5');new=count(rs,'gt_viou50_min5')
    gains=sum(r['transition_min5']=='new' for r in rs);losses=sum(r['transition_min5']=='lost' for r in rs)
    assert new-old==gains-losses
    return dict(N=len(rs),native_viou50=old,gt_viou50=new,gains=gains,losses=losses,net=new-old,
        gt_binary50=count(rs,'gt_binary50_unfiltered'),gt_tiou50=count(rs,'gt_tiou50_min5'),
        grid_hull_tiou50_feasible=count(rs,'grid_hull_tiou50_feasible_min5'))

duration={k:counts([r for r in rows if r['duration_bin']==k]) for k in ['<1s','1-3s','3-10s','>=10s']}
short=[r for r in rows if float(r['duration_sec'])<3]
entity_short={k:counts([r for r in short if entities[r['video_name'],r['gt_index']]['entity_local_bucket']==k]) for k in ['both_long','one_short','both_short']}
derived=dict(overall=counts(rows),short_lt3=counts(short),duration=duration,entity_short=entity_short,
    C=counts([r for r in rows if r['group_at100']=='C']),
    short_joint_atleast3=counts([r for r in short if float(entities[r['video_name'],r['gt_index']]['joint_local_continuous_sec'])>=3]),
    short_joint_atleast10=counts([r for r in short if float(entities[r['video_name'],r['gt_index']]['joint_local_continuous_sec'])>=10]),
    grid_feasible_but_tiou_failed=sum(r['grid_hull_tiou50_feasible_min5']=='1' and r['gt_tiou50_min5']=='0' for r in rows))
(HERE/'derived_summary.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n')
N=len(rows);o=derived['overall'];c=derived['C'];S=len(short)
lines=['# GT实体输入后，VrdONE还能否生成正确时间掩码？','',
    '**真实实验已完成。** AutoDL上使用原VrdONE EMA checkpoint，不训练；native与GT两臂均在同一进程重新推理。27段已有GT视觉特征的视频，共2,573条GT关系。','',
    f'这些视频占835个验证视频的{pct(27,835)}；关系占全部30,142条的{pct(N,30142)}。视频来自既往strict参与实体切换筛选，非随机、非独立测试集；不能将下面的增幅外推到完整数据集。','',
    '## 1. 纠正实体输入后，多了多少可匹配掩码？','',
    '统一口径：忽略谓词分类与TopK，在每对实体的9条原始query中寻找候选；按原规则解码成起止区间，至少5帧，主体和客体管道vIoU都≥0.5。GT侧必须是正确身份pair。该数字是候选存在性，允许GT挑选query，不是自动选出的检测结果。','',
    '| 条件 | 可匹配掩码覆盖的GT关系 |','| --- | ---: |',
    f'| 原检测/跟踪输入 | {pct(o["native_viou50"],N)} |',
    f'| GT框、GT轨迹、正确主体/客体配对输入 | {pct(o["gt_viou50"],N)} |',
    f'| 原来没有、GT输入后新增 | {pct(o["gains"],N)} |',
    f'| 原来有、GT输入后丢失 | {pct(o["losses"],N)} |',
    f'| 净增加 | **{o["net"]:,}条，+{100*o["net"]/N:.2f}个百分点（{o["net"]:,}/{N:,}×100）** |','',
    '输入纠正同时改变了框几何、轨迹支持时间及视觉/运动特征，因此净增反映上游输入纠正后的综合收益，不能解释成mask head单独提高了相同幅度。原输入用类别相容+几何匹配，GT输入进一步要求真身份；两侧均不评价谓词是否正确。','',
    '## 2. 只看GT输入后的时间掩码能力','',
    '| 指标 | 达标GT关系 |','| --- | ---: |',
    f'| 二值时间掩码IoU≥0.5 | **{pct(o["gt_binary50"],N)}** |',
    f'| 解码起止区间tIoU≥0.5 | {pct(o["gt_tiou50"],N)} |',
    f'| 同采样网格存在理论上可达到tIoU≥0.5的合法区间 | {pct(o["grid_hull_tiou50_feasible"],N)} |','',
    f'仍有{N-o["gt_binary50"]:,}条未找到二值掩码达标候选，比例为{pct(N-o["gt_binary50"],N)}。理论网格可表达、实际解码却未达到tIoU阈值的关系有832条，占{pct(832,N)}。所以在这批GT输入中，单纯“采样网格完全无法表示正确区间”不足以解释大部分失败；并不排除稀疏采样损失视觉信息、模型学习不足或特征分布变化。','',
    '二值掩码口径：每4帧一个采样点，正采样点保持至下一点，末尾截在完整实体共存段结束；计算整个视频时间轴上的交并比，掩码在GT之外的激活仍罚分。解码区间口径：首末正采样点包围区间，末端+1，可能填平中间空洞。这两个指标不能混为同一个“正确率”。','',
    '## 3. 短关系是否仍难？','',
    '| 关系时长 | GT关系数 | GT输入：二值mask IoU≥0.5 | GT输入：区间tIoU≥0.5 |','| --- | ---: | ---: | ---: |']
for label,d in duration.items():lines.append(f'| {label} | {d["N"]:,} | {pct(d["gt_binary50"],d["N"])} | {pct(d["gt_tiou50"],d["N"])} |')
lines+=['',f'不足3秒合计：二值mask达标{pct(derived["short_lt3"]["gt_binary50"],S)}。GT输入已排除了检测/跟踪/配对错误，但短关系仍有较多时间候选缺失。这是条件诊断，尚未控制谓词、场景、交互复杂度等群体差异。','',
    '## 4. 关系短，还是实体也短？','',
    '限定关系不足3秒。实体时长采用包含当前关系的GT连续标注段，不把多个分散片段累加；与上一轮实体/关系时长分析按(video,gt_index)精确关联。','',
    '| 当前连续实体时长 | 在966条短关系中的分布 | GT输入二值mask达标 |','| --- | ---: | ---: |']
for key,label in [('both_long','两个实体都至少3秒'),('one_short','恰有一个实体不足3秒'),('both_short','两个实体都不足3秒')]:
    d=entity_short[key];lines.append(f'| {label} | {pct(d["N"],S)} | {pct(d["gt_binary50"],d["N"])} |')
lines+=['', '**值得继续追查的画像：实体持续存在，但关系只短暂发生。** 这不是已经证明“长上下文稀释”或“旧关系污染”；群体语义构成、完整实体共存时长与目标区间占比都有待控制。两实体均短的组只有21条、来自5个视频，不能据此宣称该组已被普遍解决。GT连续标注支持也不等于人工核实的物理可见时长。', '',
    f'进一步区分：两实体各自连续至少3秒，不保证共同连续至少3秒；662条中有{pct(36,662)}的共同连续段仍短于3秒。更严格限定“关系<3秒、共同连续段≥3秒”后，有626条，二值mask达标{pct(178,626)}，区间tIoU达标{pct(174,626)}；限定共同段≥10秒后，有466条，二值mask达标{pct(66,466)}，区间tIoU达标{pct(61,466)}。这些是事后描述分层，尚未控制谓词或场景。','',
    '## 5. 原先BIG与VrdONE共同失败的C类','',
    f'C类共{c["N"]:,}条。原输入已经存在可匹配时间/空间mask的有{pct(c["native_viou50"],c["N"])}；完整模型仍失败可能涉及谓词/排序等，因此不能把全部C类归为时间mask没有生成。GT输入下达到{pct(c["gt_viou50"],c["N"])}，新增{c["gains"]:,}、丢失{c["losses"]:,}、净增{c["net"]:,}。以上仍是raw候选覆盖。','',
    '## 6. 避免同一掩码重复计数','',
    '将仅谓词不同、主体/客体与时间完全相同的关系合并后，有2,512个独立时间事件。对GT侧全部query做一对一最大匹配：','',
    f'- 二值mask IoU≥0.5：{pct(1644,2512)}。',
    f'- 解码区间tIoU≥0.5：{pct(1658,2512)}。','',
    '最大匹配仍用GT选择query，是时间输出容量诊断，不是模型自动选择正确，更不是两臂一对一净增。','',
    '## 7. 实验设置与可追溯性','',
    '- 原EMA checkpoint SHA256：`4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627`。',
    '- GT使用关系支持的正确有向身份pair集合；不给模型谓词或关系起止，枚举所有完整连续共存段，没有裁成GT关系窗口。',
    '- 168,076个实体帧的框、tid、逐帧1024维特征对齐核验通过。原预测轨迹去重会减少114个GT共存段输入，本轮在GT条件关闭该去重；仍保留原最短输入规则。',
    '- 原配置、采样stride4、offset0、9query、sigmoid阈值0.5均固定。完整forward重跑耗时约67秒；无训练、无参数更新。',
    '- 全部3,913,587个mask logit的NumPy评估与原CUDA sigmoid阈值判断一致，不一致数为0。',
    '- 旧缓存复用核验在第26个视频因浮点差异未通过，失败日志保留；主结果来自两臂全部重新推理的`paired27_fresh_v2`。',
    '- `preflight.native_raw_source`列出当时发现的历史缓存，只是可用来源记录；主run的`completion.native_fresh=true`及每视频`native_forward_seconds`证明该轮实际使用fresh forward。',
    '- GT ROI提取代码和27视频完成日志可查，但旧提取日志没有命令及当时权重哈希，历史环境无法完全追溯。',
    '- 本结论不外推全部验证集，不报告新方法mAP/R@100，不把GT输入或GT选择query作为可部署方法。','',
    '## 8. 下一步建议','',
    '固定这批GT框与身份输入，优先在“长实体共存段中的短关系”上区分：特征采样丢失、前向特征融合使局部证据变弱、mask query没有覆盖目标区间、以及解码首末点过长。先比较失败与成功的同谓词、相近实体共存时长样本，再做单一环节干预；同时报告全体2,573条的新增、丢失及净变化。当前GT网格上界已高，不能仅凭短关系画像直接断言提高采样率就是解法。','',
    '## 原始材料','',
    f'- [逐GT评价表]({IMP}/evaluation/per_gt.csv)',
    f'- [独立时间事件与一对一匹配]({IMP}/evaluation/unique_time_events.csv)',
    f'- [完整指标、配置和来源哈希]({IMP}/evaluation/summary.json)',
    f'- [输入与推理记录]({IMP}/run/completion.json)',
    f'- [阈值解码核验]({IMP}/run/decoding_consistency.json)',
    f'- [实验协议]({HERE}/PROTOCOL.md)',
    f'- [独立审查]({ROOT}/reviews/gt-mask-probe-20260929-review.md)', '']
(HERE/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(derived,ensure_ascii=False,indent=2))
