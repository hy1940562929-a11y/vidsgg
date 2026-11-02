#!/usr/bin/env python3
"""Render the one-page decision draft from saved, audited counts."""
import csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def rate(n,d):return f'{100*int(n)/int(d):.2f}%（{int(n):,}/{int(d):,}）' if int(d) else 'NA'
def main():
    diag=read(HERE/'results_v1/diag_summary.csv');off=read(HERE/'results_v1/official_detection_summary.csv')
    controls=read(HERE/'controls_v1/controlled_summary.csv')
    def get(h,out,pop='all',no='False',load='False'):
        return next(r for r in controls if r['hypothesis']==h and r['outcome']==out and r['population']==pop
                    and r['h2_no_comp_control']==no and r['load_control']==load and r['bootstrap_method']=='whole_video_with_replacement')
    def diff(r):
        numerator=float(r['weighted_numerator_side1'])-float(r['weighted_numerator_side0'])
        return (f"{float(r['standardized_gap_pp']):+.2f} pp（{numerator:+.4f}/{int(r['weighted_denominator']):,}×100；"
                f"95% CI {100*float(r['bootstrap_ci_low']):+.2f}～{100*float(r['bootstrap_ci_high']):+.2f} pp）")
    lines=['# Week1 短关系诊断：一页判决草稿','',
        '**状态：已完成全量切片与命中统计、GT27时间诊断；原生全量时间表尚缺输入。没有训练或重新推理。**','',
        '沿用077：短关系<3秒、长关系≥3秒、长共存为GT连续共同支持≥10秒。下表comp按用户字面定义：同一有序实体对另有≥3秒关系；solo指未标注此类关系。四格之外另列L-S。', '',
        '**完整val有30,142条GT；时间指标来自27视频/2,573条GT的正确实体输入诊断。** 主读出按目标谓词最高分类分数选query，不按GT时间挑选；不是正式Top100召回。', '',
        '|切片|全val N|GT27 N|时间漏检@0.5|长度≥2倍 / 有输出|平均长度比|平均tIoU|',
        '|---|---:|---:|---|---|---:|---:|']
    for name in ['S-S','S-L-solo','S-L-comp','L-L']:
        r=next(r for r in diag if r['slice_definition']=='literal_other_ge3_anywhere' and r['source_scope']=='GT27_correct_entity_input' and r['analysis_population']=='all' and r['slice']==name)
        f=next(r for r in off if r['slice_definition']=='literal_other_ge3_anywhere' and r['analysis_population']=='all' and r['slice']==name and r['model']=='VrdONE')
        lines.append(f"|{name}|{int(f['N']):,}|{int(r['N']):,}|{rate(r['miss_count_tiou05'],r['N'])}|{rate(r['stretch_count'],r['stretch_denominator'])}|{float(r['duration_bias']):.2f}|{float(r['mean_tiou']):.3f}|")
    lines+=['','L-S残余：全val 8,151、GT27 506；各表保留，分母闭合。tIoU 0.3/0.5/0.7、时长比和均值的分子分母均在CSV。原生全val缺失的时间指标记NA。','',
        '**必须区分的两件事。** 严格要求另一长关系“异谓词且时间重叠”后，comp为9.26%（2,790/30,142），solo为265条。GT27对应comp398条、solo68条/10视频，solo不足预设20视频。短组684次长度≥2倍中，65.50%（448/684）完全不接触目标GT，不能都叫边界外扩。','',
        '|假设|可观察结果|机制判决|','|---|---|---|',
        f"|H1 时长先验|控制谓词和共存档，GT27短关系时间漏检高{diff(get('H1','miss_tiou05'))}；此预言暂成立。|未决：固定区间复用与评价尺度也会产生差距。|",
        f"|H2 共存淹没|只比较无标注竞争者的短关系，控制谓词和0.25秒时长档，全val VrdONE漏检高{diff(get('H2','VrdONE_official_miss',no='True'))}；检测关联暂成立。|未决：GT27 solo支持不足，未做上下文干预。|",
        f"|H5 长短竞争|严格comp对solo，再控制谓词/时长/共存/视频负载，VrdONE为{diff(get('H5','VrdONE_official_miss',load='True'))}。未出现预期正向点差。|未决：重抽样支持不稳，且漏检接近天花板；不把幅度排除当机制证伪。|",
        '',
        '下一件：先区分固定读出“选错事件”和“查询库本身无可匹配区间”，同时准备从已有服务器文件补全原生时间诊断。073/075/078负结果保留；不增加训练模块。', '',
        '证据：[诊断总表](results_v1/diag_summary.csv) · [正式检测另表](results_v1/official_detection_summary.csv) · [受控比较](controls_v1/controlled_summary.csv) · [读出审计](readout_audit_v1/readout_audit.csv) · [范围与复现](README.md) · [判决有效性审计](POSTHOC_VALIDITY_AUDIT.md)']
    (HERE/'DECISION_ONEPAGE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
