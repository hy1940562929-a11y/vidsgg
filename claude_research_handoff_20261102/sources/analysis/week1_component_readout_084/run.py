#!/usr/bin/env python3
"""Paired, fixed-output comparison of two temporal interval families."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT/'handoff/incoming/VIDOR-TEMPORAL-SUPPORT-015/component_evidence_v1.tar.gz'
MEMBER = 'components_v1/per_gt.csv'
EXPECTED = '76ebe09c330f9dbb6deffd69a26c74b96258222f336ccb2eb7be5327face96ce'
PREVIOUS = ROOT/'analysis/week1_query_feasibility_083/results_v1/per_gt.csv'
PAIRS = {'fixed_threshold_0p5': ('default_hull', 'default_component'),
         'raw_logit_all_threshold_oracle': ('raw_hull', 'raw_component')}


def sha(data): return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def records(data):
    result = {}
    for r in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
        k = r['video_name'], int(r['gt_index'])
        require(k not in result, 'Duplicate GT')
        result[k] = r
    return result


def read_sources():
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        matches = [m for m in archive.getmembers() if m.name == MEMBER]
        require(len(matches)==1 and matches[0].isfile(), 'Bad archive member')
        body = archive.extractfile(matches[0]).read()
    require(sha(body)==EXPECTED, '015B source changed')
    prior = PREVIOUS.read_bytes()
    previous_manifest = json.loads(PREVIOUS.with_name('manifest.json').read_text())
    require(previous_manifest['complete'] is True and sha(prior)==previous_manifest['output_sha256']['per_gt.csv'], '083 receipt mismatch')
    a,b = records(body),records(prior)
    require(set(a)==set(b) and len(a)==2573, '083/015B cohort mismatch')
    result=[]
    for key, x in sorted(a.items()):
        y=b[key]
        for f in ('subject_tid','object_tid','begin_fid','end_fid'):
            require(int(x[f])==int(y[f]), 'GT identity/time mismatch')
        require(x['predicate']==y['predicate'] and x['group_at100']==y['original_group'], 'GT label mismatch')
        require(abs(float(x['duration_sec'])-float(y['relation_sec']))<1e-7,'Duration mismatch')
        for f,g in [('default_hull_tiou','default_min5'),('raw_logit_full_hull_oracle_tiou','raw_logit'),('grid_hull_upper_min5','grid_min5')]:
            require(float(x[f])==float(y[g]), '015B/083 hull or grid disagreement')
        r={f:y[f] for f in ('video_name','gt_index','subject_tid','object_tid','predicate','begin_fid','end_fid','relation_sec','duration_group')}
        require(y['chosen_pair_covers_gt'].lower() in ('true','false'), 'Invalid covers flag')
        r['chosen_pair_covers_gt']=y['chosen_pair_covers_gt'].lower()=='true'
        r['posthoc_150_focus']=(r['chosen_pair_covers_gt'] and r['duration_group']=='<1s' and
            y['stage_tiou05']=='grid_feasible_threshold_family_infeasible')
        for f,g in [('default_hull','default_hull_tiou'),('default_component','default_component_tiou'),
                    ('raw_hull','raw_logit_full_hull_oracle_tiou'),('raw_component','raw_logit_component_oracle_tiou')]:
            value=float(x[g]);require(math.isfinite(value) and 0<=value<=1, 'Invalid tIoU')
            r[f]=value
        for f in ('default_hull_candidates','default_component_candidates','oracle_component_intervals'):
            r[f]=int(x[f]);require(r[f]>=0,'Negative candidate count')
        result.append(r)
    require(sum(r['posthoc_150_focus'] for r in result)==150,'Changed posthoc 083 cohort')
    return result,dict(components=dict(archive=str(ARCHIVE), member=MEMBER, sha256=sha(body)),
                       previous=dict(path=str(PREVIOUS),sha256=sha(prior)))


def compare(group, left, right, threshold):
    n=len(group)
    cells=dict(both_hit=0,hull_only_loss=0,component_only_gain=0,neither=0)
    for r in group:
        a,b=r[left]>=threshold,r[right]>=threshold
        name='both_hit' if a and b else 'hull_only_loss' if a else 'component_only_gain' if b else 'neither'
        cells[name]+=1
    out=dict(N=n,N_videos=len({r['video_name'] for r in group}),**cells)
    require(sum(cells.values())==n,'Paired partition does not close')
    out['net_gain']=out['component_only_gain']-out['hull_only_loss']
    out['hull_hits']=out['both_hit']+out['hull_only_loss']
    out['component_hits']=out['both_hit']+out['component_only_gain']
    for f in (*cells, 'net_gain', 'hull_hits', 'component_hits'):
        out[f+'_denominator']=n
        out[f+'_fraction']=out[f]/n if n else None
    out['hull_tiou_sum']=math.fsum(r[left] for r in group)
    out['component_tiou_sum']=math.fsum(r[right] for r in group)
    out['mean_tiou_denominator']=n
    out['hull_mean_tiou']=out['hull_tiou_sum']/n if n else None
    out['component_mean_tiou']=out['component_tiou_sum']/n if n else None
    out['mean_tiou_gap']=((out['component_tiou_sum']-out['hull_tiou_sum'])/n) if n else None
    return out


def summarize(allrows):
    result=[]
    for population in ('all','chosen_pair_covers_gt','posthoc_150_focus'):
        pop=[r for r in allrows if population=='all' or r[population]]
        bins=('all',) if population=='posthoc_150_focus' else ('all','<1s','1-3s','>=3s')
        for duration in bins:
            group=[r for r in pop if duration=='all' or r['duration_group']==duration]
            for comparison,(left,right) in PAIRS.items():
                for threshold in (.3,.5,.7):
                    result.append(dict(population=population,duration_group=duration,comparison=comparison,
                        tiou_threshold=threshold,**compare(group,left,right,threshold)))
    return result


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:'NA' if v is None else v for k,v in r.items()} for r in rows)


def report(summary):
    lines=['# 084：保留分离时间段是否扩大可匹配的时间形状', '',
        '复用015B与083的同源输出，27视频/2573GT，无训练、无新推理。两侧都用GT时间挑最匹配形状，不要求正确谓词，不限分类选中query；因此下面不是部署收益。', '',
        '|范围|比较|N|hull命中|component命中|新增|损失|净差|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['tiou_threshold']!=.5 or (r['duration_group'] not in ('all','<1s')):continue
        label=r['population']+' / '+r['duration_group'];n=r['N']
        comp='固定0.5阈值' if r['comparison']=='fixed_threshold_0p5' else '全部raw-logit阈值上界'
        lines.append(f"|{label}|{comp}|{n}|{r['hull_hits']}/{n}|{r['component_hits']}/{n}|{r['component_only_gain']}/{n}|{r['hull_only_loss']}/{n}|{r['net_gain']:+d}/{n}|")
    lines += ['', 'hull和component集合不是嵌套的，因此新增和损失同时保留；不只展示150失败组。全部人群和tIoU 0.3/0.5/0.7、连续tIoU在CSV。', '',
        '150组是在083看到失败后固定的亚秒焦点，不是独立测试集。放宽到全部阈值又使用GT挑query/分量，会增加候选并借用答案；不等于实际方法能选中正确时间段。', '',
        '若component上界更高，只能定位到现有分数下区间族的差异；不能据此宣布长关系压制、特征不足或训练方法已有效。已有正式split对照负结果继续保留。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(not args.out.exists(),'Refuse overwrite')
    frozen=json.loads((HERE/'protocol_freeze.json').read_text())
    require(sha((HERE/'PROTOCOL.md').read_bytes())==frozen['protocol_sha256'],'Protocol changed')
    rows,sources=read_sources();summary=summarize(rows)
    args.out.mkdir(parents=True)
    write_csv(args.out/'per_gt.csv',rows);write_csv(args.out/'summary.csv',summary)
    (args.out/'REPORT.md').write_text(report(summary))
    manifest=dict(complete=True,created_at=datetime.now(timezone.utc).isoformat(),N_GT=len(rows),N_videos=27,
                  source_records=sources,code_sha256=sha(Path(__file__).read_bytes()),
                  protocol_sha256=frozen['protocol_sha256'],training_steps=0,model_forwards=0,
                  design='post083_existing_output_diagnostic_oracle_comparison',
                  outputs={p.name:sha(p.read_bytes()) for p in args.out.iterdir()})
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete=True,N_GT=len(rows),summary_rows=len(summary),output=str(args.out))))


if __name__=='__main__': main()
