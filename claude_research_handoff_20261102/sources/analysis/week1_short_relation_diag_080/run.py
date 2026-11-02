#!/usr/bin/env python3
"""080: recompute slice diagnostics from existing, fingerprinted artifacts.

No model execution. Raw native full-val temporal intervals cannot be inferred
from hit vectors; that scope deliberately has missing temporal measurements.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math
from pathlib import Path
from stats import build_summary, write_summary_csv, SLICES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    'durations': ROOT/'analysis/event_support_073/inputs/per_gt_durations.csv',
    'controls': ROOT/'handoff/incoming/VIDOR-RESEARCH-QUESTION-077/remote_v1/controls_v1/per_gt_controls.csv',
    'boundary': ROOT/'handoff/incoming/VIDOR-RESEARCH-QUESTION-077/remote_v1/boundary_v1/per_gt_boundary.csv',
}
EXPECTED = {'durations':'a54f973cd7704c06251ea18c0d84a22469b4412daec817b52d1fcccec285f35a',
            'boundary':'94b032eb1e317c0b4296da0a25e81e99842df1c835297d41d720165bec72f3eb'}
DEFINITIONS = ('literal_other_ge3_anywhere','overlap_other_pred_ge3',
               'overlap_other_pred_ge3_half','overlap_other_pred_ge10')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else 'NA' if v is None else v for k,v in row.items()})

def key(r):return r['video_name'],int(r['gt_index'])
def truth(x):return str(x).lower() in ('true','1')
def optional(x):return float(x) if x not in ('',None,'None','nan') else None

def same_optional_number(a,b,tolerance=1e-7):
    if a is None or b is None:return a is None and b is None
    return abs(a-b)<tolerance

def interval_iou(a,b,c,d):
    intersection=max(0,min(b,d)-max(a,c))
    return intersection/(b-a+d-c-intersection) if b>a and d>c else 0.

def classify(relation,coexist,comp):
    if coexist is None or not math.isfinite(coexist) or coexist<=0 or coexist+1e-6<relation:return 'UNKNOWN'
    if relation<3:
        return 'S-S' if coexist<10 else 'S-L-comp' if comp else 'S-L-solo'
    return 'L-S' if coexist<10 else 'L-L'

def companion_flags(r, rows):
    companions=[]
    for other in rows:
        if (other['video_name'],other['subject_tid'],other['object_tid']) != (r['video_name'],r['subject_tid'],r['object_tid']):continue
        if other['gt_index']==r['gt_index'] or other['relation_sec']<3:continue
        overlap=max(0,min(other['end_fid'],r['end_fid'])-max(other['begin_fid'],r['begin_fid']))
        companions.append(dict(gt_index=other['gt_index'],predicate=other['predicate'],
            relation_sec=other['relation_sec'],overlap_frames=overlap,
            coverage_of_target=overlap/(r['end_fid']-r['begin_fid'])))
    strict=[o for o in companions if o['predicate']!=r['predicate'] and o['overlap_frames']>0]
    flags=dict(literal_other_ge3_anywhere=bool(companions),overlap_other_pred_ge3=bool(strict),
        overlap_other_pred_ge3_half=any(o['coverage_of_target']>=.5 for o in strict),
        overlap_other_pred_ge10=any(o['relation_sec']>=10 for o in strict))
    return companions,flags

def prepare_metadata():
    source_audit={name:dict(path=str(p),sha256=sha(p)) for name,p in SOURCES.items()}
    for name,expected in EXPECTED.items():
        assert source_audit[name]['sha256']==expected,(name,'changed source')
    raw=read(SOURCES['durations']);controls={key(r):r for r in read(SOURCES['controls'])}
    assert len(raw)==30142 and len(controls)==30142
    records={};pairs=collections.defaultdict(list)
    for r in raw:
        k=key(r);assert k not in records
        c=controls[k];fps=float(r['fps']);a=int(r['begin_fid']);b=int(r['end_fid'])
        assert fps>0 and b>a
        duration=(b-a)/fps;coexist=optional(r['joint_local_continuous_sec'])
        assert abs(duration-float(r['relation_sec']))<1e-7
        assert abs(duration-float(c['duration_sec']))<1e-7
        assert same_optional_number(coexist,optional(c['joint_contiguous_support_sec']))
        for col in ('subject_tid','object_tid','predicate','subject_class','object_class'):
            assert str(r[col])==str(c[col]),(k,col)
        vh,bh=int(r['VrdONE_hit']),int(r['BIG_hit'])
        expected_group={(1,1):'A',(1,0):'B',(0,0):'C',(0,1):'D'}[vh,bh]
        assert expected_group==r['group']==c['group_at100']
        assert vh==int(c['VrdONE_hit']) and bh==int(c['BIG_hit'])
        assert truth(r['F'])==truth(c['input_feasible'])
        item=dict(video_name=k[0],gt_index=k[1],subject_tid=int(r['subject_tid']),object_tid=int(r['object_tid']),
                  predicate=r['predicate'],subject_class=r['subject_class'],object_class=r['object_class'],
                  begin_fid=a,end_fid=b,fps=fps,relation_sec=duration,coexist_sec=coexist,
                  native_F=truth(r['F']),original_group=expected_group,VrdONE_hit=vh,BIG_hit=bh,
                  size_bin=c['size_bin'],speed_bin=c['speed_bin'],
                  relation_frames_without_both_GT_boxes=int(r['relation_frames_without_both_GT_boxes']))
        records[k]=item;pairs[(k[0],item['subject_tid'],item['object_tid'])].append(item)
    assert set(records)==set(controls)
    for rows in pairs.values():
        for r in rows:
            companions,flags=companion_flags(r,rows)
            r.update(companions=companions,competition_flags=flags,
                     slices={d:classify(r['relation_sec'],r['coexist_sec'],flags[d]) for d in DEFINITIONS},
                     pair_annotation_count=len(rows),pair_predicate_annotation_count=sum(x['predicate']==r['predicate'] for x in rows))
    videos=collections.Counter(r['video_name'] for r in records.values())
    for r in records.values():r['video_GT_load']=videos[r['video_name']]
    return records,source_audit

def bind_boundary(metadata):
    raw=read(SOURCES['boundary']);assert len(raw)==2573
    assert len({r['video_name'] for r in raw})==27
    result=[];seen=set()
    for r in raw:
        k=key(r);assert k not in seen;seen.add(k);m=metadata[k]
        for col in ('subject_tid','object_tid','begin_fid','end_fid'):assert int(r[col])==m[col]
        assert r['predicate']==m['predicate'] and abs(float(r['relation_sec'])-m['relation_sec'])<1e-7
        assert same_optional_number(optional(r['joint_local_continuous_sec']),m['coexist_sec'])
        a,b=optional(r['pred_begin_fid']),optional(r['pred_end_fid'])
        valid=not any(truth(r[k]) for k in ('no_pair','no_response','min5_rejected'))
        if valid:assert a is not None and b is not None and b-a>=5
        t=interval_iou(a,b,m['begin_fid'],m['end_fid']) if valid else 0.
        assert abs(t-float(r['main_tiou_min5']))<1e-9,(k,'interval mismatch')
        ratio=(b-a)/(m['end_fid']-m['begin_fid']) if valid else None
        if valid:assert abs(ratio-float(r['length_ratio']))<1e-9
        item=dict(m,source_scope='GT27_correct_entity_input',readout='target_predicate_maxclass_query_min5',
                  pred_available=valid,tiou=t,duration_ratio=ratio,pred_begin=a if valid else None,pred_end=b if valid else None,
                  oracle_tiou=float(r['oracle_tiou_min5']),
                  chosen_pair_covers_gt=truth(r['chosen_pair_covers_gt']),
                  chosen_pair_disjoint_from_gt=truth(r['chosen_pair_disjoint_from_gt']),
                  no_response=truth(r['no_response']),min5_rejected=truth(r['min5_rejected']),
                  pair_idx=int(optional(r['pair_idx'])) if optional(r['pair_idx']) is not None else None,
                  query=int(optional(r['query'])) if optional(r['query']) is not None else None,
                  original_identity_predicate_GT_count=int(r['identity_predicate_GT_count']))
        item['prediction_key']=json.dumps([item['video_name'],item['pair_idx'],item['query'],item['predicate']],ensure_ascii=False)
        item['pred_interval_disjoint']=valid and t==0
        item['overlap_stretch']=valid and t>0 and ratio>=2
        item['no_overlap_stretch']=valid and t==0 and ratio>=2
        item['pred_interval_overlaps']=valid and t>0
        result.append(item)
    assert sum(r['tiou']>=.5 for r in result)==758
    assert sum(r['oracle_tiou']>=.5 for r in result)==1731
    return result

def expand(records,source_scope,readout):
    expanded=[]
    for m in records:
        pops=['all']+(['native_F'] if m['native_F'] else [])
        if source_scope=='GT27_correct_entity_input':
            if m['chosen_pair_covers_gt']:
                pops+=['chosen_pair_covers_gt']
                if m['pair_predicate_annotation_count']==1:pops+=['chosen_pair_covers_gt_unique_annotation']
            if m['relation_sec']<1:pops+=['subsecond_only']
        for definition in DEFINITIONS:
            for population in pops:
                r=dict(m,slice=m['slices'][definition],slice_definition=definition,analysis_population=population,
                       source_scope=source_scope,readout=readout)
                for col in ('pred_available','tiou','duration_ratio'):r.setdefault(col,None)
                expanded.append(r)
    return expanded

def native_summary(full_rows):
    groups=collections.defaultdict(list)
    for r in full_rows:groups[(r['slice_definition'],r['analysis_population'],r['slice'])].append(r)
    results=[]
    for (definition,population,slice_name),rs in sorted(groups.items()):
        for model in ('VrdONE','BIG'):
            hits=sum(r[model+'_hit'] for r in rs);n=len(rs)
            results.append(dict(source_scope='VidOR_val_30142GT_832_GT_bearing_videos',slice_definition=definition,
                analysis_population=population,slice=slice_name,model=model,N=n,N_videos=len({r['video_name'] for r in rs}),
                hit_count=hits,miss_count=n-hits,miss_rate=(n-hits)/n,
                metric='original Top100 category+subject/object vIoU>=0.5 one-to-one matching; not temporal-only tIoU'))
    return results

def oracle_summary(rows):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        groups[(r['source_scope'],r['slice_definition'],r['analysis_population'])][r['slice']].append(r)
    result=[]
    for (scope,definition,population),slices in sorted(groups.items()):
        for name in SLICES:
            rs=slices[name];n=len(rs);values=[r['oracle_tiou'] for r in rs]
            z=dict(source_scope=scope,readout='any_query_best_tIoU_ORACLE',slice_definition=definition,
                   analysis_population=population,slice=name,N=n,N_videos=len({r['video_name'] for r in rs}),
                   mean_tiou=math.fsum(values)/n if n else None,
                   stretch_rate=None,duration_bias=None,interval_selection='GT_time_ORACLE_not_deployed_readout')
            for threshold,tag in ((.3,'03'),(.5,'05'),(.7,'07')):
                misses=sum(v<threshold for v in values)
                z.update({f'miss_count_tiou{tag}':misses,f'miss_rate_tiou{tag}':misses/n if n else None,
                          f'miss_denominator_tiou{tag}':n})
            result.append(z)
    return result

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    metadata,sources=prepare_metadata();boundary=bind_boundary(metadata)
    full=expand(list(metadata.values()),'VidOR_val_30142GT_832_GT_bearing_videos','native_temporal_pair_binding_UNAVAILABLE')
    gt=expand(boundary,'GT27_correct_entity_input','target_predicate_maxclass_query_min5')
    rows=full+gt;summary=build_summary(rows)
    write_summary_csv(out/'diag_summary.csv',summary)
    write_csv(out/'official_detection_summary.csv',native_summary(full))
    write_csv(out/'per_gt_slices.csv',list(metadata.values()))
    write_csv(out/'per_gt_boundary_joined.csv',boundary)
    with (out/'per_gt_canonical.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    # Oracle receives its own table because it has no deployed selected interval.
    write_csv(out/'oracle_tiou_summary.csv',oracle_summary(gt))
    counters=collections.Counter(r['original_group'] for r in metadata.values())
    assert dict(counters)=={'A':2487,'B':1809,'C':25373,'D':473}
    receipt=dict(complete=True,model_or_training_executed=False,N_full_GT=len(metadata),
        N_GT_bearing_videos=len({r['video_name'] for r in metadata.values()}),N_boundary_GT=len(boundary),N_boundary_videos=27,
        ABCD=dict(counters),native_temporal_fullval_available=False,temporal_metric_scope='GT27 correct entity input only',
        interval_recomputation='all2573 original endpoints independently recomputed; 758 main hits and1731 oracle at0.5 preserved',
        sources=sources,script_sha256=sha(Path(__file__)),stats_sha256=sha(HERE/'stats.py'),protocol_sha256=sha(HERE/'PROTOCOL.md'))
    for name,src in sources.items():assert sha(src['path'])==src['sha256']
    receipt['output_sha256']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
