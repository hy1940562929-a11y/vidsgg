"""Rebuild fixed-identity predictions from interval outputs; official evaluation."""
from pathlib import Path
import json,hashlib,gzip,collections
import numpy as np
from frozen043 import load,dump,sha

ARMS=('baseline','mask','boundary')

def rebuild(base,selected,geometry,intervals):
    assert len(base)==len(selected)==len(intervals)
    out=[]
    for b,c,d in zip(base,selected,intervals):
        a,e=map(int,d);assert e-a>=5
        tr=[]
        for tid in (c['sid'],c['oid']):
            ts,te=geometry['durations'][tid];assert ts<=a<e<=te
            tr.append(geometry['boxes'][tid][a-ts:e-ts])
        z=dict(b,duration=[a,e],sub_traj=tr[0],obj_traj=tr[1]);out.append(z)
    return out

def canonical_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def stats(rows):
    result={'N_GT':len(rows),'N_videos':len({r['video_name'] for r in rows}),'arms':{}}
    for a in ARMS:
        z={'hits':{},'comparisons':{}}
        for k in (50,100,200):z['hits'][str(k)]=sum(0<=r['ranks'][a]<k for r in rows)
        for b in ARMS:
            z['comparisons'][b]={}
            for k in (50,100,200):
                new=sum(0<=r['ranks'][a]<k and not 0<=r['ranks'][b]<k for r in rows)
                lost=sum(0<=r['ranks'][b]<k and not 0<=r['ranks'][a]<k for r in rows)
                z['comparisons'][b][str(k)]={'new':new,'lost':lost,'net':new-lost}
        result['arms'][a]=z
    return result

def evaluate(out,cohorts):
    seal=load(out/'intervals_seal_before_GT.json')
    for rel,h in seal.items():assert sha(out/rel)==h
    rows=[];videos=[];summary={};sources={}
    for cohort,contract in cohorts.items():
        ex=Path(contract['export']);m=load(ex/'manifest.json');assert sha(ex/'manifest.json')==contract['manifest_sha256']
        assert sha(contract['oracle'])==contract['oracle_sha256'];old={(r['video_name'],r['gt_index']):r for r in load(Path(contract['oracle']))}
        official=ex/'official.py';assert sha(official)==m['outputs']['official.py']['sha256'];ns={};exec(compile(official.read_text(),str(official),'exec'),ns)
        sources[str(official)]=sha(official);sources[contract['oracle']]=contract['oracle_sha256']
        for full in m['video_names']:
            def read(n):
                p=ex/'videos'/full/n;assert sha(p)==m['outputs'][str(p.relative_to(ex))]['sha256'];sources[str(p)]=sha(p);return load(p)
            base,sel,g=read('baseline.json.gz'),read('selected.json.gz')['selected'],read('geometry.json.gz')
            gt=read('gt.json.gz');rec=load(out/'intervals'/cohort/(full+'.json.gz'))
            assert rec['video_name']==full and rec['N_candidates']==len(base)
            scores={};ranks={};predhash={}
            for arm in ARMS:
                pred=rebuild(base,sel,g,rec['durations'][arm]);predhash[arm]=canonical_sha(pred)
                assert predhash[arm]==rec['canonical_prediction_sha256'][arm]
                if arm=='baseline':assert pred==base
                prec,recall,_,rank=ns['eval_detection_scores_v2'](gt,pred,.5)
                scores[arm]=float(ns['voc_ap'](recall,prec));ranks[arm]=rank.tolist()
            videos.append(dict(cohort=cohort,video_name=full,N_GT=len(gt),AP=scores,hits={a:{str(k):sum(0<=r<k for r in ranks[a]) for k in (50,100,200)} for a in ARMS},canonical_prediction_sha256=predhash))
            for j in range(len(gt)):
                o=old[(full,j)];assert o['baseline_rank']==ranks['baseline'][j]
                t=o['original_group']=='C' and o['duration_sec']<3 and o['bucket025']=='grid_feasible_no_matching_hull'
                u=t and not o['modes']['ANY']['logit']['available'];v=u and o['modes']['SAVED100']['grid']['available']
                rows.append(dict(cohort=cohort,video_name=full,gt_index=j,predicate=o['predicate'],duration_sec=o['duration_sec'],original_group=o['original_group'],input_feasible=o['input_feasible'],T=t,U=u,V=v,ranks={a:ranks[a][j] for a in ARMS}))
        rs=[r for r in rows if r['cohort']==cohort];vs=[v for v in videos if v['cohort']==cohort];assert len(rs)==contract['N_GT'] and len(vs)==64
        groups={'all':lambda r:True,'short':lambda r:r['duration_sec']<3,'subsecond':lambda r:r['duration_sec']<1,'long':lambda r:r['duration_sec']>=3,
          'C':lambda r:r['original_group']=='C','short_C':lambda r:r['duration_sec']<3 and r['original_group']=='C','F':lambda r:r['input_feasible'],
          'short_C_F':lambda r:r['input_feasible'] and r['duration_sec']<3 and r['original_group']=='C',
          'T':lambda r:r['T'],'U':lambda r:r['U'],'V':lambda r:r['V']}
        gs={name:stats([r for r in rs if fn(r)]) for name,fn in groups.items()}
        assert gs['all']['arms']['baseline']['hits']['100']==contract['baseline_hit100']
        summary[cohort]={'N_GT':len(rs),'N_videos':64,'mAP':{a:float(np.mean([v['AP'][a] for v in vs if v['N_GT']])) for a in ARMS},'groups':gs}
    pass_all=all(s['mAP']['boundary']>=s['mAP'][a] and s['groups']['all']['arms']['boundary']['hits']['100']>=s['groups']['all']['arms'][a]['hits']['100'] for s in summary.values() for a in ('baseline','mask'))
    pass_target=all(summary['048']['groups']['short_C']['arms']['boundary']['hits']['100']>summary['048']['groups']['short_C']['arms'][a]['hits']['100'] for a in ('baseline','mask'))
    pass_V=summary['048']['groups']['V']['arms']['boundary']['hits']['100']>=1
    dump(out/'per_gt.json.gz',rows);dump(out/'per_video.json',videos)
    return dict(cohorts=summary,gates={'overall_both_cohorts':pass_all,'048_shortC_exceeds_both_controls':pass_target,'048_V_recovery':pass_V},
        decision='SMALL_PILOT_CONTINUATION_SIGNAL' if pass_all and pass_target and pass_V else 'DO_NOT_EXPAND_THIS_VERSION',evaluation_source_sha256=sources)
