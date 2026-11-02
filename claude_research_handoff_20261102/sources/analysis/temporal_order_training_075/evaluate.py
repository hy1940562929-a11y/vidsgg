"""075 official fixed-candidate evaluation, only after prediction sealing."""
from pathlib import Path
import numpy as np
from frozen043 import load, dump, sha
from frozen_evaluate049 import rebuild, canonical_sha

ARMS = ('baseline', 'original', 'balanced', 'ordered')


def stats(rows):
    out = dict(N_GT=len(rows), N_videos=len({r['video_name'] for r in rows}), arms={})
    for a in ARMS:
        out['arms'][a] = dict(hits={str(k):sum(0 <= r['ranks'][a] < k for r in rows) for k in (50,100,200)}, comparisons={})
        for b in ARMS:
            comparisons = {}
            for k in (50,100,200):
                new = sum(0 <= r['ranks'][a] < k and not 0 <= r['ranks'][b] < k for r in rows)
                lost = sum(0 <= r['ranks'][b] < k and not 0 <= r['ranks'][a] < k for r in rows)
                comparisons[str(k)] = dict(new=new,lost=lost,net=new-lost)
            out['arms'][a]['comparisons'][b] = comparisons
    return out


def evaluate(out, cohorts):
    seal=load(out/'intervals_seal_before_GT.json')
    expected={'intervals/'+c+'/'+v+'.json.gz' for c,x in cohorts.items() for v in load(Path(x['export'])/'manifest.json')['video_names']}
    assert len(expected)==128 and set(seal)==expected
    assert {str(p.relative_to(out)) for p in (out/'intervals').rglob('*.json.gz')}==expected
    for rel,h in seal.items():
        assert sha(out/rel) == h
    groups_path = Path(__file__).parent/'evaluation_groups.json'
    groups = {(r['video_name'],r['gt_index']):r for r in load(groups_path)['rows']}
    rows, videos, sources, summaries = [], [], {str(groups_path):sha(groups_path)}, {}
    for cohort,c in cohorts.items():
        ex = Path(c['export']); m = load(ex/'manifest.json')
        assert sha(ex/'manifest.json') == c['manifest_sha256']
        assert sha(c['oracle']) == c['oracle_sha256']
        old = {(r['video_name'],r['gt_index']):r for r in load(c['oracle'])}
        p = ex/'official.py'; assert sha(p) == m['outputs']['official.py']['sha256']
        sources[str(p)] = sha(p); ns = {}; exec(compile(p.read_text(),str(p),'exec'),ns)
        for full in m['video_names']:
            def read(name):
                p = ex/'videos'/full/name; assert sha(p) == m['outputs'][str(p.relative_to(ex))]['sha256']
                sources[str(p)] = sha(p); return load(p)
            base,sel,g = read('baseline.json.gz'),read('selected.json.gz')['selected'],read('geometry.json.gz')
            gt=read('gt.json.gz'); rec=load(out/'intervals'/cohort/(full+'.json.gz'))
            assert rec['video_name']==full and rec['N_candidates']==len(base)
            ap, ranks = {}, {}
            for arm in ARMS:
                pred = rebuild(base,sel,g,rec['durations'][arm])
                assert canonical_sha(pred)==rec['canonical_prediction_sha256'][arm]
                if arm=='baseline': assert pred == base
                # Candidate identities/order/scores are inherited unchanged.
                assert [p['score'] for p in pred] == [p['score'] for p in base]
                prec,recall,_,rank=ns['eval_detection_scores_v2'](gt,pred,.5)
                ap[arm]=float(ns['voc_ap'](recall,prec)); ranks[arm]=rank.tolist()
            videos.append(dict(cohort=cohort,video_name=full,N_GT=len(gt),AP=ap,
                hits={a:{str(k):sum(0<=x<k for x in ranks[a]) for k in (50,100,200)} for a in ARMS}))
            for j in range(len(gt)):
                o=old[(full,j)]; x=groups[(full,j)]
                assert ranks['baseline'][j]==o['baseline_rank']
                assert abs(float(x['relation_sec'])-o['duration_sec'])<.01
                assert x['predicate']==o['predicate']==gt[j]['triplet'][1]
                assert x['original_duration']==gt[j]['duration']
                if x['fixed073_grid_not_threshold']:
                    assert o['original_group']=='C' and x['relation_sec']<3 and x['joint_local_continuous_sec']>=10
                t=o['original_group']=='C' and o['duration_sec']<3 and o['bucket025']=='grid_feasible_no_matching_hull'
                u=t and not o['modes']['ANY']['logit']['available']; v=u and o['modes']['SAVED100']['grid']['available']
                rows.append(dict(x,cohort=cohort,original_group=o['original_group'],input_feasible=o['input_feasible'],
                    T=t,U=u,V=v,ranks={a:ranks[a][j] for a in ARMS}))
        rs=[r for r in rows if r['cohort']==cohort]; vs=[v for v in videos if v['cohort']==cohort]
        assert len(rs)==c['N_GT'] and len(vs)==64
        summaries[cohort]=cohort_summary(rs,vs)
        assert summaries[cohort]['groups']['all']['arms']['baseline']['hits']['100']==c['baseline_hit100']
    assert len(rows)==4739
    summaries['combined']=cohort_summary(rows,videos)
    assert summaries['combined']['groups']['long_coexist_short']['N_GT']==331
    assert summaries['combined']['groups']['long_coexist_short_C']['N_GT']==325
    assert summaries['combined']['groups']['fixed073_grid_not_threshold']['N_GT']==63
    hit=lambda s,g,a:s['groups'][g]['arms'][a]['hits']['100']
    controls=('baseline','original','balanced')
    gates={
      'overall_both_cohorts':all(summaries[c]['mAP']['ordered']>=summaries[c]['mAP'][a] and hit(summaries[c],'all','ordered')>=hit(summaries[c],'all',a) for c in cohorts for a in controls),
      'combined_target_exceeds_all_controls':all(hit(summaries['combined'],'long_coexist_short','ordered')>hit(summaries['combined'],'long_coexist_short',a) for a in controls),
      'target_no_net_loss_each_cohort':all(hit(summaries[c],'long_coexist_short','ordered')>=hit(summaries[c],'long_coexist_short','baseline') for c in cohorts),
      'fixed_073_grid_not_threshold_actual_recovery':summaries['combined']['groups']['fixed073_grid_not_threshold']['arms']['ordered']['comparisons']['baseline']['100']['new']>=1,
      'persistent_no_net_loss':all(hit(summaries[c],g,'ordered')>=hit(summaries[c],g,'baseline') for c in cohorts for g in ('long','long_coexist_persistent'))}
    dump(out/'per_gt.json.gz',rows); dump(out/'per_video.json',videos)
    return dict(cohorts=summaries,gates=gates,decision='SMALL_PILOT_CONTINUATION_SIGNAL' if all(gates.values()) else 'DO_NOT_EXPAND_THIS_VERSION',evaluation_source_sha256=sources)


def cohort_summary(rows,videos):
    filters={'all':lambda r:True,'short':lambda r:r['relation_sec']<3,'subsecond':lambda r:r['relation_sec']<1,'long':lambda r:r['relation_sec']>=3,
        'C':lambda r:r['original_group']=='C','short_C':lambda r:r['relation_sec']<3 and r['original_group']=='C',
        'F':lambda r:r['input_feasible'],'short_C_F':lambda r:r['input_feasible'] and r['relation_sec']<3 and r['original_group']=='C',
        'long_coexist_short':lambda r:r['relation_sec']<3 and r['joint_local_continuous_sec']>=10,
        'long_coexist_short_C':lambda r:r['relation_sec']<3 and r['joint_local_continuous_sec']>=10 and r['original_group']=='C',
        'long_coexist_persistent':lambda r:r['relation_sec']>=3 and r['joint_local_continuous_sec']>=10,
        'fixed073_grid_not_threshold':lambda r:r['fixed073_grid_not_threshold'],
        'T':lambda r:r['T'],'U':lambda r:r['U'],'V':lambda r:r['V']}
    return dict(N_GT=len(rows),N_videos=len(videos),mAP={a:float(np.mean([v['AP'][a] for v in videos if v['N_GT']])) for a in ARMS},
        mAP_numerator={a:float(sum(v['AP'][a] for v in videos if v['N_GT'])) for a in ARMS},mAP_denominator=sum(v['N_GT']>0 for v in videos),
        groups={k:stats([r for r in rows if fn(r)]) for k,fn in filters.items()})
