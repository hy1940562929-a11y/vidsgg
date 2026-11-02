#!/usr/bin/env python3
"""Decompose the fixed 077 diagnostic readout; never choose a new prediction."""
import argparse, collections, csv, json
from pathlib import Path
from run import DEFINITIONS, write_csv, sha, truth

def audit(path, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with Path(path).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    identities=collections.defaultdict(list)
    for r in rows:
        r['relation_sec']=float(r['relation_sec']);r['tiou']=float(r['tiou'])
        r['slices']=json.loads(r['slices']);r['competition_flags']=json.loads(r['competition_flags'])
        for k in ('pred_available','chosen_pair_covers_gt','pred_interval_disjoint','overlap_stretch','no_overlap_stretch'):
            r[k]=truth(r[k])
        r['duration_ratio']=None if r['duration_ratio']=='NA' else float(r['duration_ratio'])
        identity=(r['video_name'],r['subject_tid'],r['object_tid'],r['predicate'])
        r['_identity']=identity;identities[identity].append(r)
    for rs in identities.values():
        keys={r['prediction_key'] for r in rs if r['pred_available']}
        # 077 selects at most one query per identity/predicate; valid/no-valid is
        # likewise identical across repeated occurrences of this identity.
        assert len(keys)<=1
        mixed=any(r['relation_sec']<3 for r in rs) and any(r['relation_sec']>=3 for r in rs)
        for r in rs:r['_mixed']=mixed;r['_repeat']=len(rs)>1
    grouped=collections.defaultdict(list)
    for r in rows:
        pops=['all']+(['chosen_pair_covers_gt'] if r['chosen_pair_covers_gt'] else [])
        if r['chosen_pair_covers_gt'] and int(r['pair_predicate_annotation_count'])==1:
            pops.append('chosen_pair_covers_gt_unique_annotation')
        for pop in pops:
            for definition in DEFINITIONS:
                grouped[(definition,pop,r['slices'][definition])].append(r)
            if r['relation_sec']<3:grouped[('duration_only',pop,'short_lt3')].append(r)
            else:grouped[('duration_only',pop,'long_ge3')].append(r)
    result=[]
    for (definition,pop,name),rs in sorted(grouped.items()):
        available=[r for r in rs if r['pred_available']];n=len(rs);m=len(available)
        stretch=sum(r['duration_ratio']>=2 for r in available)
        overlap=sum(r['overlap_stretch'] for r in rs);disjoint=sum(r['no_overlap_stretch'] for r in rs)
        assert stretch==overlap+disjoint
        result.append(dict(slice_definition=definition,analysis_population=pop,slice=name,N=n,N_pred=m,
            N_videos=len({r['video_name'] for r in rs}),
            N_distinct_identity_predicate=len({r['_identity'] for r in rs}),
            N_distinct_selected_predictions=len({r['prediction_key'] for r in available}),
            N_GT_in_repeated_identity_predicate=sum(r['_repeat'] for r in rs),
            N_GT_in_mixed_short_long_identity_predicate=sum(r['_mixed'] for r in rs),
            N_no_prediction=n-m,N_input_segment_not_covers=sum(not r['chosen_pair_covers_gt'] for r in rs),
            N_pred_interval_disjoint=sum(r['pred_interval_disjoint'] for r in rs),
            N_stretch=stretch,N_overlap_stretch=overlap,N_disjoint_stretch=disjoint,
            disjoint_fraction_of_stretch=disjoint/stretch if stretch else None,
            hit_tiou05=sum(r['tiou']>=.5 for r in rs)))
    write_csv(out/'readout_audit.csv',result)
    cohort=dict(N=len(rows),N_identity_predicate_groups=len(identities),
        N_repeated_identity_predicate_groups=sum(len(rs)>1 for rs in identities.values()),
        N_GT_in_repeated_groups=sum(r['_repeat'] for r in rows),
        N_mixed_groups=sum(any(r['_mixed'] for r in rs) for rs in identities.values()),
        N_short_GT_in_mixed_groups=sum(r['_mixed'] and r['relation_sec']<3 for r in rows),
        no_new_model_readout=True,source=str(Path(path).resolve()),source_sha256=sha(path),
        script_sha256=sha(__file__),output_sha256=sha(out/'readout_audit.csv'))
    (out/'manifest.json').write_text(json.dumps(cohort,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(cohort,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.input,a.out)
