"""Read-only source audit; writes only this audit's recomputation.json.

No inference/training; no historical outputs are overwritten. Standard library only.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sources = {}

def read_csv(rel):
    p = ROOT / rel
    sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(rel):
    p = ROOT / rel
    sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def yes(x):
    assert str(x) in ('True', 'False', 'true', 'false', '1', '0'), x
    return str(x).lower() in ('true', '1')

def key(r):
    return r['video_name'], int(r['gt_index'])

def gt_stats(rows):
    b = [yes(r['native_viou50_min5']) for r in rows]
    n = [yes(r['gt_viou50_min5']) for r in rows]
    return dict(N=len(rows), native_viou50=sum(b), gt_viou50=sum(n),
                gains=sum(not a and c for a,c in zip(b,n)),
                losses=sum(a and not c for a,c in zip(b,n)),
                gt_binary50=sum(yes(r['gt_binary50_min5']) for r in rows),
                gt_tiou50=sum(yes(r['gt_tiou50_min5']) for r in rows),
                grid_hull_tiou50_feasible=sum(yes(r['grid_hull_tiou50_feasible_min5']) for r in rows))

def main():
    inst = read_csv('handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/instance_comparison.csv')
    imap = {key(r):r for r in inst}
    assert len(imap) == len(inst) == 30142
    base = dict(N=len(inst), videos_with_GT=len({r['video_name'] for r in inst}),
                groups=dict(Counter(r['group_at100'] for r in inst)))

    dur = read_csv('analysis/entity_relation_duration_20260929/per_gt_durations.csv')
    dm = {key(r): r for r in dur}
    assert len(dm) == len(dur) == 30142 and set(dm) == set(imap)
    for r in dur:
        i = imap[key(r)]
        assert (int(r['subject_tid']),int(r['object_tid']),r['predicate'],int(r['begin_fid']),int(r['end_fid'])) == (int(i['subject_tid']),int(i['object_tid']),i['predicate'],int(i['begin_fid']),int(i['end_fid']))
        assert float(r['relation_sec']) <= float(r['joint_local_continuous_sec']) + 1e-9
        assert float(r['joint_local_continuous_sec']) <= min(float(r['subject_local_sec']),float(r['object_local_sec'])) + 1e-9
    short = [r for r in dur if float(r['relation_sec']) < 3]
    cohorts = dict(all_short=short,short_C=[r for r in short if r['group']=='C'],
                   short_CF=[r for r in short if r['group']=='C' and yes(r['F'])],
                   short_C_infeasible=[r for r in short if r['group']=='C' and not yes(r['F'])])
    duration_results={}
    for name,rows in cohorts.items():
        duration_results[name]=dict(N=len(rows), cumulative=dict(Counter(r['entity_bucket'] for r in rows)),
            local=dict(Counter(r['entity_local_bucket'] for r in rows)),
            joint_local_ge3=sum(float(r['joint_local_continuous_sec'])>=3 for r in rows),
            joint_local_ge10=sum(float(r['joint_local_continuous_sec'])>=10 for r in rows))

    gt = read_csv('handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/per_gt.csv')
    selection = read_json('handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/run/selection.json')
    gm={key(r):r for r in gt}
    assert len(gm)==len(gt)==2573
    assert {r['video_name'] for r in gt}==set(selection)
    assert set(gm)=={k for k in imap if k[0] in selection}
    for r in gt:
        assert r['group_at100']==imap[key(r)]['group_at100']
    groups={'overall':gt,'C':[r for r in gt if r['group_at100']=='C'],
            'short_lt3':[r for r in gt if float(r['duration_sec'])<3]}
    for label in ('<1s','1-3s','3-10s','>=10s'):
        groups['duration_'+label]=[r for r in gt if r['duration_bin']==label]
    for limit in (3,10):
        groups['short_joint_ge'+str(limit)]=[r for r in gt if float(r['duration_sec'])<3 and float(dm[key(r)]['joint_local_continuous_sec'])>=limit]
    gt_results={name:gt_stats(rows) for name,rows in groups.items()}
    derived=read_json('analysis/gt_input_mask_probe_20260929/derived_summary.json')
    for name in ('overall','C','short_lt3'):
        assert all(derived[name][k]==v for k,v in gt_results[name].items())

    events=read_csv('handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/evaluation/unique_time_events.csv')
    evkeys={(r['video_name'],r['subject_tid'],r['object_tid'],r['begin_fid'],r['end_fid']) for r in events}
    assert len(events)==len(evkeys)==2512
    assert sum(int(r['relation_count']) for r in events)==2573
    event_results={}
    for metric in ('binary50','tiou50','viou50'):
        matched=[r for r in events if yes(r[metric+'_min5_matched'])]
        ids=[]
        for r in matched:
            w=json.loads(r[metric+'_min5_witness'])
            ids.append((r['video_name'], w['candidate_id']))
        assert len(ids)==len(set(ids))
        event_results[metric]=dict(N=len(events),matched=len(matched),
            independently_available=sum(yes(r[metric+'_min5_independent_exists']) for r in events))

    preflight=read_json('handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/run/preflight.json')
    assert len(preflight)==len({r['video'] for r in preflight})==27
    assert all('native_forward_seconds' in r and 'gt_forward_seconds' in r for r in preflight)
    decoding=read_json('handoff/incoming/VIDOR-GT-MASK-PROBE-20260929/run/decoding_consistency.json')
    results=dict(scope='Local existing-output aggregation and implementation audit; not rerun model or re-search absent raw outputs',
        base=base,entity_duration=duration_results,gt_probe=gt_results,gt_event_matching=event_results,
        gt_preflight=dict(videos=len(preflight),aligned_frames=sum(r['aligned_entity_frames'] for r in preflight),
          gt_pairs_restored_by_disabling_dedup=sum(r['gt_valid_segments']-r['old_loader_gt_valid_segments'] for r in preflight)),
        decoding_consistency=decoding,source_sha256=sources)
    (OUT/'recomputation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ('source_sha256','decoding_consistency')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
