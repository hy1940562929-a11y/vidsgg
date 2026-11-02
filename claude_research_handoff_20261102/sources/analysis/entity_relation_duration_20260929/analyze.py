"""Cross-tabulate official GT object support and relation duration, no model runs."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
INST = ROOT/'handoff/incoming/VIDOR-PAIR-AUDIT-005/raw-20260928/instance_comparison.csv'
INP = ROOT/'handoff/incoming/VIDOR-VRDONE-FAILURE-008/input_audit_v2/per_gt_input_features.csv'
ANN = OUT/'validation-annotation.zip'
def load_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def key(r): return r['video_name'],int(r['gt_index'])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def runs(frames):
    out=[]
    for t in sorted(frames):
        if out and out[-1][1]==t: out[-1][1]=t+1
        else: out.append([t,t+1])
    return out
def signature(r):
    return (int(r['subject_tid']),int(r['object_tid']),r['predicate'],int(r['begin_fid']),int(r['end_fid']))
def bucket(s,o,threshold=3):
    n=int(s<threshold)+int(o<threshold)
    return ['both_long','one_short','both_short'][n]
def stats(rows):
    return dict(N=len(rows), C=sum(r['C'] for r in rows), F=sum(r['F'] for r in rows),
                CF=sum(r['C'] and r['F'] for r in rows), C_input_infeasible=sum(r['C'] and not r['F'] for r in rows),
                VrdONE_hits=sum(r['VrdONE_hit'] for r in rows),BIG_hits=sum(r['BIG_hit'] for r in rows),
                input_states=dict(Counter(r['input_state'] for r in rows)))
def main():
    assert hashlib.md5(ANN.read_bytes()).hexdigest()=='96c6870910e8d4fb3836878215432c1f'
    assert sha(INST)=='d30b6faf786b7edcf2d97f077c23e62a6166cad2fea08f68f49aa9d3d6c0a139'
    assert sha(INP)=='699458fe295fd93a1f181b9b892d483a04ba3d99ff6940a4792cd6247cf3013c'
    raw=load_csv(INST); inp={key(r):r for r in load_csv(INP)}
    assert len(raw)==len(inp)==30142 and {key(r) for r in raw}==set(inp)
    byvid=defaultdict(list)
    for r in raw: byvid[r['video_name']].append(r)
    records=[];manifest=[];audit=Counter();examples=defaultdict(list)
    with zipfile.ZipFile(ANN) as z:
        names=[n for n in z.namelist() if n.endswith('.json')]
        assert len(names)==835
        for name in names:
            data=z.read(name);ann=json.loads(data)
            video=Path(ann['video_path']).with_suffix('').as_posix().replace('/','_')
            rr=byvid[video];fps=float(ann['fps'])
            assert len(ann['trajectories'])==ann['frame_count']
            assert Counter(signature(r) for r in rr)==Counter(signature(r) for r in ann['relation_instances']),(video,'GT signature mismatch')
            cats={int(x['tid']):x['category'] for x in ann['subject/objects']}
            frames=defaultdict(set)
            for t,boxes in enumerate(ann['trajectories']):
                assert len({int(b['tid']) for b in boxes})==len(boxes)
                for b in boxes:
                    frames[int(b['tid'])].add(t)
                    audit['annotation_boxes']+=1; audit['generated_boxes']+=int(b.get('generated',0)!=0)
            manifest.append(dict(member=name,sha256=hashlib.sha256(data).hexdigest()))
            entity_runs={tid:runs(ff) for tid,ff in frames.items()}
            pair_cache={}
            for r in rr:
                si,oi,pred,s,e=signature(r);q=inp[key(r)]
                assert abs(float(r['fps'])-fps)<1e-9
                assert cats[si]==r['subject_class'] and cats[oi]==r['object_class']
                assert all(r[k]==q[k] for k in ['video_id','duration_sec','predicate','subject_class','object_class','VrdONE_hit_at100'])
                sf,of=frames[si],frames[oi]
                if (si,oi) not in pair_cache:
                    common=sf & of
                    pair_cache[si,oi]=(common,runs(common))
                common,common_runs=pair_cache[si,oi]
                relation=set(range(s,e))
                missing=len(relation-common)
                audit['relations_missing_GT_entity_frames']+=int(missing>0)
                containing=[(a,b) for a,b in common_runs if a<=s and e<=b]
                local=(containing[0][1]-containing[0][0])/fps if containing else None
                srun=[(a,b) for a,b in entity_runs[si] if a<=s and e<=b]
                orun=[(a,b) for a,b in entity_runs[oi] if a<=s and e<=b]
                slocal=(srun[0][1]-srun[0][0])/fps if srun else None
                olocal=(orun[0][1]-orun[0][0])/fps if orun else None
                sd,od=len(sf)/fps,len(of)/fps
                ss,os=(max(sf)-min(sf)+1)/fps,(max(of)-min(of)+1)/fps
                rec=dict(video_name=video,gt_index=int(r['gt_index']),subject_tid=si,object_tid=oi,predicate=pred,
                         subject_class=cats[si],object_class=cats[oi],begin_fid=s,end_fid=e,fps=fps,
                         relation_sec=(e-s)/fps,subject_gt_sec=sd,object_gt_sec=od,subject_span_sec=ss,object_span_sec=os,
                         subject_local_sec=slocal,object_local_sec=olocal,
                         entity_local_bucket=bucket(slocal,olocal) if slocal is not None and olocal is not None else 'missing',
                         joint_gt_sec=len(common)/fps,joint_local_continuous_sec=local,
                         relation_frames_without_both_GT_boxes=missing,group=r['group_at100'],C=r['group_at100']=='C',
                         F=q['input_pair_state']=='feasible_input_pair',input_state=q['input_pair_state'],
                         VrdONE_hit=int(r['VrdONE_hit_at100']),BIG_hit=int(r['BIG_hit_at100']),
                         entity_bucket=bucket(sd,od),span_bucket=bucket(ss,os))
                assert abs(rec['relation_sec']-float(r['duration_sec']))<1e-9
                records.append(rec)
    assert len(records)==30142
    assert Counter(r['group'] for r in records)==dict(A=2487,B=1809,C=25373,D=473)
    short=[r for r in records if r['relation_sec']<3]
    assert len(short)==11621 and sum(r['C'] for r in short)==10626
    assert sum(r['C'] and r['F'] for r in short)==4160
    cohorts={'all_GT':records,'short_relations':short,'short_C':[r for r in short if r['C']],
             'short_C_input_infeasible':[r for r in short if r['C'] and not r['F']],
             'short_C_input_feasible':[r for r in short if r['C'] and r['F']],
             'short_VrdONE_misses':[r for r in short if not r['VrdONE_hit']]}
    res={}
    for name,rows in cohorts.items():
        res[name]={'totals':stats(rows),'by_entity_duration':{b:stats([r for r in rows if r['entity_bucket']==b]) for b in ['both_long','one_short','both_short']},
                   'by_entity_local_duration':{b:stats([r for r in rows if r['entity_local_bucket']==b]) for b in ['both_long','one_short','both_short','missing']},
                   'local_both_long_but_joint_under3s':stats([r for r in rows if r['entity_local_bucket']=='both_long' and r['joint_local_continuous_sec']<3]),
                   'joint_at_least_3s':stats([r for r in rows if r['joint_gt_sec']>=3]),
                   'both_long_but_joint_under3s':stats([r for r in rows if r['entity_bucket']=='both_long' and r['joint_gt_sec']<3]),
                   'both_entity_at_least10s':stats([r for r in rows if min(r['subject_gt_sec'],r['object_gt_sec'])>=10]),
                   'joint_at_least10s':stats([r for r in rows if r['joint_gt_sec']>=10]),
                   'local_joint_at_least3s':stats([r for r in rows if r['joint_local_continuous_sec'] is not None and r['joint_local_continuous_sec']>=3]),
                   'local_joint_under3s':stats([r for r in rows if r['joint_local_continuous_sec'] is not None and r['joint_local_continuous_sec']<3]),
                   'local_joint_at_least10s':stats([r for r in rows if r['joint_local_continuous_sec'] is not None and r['joint_local_continuous_sec']>=10]),
                   'local_joint_missing':stats([r for r in rows if r['joint_local_continuous_sec'] is None]),
                   'span_vs_count_class_changes':sum(r['entity_bucket']!=r['span_bucket'] for r in rows)}
    for b in ['both_long','one_short','both_short']:
        candidates=[r for r in short if r['entity_bucket']==b and r['relation_frames_without_both_GT_boxes']==0]
        if b=='both_long':candidates=[r for r in candidates if r['joint_local_continuous_sec']>=10]
        for r in sorted(candidates,key=lambda r:(r['video_name'],r['gt_index']))[:3]:examples[b].append(r)
    result=dict(status='COMPLETED_DESCRIPTIVE_GT_REANALYSIS_NO_MODEL_RUN',definition='relation instance weighted, GT annotated frame count; 3s threshold',
                source_hashes={'instances':sha(INST),'inputs':sha(INP),'annotation_zip':sha(ANN),'script':sha(Path(__file__))},
                audit=dict(audit),cohorts=res,examples=dict(examples))
    with (OUT/'per_gt_durations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(sorted(records,key=lambda r:(r['video_name'],r['gt_index'])))
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'annotation_members.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'audit':result['audit'],'cohorts':res},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
