#!/usr/bin/env python3
"""088 local CPU capacity-matched interval diagnosis; no model or training imports."""
import argparse,csv,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from interval_math import require,generate_pair,evaluate_event,summarize,decide
HERE=Path(__file__).resolve().parent
PRIMARY='long_coexist_short_covered'
SECONDARY='long_coexist_subsecond_covered'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def load(path):return json.loads(Path(path).read_text())

def resolve(base,value):
    p=Path(value).expanduser();return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()

def check_hash(path,expected,receipt):
    require(Path(path).is_file(),'missing frozen input '+str(path))
    actual=sha(path);require(actual==expected,'source SHA mismatch '+str(path));receipt[str(Path(path).resolve())]=actual

def verify_freezes(manifest,protocol,software):
    sources={};p=load(protocol)
    require(isinstance(p.get('frozen_files'),dict),'protocol frozen_files required')
    for name,h in p['frozen_files'].items():check_hash(resolve(protocol.parent,name),h,sources)
    require(str((HERE/'PROTOCOL.md').resolve()) in sources,'protocol freeze omits PROTOCOL.md')
    sw=load(software);require(isinstance(sw.get('files'),dict),'software files map required')
    verified=set()
    for name,h in sw['files'].items():
        path=resolve(software.parent,name);check_hash(path,h,sources);verified.add(path)
    required={HERE/'run.py',HERE/'interval_math.py',manifest,protocol}
    require({p.resolve() for p in required}<=verified,'software freeze omits executable/math/manifest/protocol freeze')
    sources[str(protocol)]=sha(protocol);sources[str(software)]=sha(software)
    return sources

def bind_inputs(manifest_path,sources):
    manifest=load(manifest_path);refs={}
    for role in ('members','eval_locked','coordinates'):
        ref=manifest['sources'][role];p=resolve(manifest_path.parent,ref['path']);check_hash(p,ref['sha256'],sources);refs[role]=p
    members=load(refs['members']);coordinates=load(refs['coordinates'])
    primary={(str(v),int(i)) for v,i in members[PRIMARY]};secondary={(str(v),int(i)) for v,i in members[SECONDARY]}
    require(len(primary)==len(members[PRIMARY])==400 and len(secondary)==len(members[SECONDARY])==163,'frozen400/163 membership mismatch')
    require(secondary<=primary,'secondary members not a subset')
    with refs['eval_locked'].open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    bykey={(r['video_name'],int(r['gt_index'])):r for r in raw}
    require(len(bykey)==len(raw)==2573 and primary<=bykey.keys(),'GT lock duplicates or missing members')
    events=[];needed={}
    for video,gi in sorted(primary):
        r=bykey[(video,gi)];pi=int(r['pair_idx']);key=f'{video}/{pi}'
        require(key in coordinates,'missing fixed-pair coordinate '+key);c=coordinates[key]
        a,b=int(r['begin_fid']),int(r['end_fid']);fps=float(r['fps'])
        require(a<b and fps>0,'invalid annotation')
        require((b-a)/fps<3 and float(r['coexist_sec'])>=10 and r['chosen_pair_covers_gt'].lower()=='true','locked cohort definition changed')
        require(int(c['pair_start'])<=a<b<=int(c['pair_end']),'locked selected pair no longer covers GT')
        require(int(c['subject_tid'])==int(r['subject_tid']) and int(c['object_tid'])==int(r['object_tid']),'fixed pair identity mismatch')
        if (video,gi) in secondary:require((b-a)/fps<1,'secondary duration definition changed')
        needed[key]=c
        events.append(dict(video_name=video,gt_index=gi,pair_idx=pi,pair_key=key,original_locked_query=int(r['query']),subject_tid=int(r['subject_tid']),object_tid=int(r['object_tid']),predicate=r['predicate'],begin_fid=a,end_fid=b,fps=fps,relation_sec=(b-a)/fps,primary400=True,subsecond163=(video,gi) in secondary))
    require(len({e['video_name'] for e in events})==23,'primary23-video population changed')
    require(set(needed)<=set(manifest['score_files']),'score manifest missing a required pair')
    return manifest,events,needed

def build_library(manifest_path,manifest,needed,sources,out):
    real={};null={};audits=[]
    for key,c in sorted(needed.items()):
        ref=manifest['score_files'][key];source_path=resolve(manifest_path.parent,ref['path'])
        materialized=resolve(manifest_path.parent,ref['materialized_path']) if ref.get('materialized_path') else None
        path=materialized if materialized is not None and materialized.is_file() else source_path
        check_hash(path,ref['sha256'],sources)
        video,pair=key.rsplit('/',1)
        with np.load(path,allow_pickle=False) as z:
            require('frames' in z and 'native_masks' in z,'missing raw native score arrays')
            frames=z['frames'];masks=z['native_masks']
            require(masks.shape==(9,int(c['N_valid'])) and frames.shape==(int(c['N_valid']),),'full native sequence shape changed')
            expected=int(c['origin'])+4*np.arange(int(c['N_valid']))
            require(np.array_equal(frames,expected),'sample coordinate mismatch')
            r,n,a=generate_pair(video,int(pair),frames,masks)
            a.update(source_path=str(source_path),read_path=str(path),source_sha256=ref['sha256'],native_masks_sha256=hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest())
        real[key]=r;null[key]=n;audits.append(a)
    write(out/'candidates.json',dict(schema_version=1,generation_uses_GT=False,selector='mean float32 sigmoid accumulated in float64; tie peak probability then earliest start',pairs=real))
    write(out/'null_candidates.json',dict(schema_version=1,repetitions=100,hash_key='88/video/pair/query/repetition',repetition_indices='0..99',first64='big-endian',start_mapping='hash64*(N-length+1)//2**64',GT_or_arm_in_key=False,pairs=null))
    seal=dict(sealed_before_GT_evaluation=True,N_pairs=len(real),candidates_sha256=sha(out/'candidates.json'),null_sha256=sha(out/'null_candidates.json'))
    write(out/'candidate_seal.json',seal);write(out/'candidate_audits.json',audits)
    # Reload the sealed artifacts for the separate evaluation stage.
    require(sha(out/'candidates.json')==seal['candidates_sha256'] and sha(out/'null_candidates.json')==seal['null_sha256'],'candidate seal changed')
    return load(out/'candidates.json')['pairs'],load(out/'null_candidates.json')['pairs'],audits,seal

def output_per_gt_csv(path,rows):
    fields=['video_name','gt_index','pair_idx','original_locked_query','predicate','begin_fid','end_fid','subsecond163','H_hit','C_hit','gain','loss','H_max_tiou','C_max_tiou','null_H_hit_count_100','null_C_hit_count_100','null_H_expected_hit','null_C_expected_hit']
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            d={k:r[k] for k in fields if k in r}
            d.update(H_hit=int(r['H']['hit']),C_hit=int(r['C']['hit']),gain=int(r['C']['hit'] and not r['H']['hit']),loss=int(r['H']['hit'] and not r['C']['hit']),H_max_tiou=r['H']['max_tiou'],C_max_tiou=r['C']['max_tiou'],null_H_hit_count_100=r['null_H']['hit_count'],null_C_hit_count_100=r['null_C']['hit_count'],null_H_expected_hit=r['null_H']['expected_hit'],null_C_expected_hit=r['null_C']['expected_hit'])
            writer.writerow(d)

def run(manifest_path,out,protocol,software):
    begin=time.monotonic();out=Path(out).resolve();require(not out.exists(),'fresh output directory required');out.mkdir(parents=True)
    phase='FROZEN_INPUT_VALIDATION'
    try:
        manifest_path=Path(manifest_path).resolve();protocol=Path(protocol).resolve();software=Path(software).resolve()
        sources=verify_freezes(manifest_path,protocol,software);manifest,events,needed=bind_inputs(manifest_path,sources)
        write(out/'source_audit.json',dict(N_primary=400,N_secondary=163,N_primary_videos=23,N_required_pairs=len(needed),linear_probe_support_exclusion_applied=False,classification_filter_applied=False,pair_reselection=False,old47_baseline_comparison_performed=False,sha256=sources))
        phase='GT_FREE_CANDIDATE_GENERATION'
        real,null,audits,seal=build_library(manifest_path,manifest,needed,sources,out)
        print('088_CANDIDATES_SEALED',len(real),flush=True)
        phase='SEALED_CANDIDATE_GT_EVALUATION'
        rows=[evaluate_event(e,real[e['pair_key']],null[e['pair_key']]) for e in events]
        require(len(rows)==400 and sum(r['subsecond163'] for r in rows)==163,'evaluation denominator changed')
        write(out/'per_gt.json',rows);output_per_gt_csv(out/'per_gt.csv',rows)
        primary,pi=summarize(rows);secondary,si=summarize([r for r in rows if r['subsecond163']])
        np.savez(out/'bootstrap_indices.npz',primary400=pi,subsecond163=si)
        write(out/'per_video.json',dict(primary400=primary['per_video'],subsecond163=secondary['per_video']))
        capacity=dict(N_distinct_pairs=len(audits),query_slots_per_pair=9,totals={k:sum(a[k] for a in audits) for k in ('H_candidates','C_candidates','H_empty_slots','C_empty_slots','C_selectable_components')},per_pair_actual_C_not_above_H=all(a['C_candidates']<=a['H_candidates'] for a in audits),null_exact_actual_count_and_lengths=True)
        result=decide(primary)
        summary=dict(complete=True,experiment='088',decision=result,primary400=primary,subsecond163=secondary,capacity=capacity,training_steps=0,model_forwards=0,source_sha256=sources,candidate_seal=seal,numpy_version=np.__version__,elapsed_seconds=time.monotonic()-begin,independent_review='PENDING',scope='fixed-pair nine-query temporal-shape coverage; predicate unconstrained; GT evaluates but never selects components; not one-to-one detection recall',historical47_scope='old scores; all same-identity continuous segments and all queries; not this baseline',null_interpretation='mean of100 separate capacity-matched libraries; not pooled or best repetition',CI_interpretation='paired video bootstrap conditional on fixed100 null libraries; upper bound below0.05 excludes prespecified one-sided advantage, not two-sided equivalence')
        phase='FINAL_SOURCE_AND_SEAL_AUDIT'
        for path,digest in sources.items():require(sha(path)==digest,'input changed during execution '+path)
        require(sha(out/'candidates.json')==seal['candidates_sha256'] and sha(out/'null_candidates.json')==seal['null_sha256'],'candidate library changed during evaluation')
        write(out/'summary.json',summary);write(out/'source_sha256.json',sources)
        write(out/'artifact_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
        write(out/'status.json',dict(complete=True,status=result['status'],exit=result['exit']))
        print('COMPLETE088',result['status'],result['exit'],flush=True)
    except BaseException as error:
        write(out/'status.json',dict(complete=False,status='INCONCLUSIVE',technical_status='TECHNICAL_FAILURE',phase=phase,error=repr(error),elapsed_seconds=time.monotonic()-begin))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--input-manifest',type=Path,default=HERE/'input_manifest.json');p.add_argument('--protocol-freeze',type=Path,default=HERE/'protocol_freeze.json');p.add_argument('--software-freeze',type=Path,default=HERE/'software_freeze.json');a=p.parse_args();run(a.input_manifest,a.out,a.protocol_freeze,a.software_freeze)
