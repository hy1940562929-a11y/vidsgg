"""094 CPU-only field/support audit. No model forward, optimizer or performance evaluation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch

ROOT = Path('/root/autodl-tmp/VrdONE-data/commander_cache_import_094/cache_v4')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)
def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def add(name, source):
    p = ROOT/name
    manifest['files'].append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size, source=source))

torch.set_num_threads(2)
manifest = json.loads((ROOT/'manifest.json').read_text())
require(manifest['status']=='REMOTE_ASSEMBLY_PASS_PENDING_LOCAL_IMPORT','ASSEMBLY_NOT_PASS')
for f in manifest['files']:
    require(sha(ROOT/f['path'])==f['sha256'], 'FILE_SHA '+f['path'])
consistency = json.loads((ROOT/'forward_consistency.json').read_text())
require(consistency['status']=='PASS','NUMERICAL_NOT_PASS')
require(all(all(v==0 for v in c['max_abs'].values()) for c in consistency['checks']),'ASSIGNMENT_RECOMPUTATION_REQUIRED_NONEXACT')
matching = json.loads((ROOT/'frozen_matching.json').read_text())
match_by_key = {(m['role'],m['batch'],m['row']):m for m in matching['entries']}
require(len(match_by_key)==len(matching['entries'])==439,'MATCHING_CARDINALITY')
require(matching['initial_mask_embed_sha256']==sha(ROOT/'initial_mask_embed.pt'),'INIT_BINDING')
support=[]; pair_support=[]; schema={}; split_stats={}
for role in ['train','adapter_dev','eval']:
    stats=Counter()
    for path in sorted((ROOT/role).glob('*.pt')):
        c=load(path); B,C,T=c['f'].shape
        require(C==256 and c['hs'].shape==(4*B,256,9),'HS_SHAPE')
        require(c['valid'].shape==(B,1,T) and c['valid'].dtype==torch.bool,'VALID_SHAPE')
        require(c['cls'].shape==(4,B,9,51) and c['masks'].shape==(4,B,9,T),'OUTPUT_SHAPE')
        for k in ['f','hs','cls','masks']:
            require(c[k].dtype==torch.float32 and torch.isfinite(c[k]).all().item(),'DTYPE_FINITE '+k)
        schema.setdefault(role,dict(example_file=str(path.relative_to(ROOT)),fields={k:dict(shape=list(c[k].shape),dtype=str(c[k].dtype)) for k in ['f','hs','valid','cls','masks']}))
        valid=c['valid'].squeeze(1)
        for b in range(B):
            n=int(valid[b].sum())
            require(n>0 and torch.equal(valid[b],torch.arange(T)<n),'VALID_NOT_PREFIX')
            if role=='eval':
                require(B==1 and c['metadata']['N_valid']==n,'EVAL_VALID')
                continue
            meta=c['metadata'][b];target=c['targets'][b];preds=c['preds'][b]
            require(meta['N_valid']==n and meta['feat_stride']==4,'TRAIN_COORDINATES')
            require(len(meta['events'])==len(target)==len(preds)<=9,'EVENT_CARDINALITY')
            require(target.shape[1]==T and not target[:,n:].bool().any().item(),'TARGET_PADDING')
            ri=int(path.stem.rsplit('_',1)[1]);match=match_by_key[(role,ri,b)]
            require(match['cache_sha256']==sha(path),'MATCH_CACHE_SHA')
            q=match['query_indices'];g=match['target_indices']
            require(sorted(g)==list(range(len(target))) and len(set(q))==len(q) and all(0<=i<9 for i in q),'MATCH_COVERAGE')
            require(match['event_ids']==[meta['events'][i]['event_id'] for i in g],'MATCH_EVENT_ID')
            frames=meta['sample_frame_origin']+4*torch.arange(n)
            pair_support.append(dict(split=role,cache_path=str(path.relative_to(ROOT)),row=b,video_id=meta['video_id'],pair_key=meta['pair_key'],N_valid=n,crop_start_sample=meta['crop_start_sample'],sampling_seed=meta['sampling_seed'],stride_offset=meta['stride_offset'],randint_draws=meta['randint_draws'],censored_positive_cells=meta['censored_positive_cells']))
            for ti,event in enumerate(meta['events']):
                y=target[ti,:n].bool()
                require(torch.equal(y,(frames>=event['begin_fid'])&(frames<event['end_fid'])),'TARGET_COORDINATE_ID')
                positives=int(y.sum())
                label='zero_positive' if positives==0 else ('single_positive' if positives==1 else 'at_least_two_positive')
                stats[label]+=1;stats['targets']+=1
                sampled_support_end=int(frames[-1])+1
                support.append(dict(split=role,cache_path=str(path.relative_to(ROOT)),row=b,video_id=meta['video_id'],pair_key=str(meta['pair_key']),**event,query=q[g.index(ti)],positive_samples_checked=positives,loss_support=label,reason=('no positive point on retained grid' if positives==0 else 'retained'),sample_origin=int(frames[0]),sample_last_plus1=sampled_support_end,grid_starts_after_event_start=int(frames[0])>event['begin_fid'],grid_ends_before_event_end=sampled_support_end<event['end_fid'],crop_start_sample=meta['crop_start_sample']))
        stats['pairs']+=B;stats['batches']+=1
    split_stats[role]=dict(stats)
with (ROOT/'event_support.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
add('event_support.csv','CPU inspection of independent event targets and frozen matching')
write('pair_sampling_support.json',pair_support);add('pair_sampling_support.json','unchanged per-pair sampling/crop records')
locked=list(csv.DictReader((ROOT/'sources/087/eval_locked.csv').open()))
require(len(locked)==2573,'LOCKED_GT_CARDINALITY')
eval_support=[]
for r in locked:
    v=r['video_name'];pi=r['pair_idx']
    p=ROOT/'eval'/f'gt27_{v}_{int(pi):04d}.pt' if pi!='' else None
    require(p is not None and p.is_file(),'LOCKED_EVAL_CACHE_MISSING')
    eval_support.append(dict(video_name=v,gt_index=int(r['gt_index']),pair_idx=int(pi),legacy_query=int(r['query']),cache_path=str(p.relative_to(ROOT)),status='present',missing_reason=None))
write('eval_support.json',eval_support);add('eval_support.json','all2573 original locked rows; no member filtering')
init=load(ROOT/'initial_mask_embed.pt')
write('tensor_schema.json',dict(
    examples=schema,
    axes=dict(f='[B,256,T]',hs='[4*B,256,9], layer-major; decoder layers0..3',valid='[B,1,T], true prefix',cls='[4,B,9,51], background index0',masks='[4,B,9,T]',preds='list[B] of [n_events] predicate IDs',targets='list[B] of [n_events,T] independent event masks'),
    init_parameters={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in init['state_dict'].items()},
    initialization=dict(parent_checkpoint_sha256=init['parent_checkpoint_sha256'],checkpoint_key=init['checkpoint_key'],trained=init['trained']),
    layer_contract='original shared mask_embed; future E091 updates applied to layer3 only; early-layer outputs remain cached',
    coordinates=dict(unit='original video frame index',GT_and_pair_interval='half-open [start,end)',train_grid='sample_frame_origin + feat_stride*k',eval_grid='origin + 4*k',stride=4,decoded_interval='[first_positive_frame,last_positive_frame+1)',min_decoded_frame_length=5,fps_source='sources/annotations; train metadata also contains fps',sampling='original seeds, stride_offset, crop_start_sample and randint_draws retained',grid_flags='grid endpoint mismatch includes stride effects; not automatically attributed to truncation'),
))
add('tensor_schema.json','verified imported tensor fields and source coordinate contract')
write('support_audit.json',dict(status='PASS',splits=split_stats,matching_entries=len(matching['entries']),each_target_matched_once=True,unique_queries_per_pair=True,all_projection_errors_zero=True,assignment_consistency='cached and reconstructed class/mask tensors are exactly equal; identical deterministic Hungarian inputs',no_training=True,no_performance_evaluation=True))
add('support_audit.json','CPU-only material audit')
manifest['support_audit']='support_audit.json';manifest['tensor_schema']='tensor_schema.json'
manifest['bytes']=sum(f['bytes'] for f in manifest['files'])
write('manifest.json',manifest)
print(json.dumps(dict(status='REMOTE_MATERIAL_AUDIT_PASS',splits=split_stats,files=len(manifest['files']),bytes=manifest['bytes'])))
