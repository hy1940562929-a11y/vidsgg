"""073: bounded, GT-assisted feature-history sensitivity. No source modifications."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import argparse, hashlib, json, math, signal, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

REPO=Path('/root/rebinding_project/VrdONE')
DATA=Path('/root/autodl-tmp/VrdONE-data')
CHECKPOINT_SHA='4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627'
CFG=DATA/'commander_native_error_025/export_v1/effective_config.json'
HERE=Path(__file__).resolve().parent

def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def setup():
    os.chdir(REPO);sys.path.insert(0,str(REPO))
    import torch
    from dataloaders.vidor import VidOR
    from dataloaders.category import vidor_pred_name_to_id
    import gt_input_source as helper
    cfg=json.loads(CFG.read_text());dc=cfg['dataset_config'];ds=VidOR.__new__(VidOR)
    for k in ('with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset'):
        setattr(ds,k,dc[k])
    assert dc['feat_stride']==4 and dc['stride_offset']==0 and not dc['random_stride']
    assert cfg['model_config']['max_seq_len']==512
    torch.set_num_threads(2)
    return torch,ds,cfg,helper,vidor_pred_name_to_id

def video_input(full,ds,helper):
    group,vid=full.split('_');ap=REPO/'datasets/vidor/annotations/validation'/group/(vid+'.json')
    ann=json.loads(ap.read_text());cache,tids,_=helper.gt_cache(full,ann,audit=False)
    inp=ds._val_getitem(cache,viou_threshold=1.01)
    return ann,inp,tids,ap

def windows(bits,size=8):
    # All contiguous candidates; candidates are selected without model outputs.
    if len(bits)<size:return []
    sums=np.convolve(np.asarray(bits,dtype=np.int32),np.ones(size,dtype=np.int32),'valid')
    return np.flatnonzero(sums==size).tolist()

def descriptors(block,source):
    a=block.astype(np.float64);s=source.astype(np.float64)
    return dict(rms=float(np.sqrt(np.mean((a-s)**2))),
                geometry_rms=float(np.sqrt(np.mean((a[-21:]-s[-21:])**2))),
                source_rms=float(np.sqrt(np.mean(s*s))),donor_rms=float(np.sqrt(np.mean(a*a))))

def choose_donors(x,edit,pstarts,nstarts):
    target=x[:,edit:edit+8]
    def records(starts):
        rows=[]
        for k in starts:
            d=descriptors(x[:,k:k+8],target)
            if d['rms']>1e-6:rows.append(dict(start=k,**d))
        return sorted(rows,key=lambda z:(z['geometry_rms'],z['rms'],z['start']))[:32]
    pp,nn=records(pstarts),records(nstarts);choices=[]
    for p in pp:
        for n in nn:
            ratio=n['rms']/p['rms']
            if .5<=ratio<=2:
                cost=abs(math.log(ratio))+p['geometry_rms']+n['geometry_rms']
                choices.append((cost,p['start'],n['start'],p,n))
    if not choices:return None
    _,_,_,p,n=min(choices,key=lambda z:z[:3]);return dict(P=p,N=n,RMS_ratio_N_over_P=n['rms']/p['rms'])

def prepare(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    torch,ds,cfg,helper,predids=setup();selected=[];exclusions=Counter();all_candidates=[];sources={};pervideo=[]
    videos=sorted(p.stem for p in helper.GT_PROP.glob('*.pkl'));assert len(videos)==27
    for full in videos:
        ann,inp,tids,ap=video_input(full,ds,helper);fps=float(ann['fps']);rel=defaultdict(list)
        for j,r in enumerate(ann['relation_instances']):
            rel[(int(r['subject_tid']),int(r['object_tid']),r['predicate'])].append((int(r['begin_fid']),int(r['end_fid']),j))
        sources[str(ap)]=sha(ap);sources[str(helper.GT_PROP/(full+'.pkl'))]=sha(helper.GT_PROP/(full+'.pkl'))
        eligible=[];rawcount=0
        for pi,(si,oi) in enumerate(zip(inp['sids'].tolist(),inp['oids'].tolist())):
            sid,oid=tids[si],tids[oi];a=max(int(inp['traj_durations'][si,0]),int(inp['traj_durations'][oi,0]));b=min(int(inp['traj_durations'][si,1]),int(inp['traj_durations'][oi,1]))
            if (b-a)/fps<10:continue
            x=inp['so_features_list'][pi];N=x.shape[1]
            if N>2048:exclusions['PAIR_OVER2048']+=1;continue
            origin=a+int(inp['so_offset'][pi]);tt=origin+4*np.arange(N);xx=x.numpy();guard=max(4,int(math.ceil(.5*fps)))
            pairrels=[(key[2],r) for key,rs in rel.items() if key[:2]==(sid,oid) for r in rs]
            for key,rs in rel.items():
                if key[:2]!=(sid,oid):continue
                unique={}
                for s,e,j in rs:unique.setdefault((s,e),j)
                rs=sorted((s,e,j) for (s,e),j in unique.items());allactive=np.zeros(N,dtype=bool);expanded=np.zeros(N,dtype=bool)
                for s,e,j in rs:
                    allactive|=(tt>=s)&(tt+4<=e)
                    expanded|=(tt>=s-guard)&(tt<e+guard)
                for (s,e,j),(u,w,k) in zip(rs,rs[1:]):
                    if not(a<=s<e<u<w<=b):continue
                    rawcount+=1
                    if any(max(e,aa)<min(u,bb) for aa,bb,_ in rs):
                        exclusions['GAP_COVERED_BY_OTHER_TARGET_ANNOTATION']+=1;continue
                    if (u-e)/fps<.5:exclusions['GAP_UNDER_HALF_SECOND']+=1;continue
                    gap=np.flatnonzero((tt>=e)&(tt+4<=u));nxt=np.flatnonzero((tt>=u)&(tt+4<=w))
                    if len(gap)<2 or len(nxt)<2:exclusions['GAP_OR_NEXT_LT2_POINTS']+=1;continue
                    editable=np.flatnonzero((tt>=s)&(tt+4<=e-guard))
                    if len(editable)<8:exclusions['PAST_ACTIVE_BLOCK_LT8']+=1;continue
                    end=int(editable[-1])+1;edit=end-8
                    past=(np.arange(N)+8<=edit)
                    # A complete P/N donor ends before the edited block; N is before first event.
                    pp=[q for q in windows(allactive) if q+8<=edit]
                    nn=[q for q in windows(~expanded) if q+8<=edit and tt[q+7]+4<=s-guard]
                    if not pp or not nn:exclusions['NO_P_OR_N_DONOR']+=1;continue
                    donors=choose_donors(xx,edit,pp,nn)
                    if donors is None:exclusions['NO_STRENGTH_MATCHED_DONORS']+=1;continue
                    ident=f'{full}:{sid}:{oid}:{key[2]}:{s}:{e}:{u}:{w}:{pi}'
                    cid=hashlib.sha256(ident.encode()).hexdigest()[:16]
                    seed=int(cid[:8],16);perm=np.random.default_rng(seed).permutation(8).tolist()
                    if perm==list(range(8)):perm=perm[1:]+perm[:1]
                    continuous=sorted({pred for pred,(z,v,_) in pairrels if pred!=key[2] and z<=tt[edit] and v>=w})
                    record=dict(case_id=cid,video_name=full,pair_idx=pi,subject_tid=sid,object_tid=oid,predicate=key[2],predicate_id=int(predids[key[2]]),
                        previous_gt_index=j,next_gt_index=k,previous=[s,e],next=[u,w],next_sec=(w-u)/fps,next_short=(w-u)/fps<3,
                        fps=fps,pair_support=[a,b],origin=origin,N=N,Tpad=max(512,int(math.ceil(N/64)*64)),edit=[edit,end],donors=donors,shuffle=perm,
                        gap_indices=gap.tolist(),next_indices=nxt.tolist(),continuing_predicates=[dict(predicate=p,predicate_id=int(predids[p])) for p in continuous],
                        feature_sha256=tensor_sha(x),visual_audit='PENDING',selection_uses_model_outputs=False)
                    eligible.append(record)
        eligible.sort(key=lambda r:r['case_id']);kept=[];perkey=Counter()
        for r in eligible:
            key=(r['subject_tid'],r['object_tid'],r['predicate'])
            if len(kept)>=8 or perkey[key]>=2:continue
            kept.append(r);perkey[key]+=1
        selected.extend(kept);all_candidates.extend(eligible);pervideo.append(dict(video_name=full,raw_transitions=rawcount,eligible=len(eligible),selected=len(kept)))
        print('PREPARE073',full,'eligible',len(eligible),'selected',len(kept),flush=True)
        del inp
    assert len({r['case_id'] for r in selected})==len(selected)
    sources[str(CFG)]=sha(CFG)
    for p in HERE.glob('*.py'):sources[str(p)]=sha(p)
    for relpath in ['models/maskvrd.py','models/predictor.py','dataloaders/vidor.py']:
        sources[str(REPO/relpath)]=sha(REPO/relpath)
    plan=dict(complete=True,kind='GT_ASSISTED_ANNOTATION_CONDITIONED_HISTORY',N_cases=len(selected),N_videos=len({r['video_name'] for r in selected}),
        N_short_next=sum(r['next_short'] for r in selected),N_continuing_control=sum(bool(r['continuing_predicates']) for r in selected),cases=selected,
        all_eligible_count=len(all_candidates),per_video=pervideo,exclusions=dict(exclusions),sources=sources,checkpoint_sha256=CHECKPOINT_SHA,
        prepare_seconds=time.monotonic()-start,visual_audit='PENDING',semantic_causal_claim_allowed=False)
    dump(out/'selection.json',plan);dump(out/'all_eligible.json',all_candidates);dump(out/'effective_config.json',cfg)
    print('PREPARE073_COMPLETE',json.dumps({k:plan[k] for k in ['N_cases','N_videos','N_short_next','N_continuing_control','all_eligible_count','exclusions']}),flush=True)

def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def infer(planpath,out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir();plan=json.loads(planpath.read_text())
    def timeout(*args):raise TimeoutError('073 wall budget 1200 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    torch,ds,cfg,helper,predids=setup();cp=REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==CHECKPOINT_SHA
    for p,h in plan['sources'].items():assert sha(p)==h,('SOURCE_CHANGED',p)
    from models.maskvrd import MaskVRD
    import utils.misc as misc
    misc.set_seed(42);model=MaskVRD(cfg['model_config'],device='cuda:0').eval();model._config_eval(cfg['inference_config'])
    weights=torch.load(cp,map_location='cpu',weights_only=False)['model_state_dict_ema']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True);del weights;model.cuda();before=state_sha(model)
    assert model.max_seq_len==512 and model.max_div_factor==64
    groups=defaultdict(list)
    for r in plan['cases']:groups[r['video_name']].append(r)
    audits=[];calls=0;noopdone=False
    try:
        for full,rows in groups.items():
            ann,inp,tids,ap=video_input(full,ds,helper)
            buckets=defaultdict(list)
            for r in rows:
                x=inp['so_features_list'][r['pair_idx']];assert tensor_sha(x)==r['feature_sha256'];buckets[r['Tpad']].append(r)
            for T,rr in buckets.items():
                for base in range(0,len(rr),2):
                    cases=rr[base:base+2];batch=[];masks=[];index=[]
                    for r in cases:
                        x=inp['so_features_list'][r['pair_idx']];l,u=r['edit'];N=r['N']
                        for arm in ['B','N','P','R']:
                            y=x.clone()
                            if arm in ['N','P']:
                                d=r['donors'][arm]['start'];y[:,l:u]=x[:,d:d+8]
                            elif arm=='R':y[:,l:u]=x[:,l:u][:,r['shuffle']]
                            assert torch.equal(y[:,:l],x[:,:l]) and torch.equal(y[:,u:],x[:,u:])
                            assert torch.equal(y[:,r['gap_indices']],x[:,r['gap_indices']]) and torch.equal(y[:,r['next_indices']],x[:,r['next_indices']])
                            z=torch.zeros((2069,T));z[:,:N]=y;mm=torch.zeros((1,T),dtype=torch.bool);mm[:,:N]=True
                            batch.append(z);masks.append(mm);index.append((r,arm))
                            audits.append(dict(case_id=r['case_id'],arm=arm,edited_input_sha256=tensor_sha(y),future_exact=True,gap_exact=True,next_exact=True,length_unchanged=True,
                                edited_block_rms=float((y[:,l:u]-x[:,l:u]).square().mean().sqrt())))
                    xb=torch.stack(batch).cuda();mb=torch.stack(masks).cuda()
                    # Independently use the original preprocessing, including native
                    # short-pair padding to 512. These tensors must be identical.
                    native_x,native_m,native_ids=model.preprocessing([z[:,:r['N']] for z,(r,arm) in zip(batch,index)])
                    branch=0 if T==512 else 1
                    assert native_ids[branch]==list(range(len(index)))
                    assert torch.equal(xb,native_x[branch]) and torch.equal(mb,native_m[branch]),'NATIVE_PREPROCESSING_MISMATCH'
                    with torch.inference_mode():
                        y=model._mask_vrd(xb,mb);calls+=1;assert torch.equal(y['output_mask'],mb)
                        if not noopdone:
                            yy=model._mask_vrd(native_x[branch],native_m[branch]);calls+=1
                            for k in ['pred_logits','pred_masks','output_mask']:assert torch.equal(y[k],yy[k]),('NOOP_NOT_EXACT',k)
                            del yy;noopdone=True
                        cl=y['pred_logits'].cpu().numpy();ml=y['pred_masks'].cpu().numpy()
                        bits=(y['pred_masks'].sigmoid()>.5).cpu().numpy()
                        tops=(y['pred_logits'].softmax(-1)[...,1:].topk(6,dim=-1).indices+1).cpu().numpy()
                    assert np.isfinite(cl).all() and np.isfinite(ml).all()
                    for i,(r,arm) in enumerate(index):
                        np.savez_compressed(out/'raw'/f"{r['case_id']}_{arm}.npz",pred_logits=cl[i],pred_masks=ml[i,:,:r['N']],bits=bits[i,:,:r['N']],top6=tops[i])
                    print('INFER073',full,'cases',len(cases),'seconds',round(time.monotonic()-start,2),flush=True)
                    del xb,mb,y,batch,masks,native_x,native_m
            del inp
        torch.cuda.synchronize();assert state_sha(model)==before
        dump(out/'input_audits.json',audits)
        dump(out/'raw_seal.json',dict(complete=True,raw_sha256={p.name:sha(p) for p in sorted((out/'raw').glob('*.npz'))},selection_sha256=sha(planpath),
            model_state_sha256=before,model_state_unchanged=True,noop_exact=noopdone,native_preprocessing_exact_all_batches=True,forward_calls=calls,training_steps=0,GPU_name=torch.cuda.get_device_name(0),elapsed_seconds=time.monotonic()-start))
        print('INFER073_COMPLETE',calls,'calls',time.monotonic()-start,flush=True)
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer']);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.out)
    else:infer(a.plan,a.out)
