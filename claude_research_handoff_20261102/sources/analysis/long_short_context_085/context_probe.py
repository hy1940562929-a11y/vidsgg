#!/usr/bin/env python3
"""085 annotation-conditioned frozen-model context sensitivity; never training."""
import argparse, hashlib, json, math, os, signal, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
ARMS=('baseline','noop','editL','editN')

def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def ah(x):
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()
def win(bits):
    return np.flatnonzero(np.convolve(bits.astype(int),np.ones(8,dtype=int),'valid')==8).tolist() if len(bits)>=8 else []
def ident(s):return hashlib.sha256(s.encode()).hexdigest()[:20]
def overlap_cells(tt,s,e):return (tt<e)&(tt+4>s)
def rms(a,b):return float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))
def choose(x,tt,target,longspan,all_long,all_target,fps):
    """First feasible triple in fixed geometric order; no prediction dependence."""
    s,e=target;protect=overlap_cells(tt,s-.5*fps,e+.5*fps)
    lbits=(tt>=longspan[0])&(tt+4<=longspan[1])&~protect
    for a,b in all_target:lbits &= ~overlap_cells(tt,a,b)
    excluded=protect.copy()
    for a,b in all_long+all_target:excluded|=overlap_cells(tt,a,b)
    ls=win(lbits);ns=win(~excluded)
    if not ls:return None,'NO_L_WINDOW'
    if len(ns)<2:return None,'NO_N_DONOR_WINDOWS'
    # Window distance is nearest edge distance to target; both edits same side.
    def pos(k):
        a,b=tt[k],tt[k+7]+4
        return (-1,s-b) if b<=s else (1,a-e)
    pairs=[]
    for l in ls:
        sl,dl=pos(l)
        for n in ns:
            sn,dn=pos(n)
            if sl==sn and abs(dl-dn)<=2*fps:pairs.append((abs(dl-dn),max(dl,dn),l,n))
    if not pairs:return None,'NO_SAME_SIDE_DISTANCE_MATCH_2S'
    # Donor pool is every eligible 8-token window; donor cannot overlap either edit.
    for diff,dist,l,n in sorted(pairs):
        for d in sorted(ns,key=lambda d:(abs(d-n),d)):
            if abs(d-n)<8 or abs(d-l)<8:continue
            rl=rms(x[:2048,l:l+8],x[:2048,d:d+8]);rn=rms(x[:2048,n:n+8],x[:2048,d:d+8])
            if rl>1e-8 and rn>1e-8 and .5<=rl/rn<=2:
                return dict(L=l,N_start=n,donor=d,L_rms=rl,N_rms=rn,RMS_ratio_L_over_N=rl/rn,
                            distance_difference_seconds=diff/fps,L_distance_seconds=pos(l)[1]/fps,N_distance_seconds=pos(n)[1]/fps,side=pos(l)[0],protected_indices=np.flatnonzero(protect).tolist()),None
    return None,'NO_RMS_MATCHED_NONOVERLAP_DONOR'

def helper_load(path):
    sys.path.insert(0,str(Path(path).resolve()));import history_probe as h
    return h

def prepare(args):
    t=time.monotonic();out=args.out;out.mkdir(parents=True,exist_ok=False)
    h=helper_load(args.helper);torch,ds,cfg,helper,pids=h.setup()
    videos=sorted(p.stem for p in helper.GT_PROP.glob('*.pkl'));assert len(videos)==27
    candidates=[];excluded=[];sources={str(Path(h.__file__)):sha(h.__file__),str(Path(helper.__file__)):sha(helper.__file__),str(h.CFG):sha(h.CFG),str(Path(__file__).resolve()):sha(__file__)}
    for v in videos:
        ann,inp,tids,ap=h.video_input(v,ds,helper);fps=float(ann['fps']);rels=defaultdict(list)
        sources[str(ap)]=sha(ap);p=helper.GT_PROP/(v+'.pkl');sources[str(p)]=sha(p)
        for gi,r in enumerate(ann['relation_instances']):
            rels[(int(r['subject_tid']),int(r['object_tid']))].append((r['predicate'],int(r['begin_fid']),int(r['end_fid']),gi))
        for pi,(si,oi) in enumerate(zip(inp['sids'].tolist(),inp['oids'].tolist())):
            a=max(int(inp['traj_durations'][si,0]),int(inp['traj_durations'][oi,0]));b=min(int(inp['traj_durations'][si,1]),int(inp['traj_durations'][oi,1]))
            if (b-a)/fps<10:continue
            rr=rels[(int(tids[si]),int(tids[oi]))];x=inp['so_features_list'][pi].numpy();N=x.shape[-1];tt=a+int(inp['so_offset'][pi])+4*np.arange(N)
            assert x.shape[0]==2069
            longs=[(p,s,e,j) for p,s,e,j in rr if (e-s)/fps>=3]
            for pred,s,e,gi in rr:
                if not (0<(e-s)/fps<3 and a<=s<e<=b):continue
                cid=ident(f'{v}:{tids[si]}:{tids[oi]}:{pred}:{s}:{e}')
                base=dict(case_id=cid,video=v,pair_idx=pi,subject_tid=int(tids[si]),object_tid=int(tids[oi]),predicate=pred,predicate_id=int(pids[pred]),gt_index=gi,target=[s,e],fps=fps,pair_support=[a,b],origin=int(tt[0]),N=N)
                covers=[r for r in longs if r[0]!=pred and max(0,min(e,r[2])-max(s,r[1]))/(e-s)>=.5]
                reason='NO_DIFFERENT_LONG_COVERING_HALF'
                if N>2048:reason='PAIR_OVER2048';covers=[]
                if not ((tt>=s)&(tt<e)).any():reason='NO_TARGET_TOKEN';covers=[]
                choices=[]
                for lp,ls,le,lj in sorted(covers,key=lambda r:(r[0],r[1],r[2])):
                    c,reason=choose(x,tt,(s,e),(ls,le),[(u,w) for _,u,w,_ in longs],[(u,w) for p,u,w,_ in rr if p==pred],fps)
                    if c:
                        boundary=np.zeros(N,dtype=bool);boundary[c['protected_indices']]=True;boundary &= (tt+4<=s)|(tt>=e)
                        if not boundary.any():reason='NO_UNCHANGED_BOUNDARY_TOKEN';continue
                        choices.append(dict(**base,**c,long_predicate=lp,long_predicate_id=int(pids[lp]),long_target=[ls,le],long_gt_index=lj,feature_sha256=ah(x)));break
                if choices:candidates.extend(choices)
                else:excluded.append(dict(**base,reason=reason))
        print('CENSUS085',v,'eligible',len(candidates),'excluded',len(excluded),flush=True)
    # Deduplicate duplicate annotation entries before quotas.
    unique={}
    for r in sorted(candidates,key=lambda r:(r['case_id'],r['pair_idx'])):unique.setdefault(r['case_id'],r)
    selected=[];counts=Counter()
    for r in sorted(unique.values(),key=lambda r:r['case_id']):
        if len(selected)<64 and counts[r['video']]<4:selected.append(r);counts[r['video']]+=1
        else:excluded.append(dict(case_id=r['case_id'],video=r['video'],reason='STABLE_HASH_QUOTA'))
    for p in ['models/maskvrd.py','models/predictor.py','dataloaders/vidor.py']:
        q=h.REPO/p;sources[str(q)]=sha(q)
    plan=dict(kind='ANNOTATION_CONDITIONED_CONTEXT_SENSITIVITY',complete=True,cases=selected,eligible_count=len(unique),selected_count=len(selected),exclusions=dict(Counter(r['reason'] for r in excluded)),sources=sources,checkpoint_sha256=h.CHECKPOINT_SHA,selection_uses_model_outputs=False,distance_limit_seconds=2,elapsed_seconds=time.monotonic()-t)
    dump(out/'selection.json',plan);dump(out/'exclusions.json',excluded);dump(out/'all_eligible.json',list(unique.values()));dump(out/'effective_config.json',cfg)
    print('CENSUS085_COMPLETE',json.dumps({k:plan[k] for k in ['eligible_count','selected_count','exclusions']}),flush=True)

def softmax(x):
    a=np.exp(x-x.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
def sigmoid(x):return np.exp(-np.logaddexp(0,-x))
def spans(bits,tt):
    result=[]
    for row in bits:
        idx=np.flatnonzero(row)
        # Native min_pred_len=5, first/last-positive envelope; last frame + 1.
        if len(idx)==0:result.append(None);continue
        a,b=int(tt[idx[0]]),int(tt[idx[-1]]+1)
        result.append([a,b] if b-a>=5 else None)
    return result
def iou(span,target):
    if span is None:return 0.
    inter=max(0,min(span[1],target[1])-max(span[0],target[0]));return inter/(max(span[1],target[1])-min(span[0],target[0]))
def metrics(m,c,bits,r,q,lq):
    tt=r['origin']+4*np.arange(r['N']);inside=(tt>=r['target'][0])&(tt<r['target'][1]);outside=np.zeros(r['N'],dtype=bool);outside[r['protected_indices']]=True;outside &= (tt+4<=r['target'][0])|(tt>=r['target'][1]);assert outside.any()
    pr=softmax(c);sp=spans(bits,tt);ii=[iou(z,r['target']) for z in sp];lp=r['long_predicate_id'];L=r['L']
    return dict(query=q,target_mask_logit=float(m[q,inside].mean()),target_mask_probability=float(sigmoid(m[q,inside]).mean()),target_vs_outside_margin=float(m[q,inside].mean()-m[q,outside].mean()),target_class_probability=float(pr[q,r['predicate_id']]),decoded_interval=sp[q],tiou=ii[q],hit50=ii[q]>=.5,
      gt_time_any_query_oracle_tiou=max(ii),gt_time_any_query_oracle_query=int(np.argmax(ii)),
      long_query=lq,long_class_probability=float(pr[lq,lp]),long_L_mask_logit=float(m[lq,L:L+8].mean()),long_L_mask_probability=float(sigmoid(m[lq,L:L+8]).mean()))

def infer(args):
    t=time.monotonic();args.out.mkdir(parents=True,exist_ok=False);(args.out/'raw').mkdir();plan=json.loads(args.plan.read_text())
    def timeout(*_):raise TimeoutError('085 wall limit 900 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    h=helper_load(args.helper);torch,ds,cfg,helper,pids=h.setup()
    for p,digest in plan['sources'].items():assert sha(p)==digest,('SOURCE_CHANGED',p)
    cp=h.REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==plan['checkpoint_sha256']
    from models.maskvrd import MaskVRD
    import utils.misc as misc
    misc.set_seed(42);model=MaskVRD(cfg['model_config'],device='cuda:0').eval();model._config_eval(cfg['inference_config'])
    assert model.feat_stride==4 and model.pred_min_frames==5, 'NATIVE_DECODER_CONFIG_MISMATCH'
    weights=torch.load(cp,map_location='cpu',weights_only=False)['model_state_dict_ema'];model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True);del weights
    model.cuda();before=h.state_sha(model);rows=[];audits=[];group=defaultdict(list)
    for r in plan['cases']:group[r['video']].append(r)
    try:
        for v,events in group.items():
            ann,inp,tids,ap=h.video_input(v,ds,helper)
            for r in events:
                x=inp['so_features_list'][r['pair_idx']];assert ah(x.numpy())==r['feature_sha256'];results={};raws={}
                for arm in ARMS:
                    y=x.clone()
                    if arm in ('editL','editN'):
                        k=r['L' if arm=='editL' else 'N_start'];d=r['donor'];y[:2048,k:k+8]=x[:2048,d:d+8]
                        keep=np.ones(r['N'],dtype=bool);keep[k:k+8]=False;assert torch.equal(y[:,keep],x[:,keep])
                    assert torch.equal(y[2048:],x[2048:]);assert torch.equal(y[:,r['protected_indices']],x[:,r['protected_indices']])
                    xx,mm,ids=model.preprocessing([y]);branches=[i for i in range(2) if xx[i] is not None];assert len(branches)==1;k=branches[0];assert ids[k]==[0]
                    with torch.inference_mode():raw=model._mask_vrd(xx[k],mm[k])
                    assert torch.equal(raw['output_mask'],mm[k]);assert int(raw['output_mask'].sum())==r['N']
                    masks=raw['pred_masks'][0,:,:r['N']];m=masks.cpu().numpy();c=raw['pred_logits'][0].cpu().numpy();bits=(masks.sigmoid()>.5).cpu().numpy();assert np.isfinite(m).all() and np.isfinite(c).all()
                    if arm=='baseline':
                        probs=softmax(c);q=int(probs[:,r['predicate_id']].argmax());lq=int(probs[:,r['long_predicate_id']].argmax())
                    raws[arm]=(m.copy(),c.copy(),bits.copy());results[arm]=metrics(m,c,bits,r,q,lq)
                    np.savez_compressed(args.out/'raw'/f"{r['case_id']}_{arm}.npz",pred_masks=m,pred_logits=c,binary=bits)
                    audits.append(dict(case_id=r['case_id'],arm=arm,input_sha256=ah(y.numpy()),output_sha256=hashlib.sha256((ah(m)+ah(c)+ah(bits)).encode()).hexdigest(),protected_exact=True,geometry_exact=True,length=r['N']))
                    del raw,xx,mm
                for a,b in zip(raws['baseline'],raws['noop']):assert np.array_equal(a,b),'NOOP_NOT_EXACT'
                row=dict(**r,arms=results,noop_exact=True,delta={})
                for arm in ('editL','editN'):
                    row['delta'][arm]={k:results[arm][k]-results['baseline'][k] for k in ('target_mask_logit','target_mask_probability','target_vs_outside_margin','target_class_probability','tiou','long_class_probability','long_L_mask_logit','long_L_mask_probability')}
                    row['delta'][arm]['gain50']=int(results[arm]['hit50'] and not results['baseline']['hit50']);row['delta'][arm]['loss50']=int(results['baseline']['hit50'] and not results[arm]['hit50'])
                row['paired_editL_minus_editN']={k:row['delta']['editL'][k]-row['delta']['editN'][k] for k in row['delta']['editL']};rows.append(row)
                dump(args.out/'per_event.json',rows);dump(args.out/'input_audits.json',audits)
                print('INFER085',len(rows),'of',len(plan['cases']),'seconds',round(time.monotonic()-t,2),flush=True)
        assert h.state_sha(model)==before
        summary=dict(complete=True,N=len(rows),noop_exact_all=all(r['noop_exact'] for r in rows),model_state_sha256=before,model_state_unchanged=True,checkpoint_sha256=sha(cp),selection_sha256=sha(args.plan),raw_sha256={p.name:sha(p) for p in sorted((args.out/'raw').glob('*.npz'))},training_steps=0,microbatch=1,elapsed_seconds=time.monotonic()-t,semantic_deletion_verified=False)
        summary['mean_paired_editL_minus_editN']={k:float(np.mean([r['paired_editL_minus_editN'][k] for r in rows])) for k in rows[0]['paired_editL_minus_editN']} if rows else {}
        byvideo=defaultdict(list)
        for r in rows:byvideo[r['video']].append(r)
        vm=[float(np.mean([r['paired_editL_minus_editN']['target_mask_probability'] for r in rr])) for rr in byvideo.values()]
        vb=[float(np.mean([r['delta']['editL']['target_mask_probability'] for r in rr])) for rr in byvideo.values()]
        if vm:
            rng=np.random.default_rng(85);boot=np.asarray(vm)[rng.integers(0,len(vm),size=(10000,len(vm)))].mean(1)
            ci=np.quantile(boot,[.025,.975]).tolist();effect=float(np.mean(vm));baseline_effect=float(np.mean(vb))
            summary['primary']=dict(video_count=len(vm),video_equal_mean_mask_probability_editL_minus_editN=effect,video_bootstrap95=ci,editL_minus_baseline_video_equal=baseline_effect,support_threshold=.01,supported=len(vm)>=10 and effect>=.01 and ci[0]>0 and baseline_effect>0,uncertainty='annotation-conditioned perturbation; does not establish semantic deletion')
        else:summary['primary']=dict(video_count=0,supported=False,uncertainty='no feasible events')
        summary['distance_difference_seconds']=[r['distance_difference_seconds'] for r in rows]
        dump(args.out/'summary.json',summary)
    except BaseException as e:
        dump(args.out/'failure.json',dict(complete=False,error=repr(e),completed_events=len(rows),elapsed_seconds=time.monotonic()-t));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer']);p.add_argument('--helper',type=Path,default=Path('/root/autodl-tmp/VrdONE-data/commander_event_support_073/code_v2'));p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path);a=p.parse_args()
    a.out=a.out.resolve();a.plan=a.plan.resolve() if a.plan else None;a.helper=a.helper.resolve()
    if a.phase=='prepare':prepare(a)
    else:infer(a)
