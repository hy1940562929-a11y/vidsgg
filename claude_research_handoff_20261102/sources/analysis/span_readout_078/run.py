"""078: bounded frozen-feature interval comparison; predictions sealed before GT."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import sys,time,copy,signal,math,collections
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frozen043 import load,dump,sha,tree_sha
from frozen049 import forward_video,mask_loss,masks_intervals
from inherited075 import prepare_cache,verify_retained_sources
from readout import project
from heads import SpanHead,target_span_loss
from evaluate import evaluate,rebuild,canonical_sha,ARMS

HERE=Path(__file__).resolve().parent
REPO=Path('/root/rebinding_project/VrdONE')
JOB=Path('/root/autodl-tmp/VrdONE-data/commander_span_readout_078')
SPAN_ARMS=('independent','joint','joint_hard','joint_random')
STEPS=500

def iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
    return inter/(a[1]-a[0]+b[1]-b[0]+2-inter)

def attach_contrasts(caches,out):
    rng=np.random.default_rng(78);audit=[]
    for role,records in caches.items():
        for ri,r in enumerate(records):
            m=r['match'][-1]; domains=[]
            targetids=[(b,int(j)) for b,(_,js) in enumerate(m['indices']) for j in js.tolist()]
            assert [b for b,j in targetids]==m['b'].tolist()
            for z,(b,j) in enumerate(targetids):
                first,last=int(m['first'][z]),int(m['last'][z]);gt=(first,last)
                n=int(r['valid'][b,0].sum());assert r['valid'][b,0,:n].all()
                meta=r['gt_metadata'][b][j];pid=int(r['preds'][b][j]);hard=[];random=[]
                if first<last and not meta['censored_pair']:
                    episodes=[]
                    for pred,y in zip(r['preds'][b],r['targets'][b]):
                        if int(pred)==pid:
                            ids=y.nonzero().flatten();episodes.append((int(ids[0]),int(ids[-1])))
                    quality=lambda d:max(iou(d,e) for e in episodes)
                    width=last-first+1;offsets=sorted({1,max(1,width//4),max(1,width//2)})
                    candidates=set()
                    for d in offsets:
                        candidates.update([(first-d,last),(first,last+d),(first-d,last+d),
                            (first+d,last),(first,last-d),(first-d,last-d),(first+d,last+d)])
                    for a,e in sorted(candidates):
                        if not 0<=a<e<n:continue
                        quality_h=quality((a,e))
                        if quality_h>=.75:continue
                        length=e-a+1
                        starts=[s for s in range(n-length+1) if quality((s,s+length-1))<.75]
                        if not starts:continue
                        s=int(rng.choice(starts));re=s+length-1;quality_r=quality((s,re))
                        hard.append((a,e,quality_h));random.append((s,re,quality_r))
                assert len(hard)==len(random)
                domains.append(dict(hard=hard,random=random))
                audit.append(dict(role=role,batch=ri,pair_row=b,query=int(m['q'][z]),
                    first=first,last=last,N_valid=n,singleton=first==last,
                    eligible_span=first<last,censored=meta['censored_pair'],hard=hard,random=random,**meta))
            r['contrast_domains']=domains
    dump(out/'contrast_domains.json.gz',audit)
    counts={role:dict(N_GT=sum(x['role']==role for x in audit),
        eligible_span=sum(x['role']==role and x['eligible_span'] for x in audit),
        singleton=sum(x['role']==role and x['singleton'] for x in audit),
        contrast_targets=sum(x['role']==role and bool(x['hard']) for x in audit),
        contrast_pairs=sum(len(x['hard']) for x in audit if x['role']==role)) for role in caches}
    dump(out/'target_support_audit.json',counts)
    assert counts['train']['eligible_span']==561 and counts['adapter_dev']['eligible_span']==168
    return counts

def contrast_loss(scores,m,domains,kind):
    terms=[]
    for n,d in enumerate(domains):
        candidates=d[kind]
        if not candidates:continue
        a,e=int(m['first'][n]),int(m['last'][n]);assert a<e
        ids=torch.tensor([[x[0],x[1]] for x in candidates],device=scores.device)
        quality=scores.new_tensor([x[2] for x in candidates])
        terms.append(F.relu((1-quality)-scores[n,a,e]+scores[n,ids[:,0],ids[:,1]]).mean())
    return torch.stack(terms).mean() if terms else scores.sum()*0

@torch.no_grad()
def diagnostics(model,heads,caches,out,stage):
    rows=[]
    for role,records in caches.items():
        for ri,r in enumerate(records):
            m=r['match'][-1];b,q=m['b'],m['q'];v=r['valid'][b,0]
            initial=masks_intervals(r['masks'][-1,b,q],v)
            decoded={'baseline':initial,'mask':masks_intervals(project(heads['mask'],r)[-1,b,q],v)}
            decoded.update({a:heads[a].decode(r,b,q) for a in SPAN_ARMS})
            targetids=[(bb,int(j)) for bb,(_,js) in enumerate(m['indices']) for j in js.tolist()]
            for k,(bb,j) in enumerate(targetids):
                gt=[int(m['first'][k]),int(m['last'][k])];values={}
                for arm,(starts,ends) in decoded.items():
                    d=[int(starts[k]),int(ends[k])]
                    values[arm]=dict(interval=d,sampled_tIoU=iou(d,gt) if 0<=d[0]<d[1] else 0.)
                rows.append(dict(role=role,batch=ri,pair_row=bb,query=int(q[k]),GT_interval=gt,
                    singleton=gt[0]==gt[1],native_anchor_fallback=not (0<=int(initial[0][k])<int(initial[1][k])),**r['gt_metadata'][bb][j],arms=values))
    dump(out/('diagnostics_'+stage+'.json.gz'),rows)
    result={}
    for role in caches:
        result[role]={}
        for group,fn in {'all':lambda r:True,'short':lambda r:r['relation_sec']<3,'long':lambda r:r['relation_sec']>=3}.items():
            rs=[r for r in rows if r['role']==role and fn(r)]
            result[role][group]=dict(N_GT=len(rs),arms={a:dict(
                matched_query_tIoU50=sum(r['arms'][a]['sampled_tIoU']>=.5 for r in rs),
                mean_tIoU=float(np.mean([r['arms'][a]['sampled_tIoU'] for r in rs])) if rs else None) for a in ARMS})
    dump(out/('diagnostic_summary_'+stage+'.json'),result)

def train(model,heads,caches,out):
    opts={a:torch.optim.AdamW(h.parameters(),lr=1e-5 if a=='mask' else 3e-4,weight_decay=.05) for a,h in heads.items()}
    rng=np.random.default_rng(78);order=[]
    while len(order)<STEPS:order.extend(rng.permutation(len(caches['train'])).tolist())
    order=order[:STEPS]
    dump(out/'training_schedule.json',dict(seed=78,steps=STEPS,order=order,
        lr={a:o.param_groups[0]['lr'] for a,o in opts.items()},contrast_weight=.1,
        parameter_count={a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()}))
    diagnostics(model,heads,caches,out,'initial');history=[];start=time.monotonic()
    for step,ri in enumerate(order,1):
        r=caches['train'][ri];m=r['match'][-1]
        for arm,h in heads.items():
            opts[arm].zero_grad(set_to_none=True)
            if arm=='mask':
                z=project(h,r).masked_fill(~r['valid'][None,...],-10.)
                ce=mask_loss(model,z,r);aux=ce*0;eligible=m['N']
            else:
                z=h.scores(r,m['b'],m['q'])
                ce,eligible=target_span_loss(z,m['first'],m['last'])
                aux=contrast_loss(z,m,r['contrast_domains'],'hard' if arm=='joint_hard' else 'random') if arm in ('joint_hard','joint_random') else ce*0
            loss=ce+.1*aux
            assert torch.isfinite(loss),(step,arm,loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.)
            assert torch.isfinite(norm);opts[arm].step()
            history.append(dict(step=step,batch=ri,arm=arm,loss=float(loss.detach()),
                main=float(ce.detach()),aux=float(aux.detach()),eligible=eligible,grad_norm=float(norm)))
        if step%50==0:
            dump(out/'training_history.json',history)
            print('TRAIN078',step,'elapsed',round(time.monotonic()-start,1),
                'loss', {x['arm']:round(x['loss'],4) for x in history[-len(heads):]},flush=True)
    for arm,h in heads.items():torch.save(dict(state=h.state_dict(),step=STEPS,seed=78,arm=arm),out/(arm+'_step500.pt'))
    diagnostics(model,heads,caches,out,'final')

@torch.no_grad()
def predict(heads,zerohead,data):
    g,sel,base,records,mapping=data
    lookup={a:{} for a in ARMS if a!='baseline'};fallback={a:0 for a in lookup};zero_count=0
    wanted=collections.defaultdict(set)
    for c in sel:wanted[c['pair_idx']].add(c['query'])
    for r,ids in zip(records,mapping):
        items=[(row,pi,q) for row,pi in enumerate(ids) for q in sorted(wanted[pi])]
        if not items:continue
        mask=project(heads['mask'],r)[-1]
        for offset in range(0,len(items),16):
            chunk=items[offset:offset+16]
            b=torch.tensor([x[0] for x in chunk],device='cuda');q=torch.tensor([x[2] for x in chunk],device='cuda')
            native=masks_intervals(r['masks'][-1,b,q],r['valid'][b,0])
            zz=zerohead.decode(r,b,q)
            assert all(torch.equal(x,y) for x,y in zip(native,zz)), 'Zero residual fails to reproduce native selected intervals'
            zero_count+=len(chunk)
            decoded={'mask':masks_intervals(mask[b,q],r['valid'][b,0])}
            decoded.update({a:heads[a].decode(r,b,q) for a in SPAN_ARMS})
            for a,(starts,ends) in decoded.items():
                for (row,pi,qi),first,last in zip(chunk,starts.tolist(),ends.tolist()):lookup[a][pi,qi]=(first,last)
    durations={a:[] for a in ARMS}
    for c,b in zip(sel,base):
        pi,q=c['pair_idx'],c['query'];p=g['pairs'][pi];origin=p['pair_start']+p['offset']
        durations['baseline'].append(b['duration'])
        for a in lookup:
            first,last=lookup[a][pi,q]
            d=[origin+4*first,origin+4*last+1]
            if not 0<=first<last<p['N_samples']:
                d=b['duration'];fallback[a]+=1
            durations[a].append(d)
    hashes={a:canonical_sha(rebuild(base,sel,g,d)) for a,d in durations.items()}
    assert hashes['baseline']==canonical_sha(base)
    return dict(N_candidates=len(base),durations=durations,canonical_prediction_sha256=hashes,
        fallback_candidates=fallback,zero_update_checked_pair_queries=zero_count,source_forward_batches=len(records))

def main():
    start=time.monotonic();out=JOB/'results_v1';out.mkdir(parents=True,exist_ok=False);sources={};phase='SOURCE_BINDING'
    def timeout(*args):raise TimeoutError('078 bounded 1800 second budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        for n,h in load(HERE/'software_freeze.json').items():assert sha(HERE/n)==h,n
        cohorts=load(HERE/'cohorts.json');contract=load(HERE/'source_contract043.json');inventory=load(HERE/'CACHE_INVENTORY.json')
        for rel,h in contract['repo_sources'].items():assert sha(REPO/rel)==h; sources[str(REPO/rel)]=h
        cp=REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==contract['checkpoint_sha256'];sources[str(cp)]=sha(cp)
        manifests={k:load(Path(c['export'])/'manifest.json') for k,c in cohorts.items()}
        for k,c in cohorts.items():assert sha(Path(c['export'])/'manifest.json')==c['manifest_sha256']
        cfgpath=Path(cohorts['047']['export'])/'effective_config.json';assert sha(cfgpath)==manifests['047']['outputs']['effective_config.json']['sha256'];cfg=load(cfgpath)
        os.chdir(REPO);sys.path.insert(0,str(REPO))
        from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        import utils.misc as utils
        utils.set_seed(78);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        model=MaskVRD(cfg['model_config'],'cuda:0').eval();model._config_eval(cfg['inference_config'])
        state=torch.load(cp,map_location='cpu',weights_only=False)[contract['checkpoint_key']]
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
        model.cuda().requires_grad_(False);before=tree_sha(model.state_dict())
        h0=SpanHead(mode='joint').cuda();zerohead=copy.deepcopy(h0).requires_grad_(False)
        heads={a:copy.deepcopy(h0) for a in SPAN_ARMS};heads['independent'].mode='independent'
        heads={'mask':copy.deepcopy(model.predictor.mask_embed).requires_grad_(True),**heads}
        init={a:tree_sha(heads[a].state_dict()) for a in SPAN_ARMS};assert len(set(init.values()))==1
        dump(out/'initial_heads.json',dict(sha256=init,common_zero_residual=True))
        ds=VidOR.__new__(VidOR)
        for key in ('with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset'):setattr(ds,key,cfg['dataset_config'][key])
        phase='ZERO_UPDATE_GATE';first_name=manifests['047']['video_names'][0]
        first=forward_video(model,ds,utils,Path(cohorts['047']['export']),manifests['047'],first_name,sources)
        zero=predict(heads,zerohead,first);assert len(set(zero['canonical_prediction_sha256'].values()))==1
        dump(out/'zero_update_gate.json',dict(video_name=first_name,all_six_arms_identical=True,**zero))
        print('ZERO_UPDATE078 PASS',zero['N_candidates'],flush=True)
        phase='TRAINING_CACHE';caches=prepare_cache(model,inventory,cohorts,out,sources)
        verify_retained_sources(caches,out,sources);support=attach_contrasts(caches,out)
        print('SUPPORT078',support,flush=True)
        phase='TRAIN';train(model,heads,caches,out);del caches
        assert tree_sha(model.state_dict())==before
        phase='INFERENCE';audits=[]
        for k,c in cohorts.items():
            for i,full in enumerate(manifests[k]['video_names']):
                data=first if k=='047' and i==0 else forward_video(model,ds,utils,Path(c['export']),manifests[k],full,sources)
                z=predict(heads,zerohead,data);z.update(video_name=full,cohort=k)
                dump(out/'intervals'/k/(full+'.json.gz'),z)
                audits.append(dict(cohort=k,video_name=full,N_candidates=z['N_candidates'],
                    zero_update_checked_pair_queries=z['zero_update_checked_pair_queries'],fallback=z['fallback_candidates']))
                if k=='047' and i==0:first=None
                del data,z
                if (i+1)%8==0:print('NATIVE078',k,i+1,'elapsed',round(time.monotonic()-start,1),flush=True)
        assert tree_sha(model.state_dict())==before and sha(cp)==contract['checkpoint_sha256']
        seal={str(p.relative_to(out)):sha(p) for p in (out/'intervals').rglob('*.gz')};assert len(seal)==128
        dump(out/'intervals_seal_before_GT.json',seal);dump(out/'native_audits.json',audits)
        phase='OFFICIAL_EVALUATION';result=evaluate(out,cohorts)
        result.update(complete=True,elapsed_seconds=time.monotonic()-start,steps=STEPS,seed=78,
            original_model_state_unchanged=True,source_sha256=sources,
            scope='Fixed selected candidates, frozen features, exposed 128-video validation subset; no full-benchmark/generalization claim',independent_review='PENDING')
        dump(out/'summary.json',result);dump(out/'status.json',dict(complete=True,decision=result['decision']))
        s=result['cohorts']['combined'];print('COMPLETE078',result['decision'],flush=True)
        for a in ARMS:
            print(a,'mAP',s['mAP'][a],'all',s['groups']['all']['arms'][a],
                'target',s['groups']['long_coexist_short']['arms'][a]['hits'],flush=True)
    except Exception as e:
        dump(out/'status.json',dict(complete=False,phase=phase,error=repr(e),elapsed_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
