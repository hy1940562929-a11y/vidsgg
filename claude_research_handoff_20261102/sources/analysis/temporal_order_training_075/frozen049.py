"""049 bounded fixed-candidate learned readout experiment; GT-free inference."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import sys,time,signal,json,copy,pickle,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from readout import EndpointProjection,project,best_interval,endpoint_ce
from frozen043 import load,dump,sha,tree_sha,Capture
from frozen_fast043 import collect
from evaluate import evaluate,rebuild,canonical_sha

HERE=Path(__file__).resolve().parent;REPO=Path('/root/rebinding_project/VrdONE')
JOB=Path('/root/autodl-tmp/VrdONE-data/commander_boundary_projection_049')
CACHE=Path('/root/autodl-tmp/VrdONE-data/commander_mask_training_043/results_v1')
STEPS=500

def matched(model,record):
    result=[];n=sum(len(x) for x in record['preds'])
    with torch.no_grad():
        for l in range(4):
            ix,valid=model.bipartite_match(record['cls'][l],record['preds'],record['masks'][l],record['targets'],None,record['valid'])
            b,q=model._get_src_permutation_idx(ix);y=torch.cat([gt[J] for gt,(_,J) in zip(record['targets'],ix)])
            assert len(y)==n and (y.sum(-1)>0).all()
            grid=torch.arange(y.shape[-1],device=y.device)[None,:];first=torch.where(y>0,grid,y.shape[-1]).min(-1).values;last=torch.where(y>0,grid,-1).max(-1).values
            result.append(dict(b=b,q=q,y=y,valid=valid,first=first,last=last,indices=ix,N=n))
    return result

def mask_loss(model,values,record,check=False):
    from models.losses import masked_dice_loss
    total=values.sum()*0
    for l,z in enumerate(record['match']):
        x=values[l,z['b'],z['q']];y=z['y'];v=z['valid'];nt=x.new_tensor(float(z['N']))
        prob=x.sigmoid();pt=prob*y+(1-prob)*(1-y)
        fl=(F.binary_cross_entropy_with_logits(x,y,reduction='none')*(1-pt)**2*(.25*y+.75*(1-y))*v).mean(-1).sum()/nt
        di=masked_dice_loss(x,y,nt,v);loss=model.loss_factor['loss_mask']*fl+model.loss_factor['loss_dice']*di
        if check:
            original=model.loss_masks(record['cls'][l],values[l],record['preds'],record['targets'],None,z['indices'],nt,v)
            assert torch.allclose(loss,original['loss_mask']+original['loss_dice'],atol=1e-6,rtol=1e-6)
        total=total+loss
    return total

def boundary_loss(values,record):
    s,e=values;terms=[]
    for l,z in enumerate(record['match']):
        terms.append(endpoint_ce(s[l,z['b'],z['q']],e[l,z['b'],z['q']],z['first'],z['last']).sum()/z['N'])
    return torch.stack(terms).sum()

def cuda_record(r):
    return {k:([v.cuda() for v in x] if k in ('preds','targets') else x.cuda() if torch.is_tensor(x) else x) for k,x in r.items()}

def prepare_cache(model,inventory,cohorts,out,sources):
    caches={};audit=[];video_sets={}
    for role in ('train','adapter_dev'):
        p=CACHE/(role+'_cache.pt');assert sha(p)==inventory[role]['sha256'];sources[str(p)]=sha(p)
        cpu=torch.load(p,map_location='cpu',weights_only=False);assert len(cpu)==inventory[role]['records'];records=[];videos=set()
        for ri,rr in enumerate(cpu):
            r=cuda_record(rr);del rr
            with torch.no_grad():base=project(model.predictor.mask_embed,r).masked_fill(~r['valid'][None,...],-10.)
            assert torch.equal(base,r['masks']),'049 cached original projection changed'
            r['match']=matched(model,r);records.append(r);videos.update(x['video_id'] for x in r['metadata'])
            audit.append(dict(role=role,batch=ri,N_pairs=len(r['targets']),N_GT=sum(len(x) for x in r['targets']),exact_projection=True,
              layer_matches=[dict(b=z['b'].cpu().tolist(),q=z['q'].cpu().tolist(),first=z['first'].cpu().tolist(),last=z['last'].cpu().tolist()) for z in r['match']]))
        assert sum(len(x['targets']) for x in records)==inventory[role]['N_pairs']
        caches[role]=records;video_sets[role]=videos;del cpu
    assert not video_sets['train']&video_sets['adapter_dev']
    val=set(n for c in cohorts.values() for n in load(Path(c['export'])/'manifest.json')['video_names'])
    assert not val&(video_sets['train']|video_sets['adapter_dev'])
    dump(out/'training_cache_audit.json',dict(records=audit,video_names={k:sorted(v) for k,v in video_sets.items()}))
    return caches

def masks_intervals(x,valid):
    bits=(x.sigmoid()>.5)&valid;grid=torch.arange(x.shape[-1],device=x.device)
    first=torch.where(bits,grid,x.shape[-1]).min(-1).values;last=torch.where(bits,grid,-1).max(-1).values
    return first,last

def diagnostics(model,heads,records):
    rows=[]
    with torch.no_grad():
        for ri,r in enumerate(records):
            z=r['match'][-1];b,q=z['b'],z['q'];valid=r['valid'][b,0];anchor=project(model.predictor.mask_embed,r)[-1,b,q]
            af,ae=masks_intervals(anchor,valid);ms=project(heads['mask'],r)[-1,b,q];mf,me=masks_intervals(ms,valid);fallback=me<=mf;mf=torch.where(fallback,af,mf);me=torch.where(fallback,ae,me)
            ss,ee=heads['boundary'](r);bf,be=best_interval(ss[-1,b,q],ee[-1,b,q],valid)
            for j in range(len(b)):
                g0,g1=int(z['first'][j]),int(z['last'][j]);values={}
                for arm,(first,last) in dict(baseline=(af,ae),mask=(mf,me),boundary=(bf,be)).items():
                    a,e=int(first[j]),int(last[j]);inter=max(0,min(e,g1)-max(a,g0)+1);union=max(e,g1)-min(a,g0)+1
                    values[arm]=dict(interval=[a,e],visible_sample_tIoU=inter/union if e>a else 0.)
                rows.append(dict(batch=ri,batch_row=int(b[j]),query=int(q[j]),GT_visible_interval=[g0,g1],singleton_GT=g0==g1,arms=values))
    return rows

def train(model,heads,caches,out):
    opt={a:torch.optim.AdamW(h.parameters(),lr=1e-5,weight_decay=.05) for a,h in heads.items()};rng=np.random.default_rng(49);order=[]
    while len(order)<STEPS:order+=rng.permutation(len(caches['train'])).tolist()
    order=order[:STEPS]
    dump(out/'training_schedule.json',dict(seed=49,steps=STEPS,order=order,lr=1e-5,weight_decay=.05,clip_norm=1.,parameter_count={a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()}))
    for a,h in heads.items():torch.save(h.state_dict(),out/(a+'_initial.pt'))
    dump(out/'dev_initial.json.gz',diagnostics(model,heads,caches['adapter_dev']))
    r=caches['train'][0];mask_loss(model,project(heads['mask'],r).masked_fill(~r['valid'][None,...],-10.),r,True)
    history=[];start=time.monotonic()
    for step,ri in enumerate(order,1):
        r=caches['train'][ri]
        for arm,h in heads.items():
            opt[arm].zero_grad(set_to_none=True)
            loss=mask_loss(model,project(h,r).masked_fill(~r['valid'][None,...],-10.),r) if arm=='mask' else boundary_loss(h(r),r)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.)
            assert torch.isfinite(norm);opt[arm].step();history.append(dict(step=step,batch=ri,arm=arm,loss=float(loss.detach()),grad_norm_before_clip=float(norm)))
        if step%50==0:dump(out/'training_history.json',history);print('TRAIN049',step,round(time.monotonic()-start,2),flush=True)
    for arm,h in heads.items():torch.save(dict(state=h.state_dict(),step=STEPS,seed=49,arm=arm),out/(arm+'_step500.pt'))
    dump(out/'dev_final.json.gz',diagnostics(model,heads,caches['adapter_dev']))

def source_video(ex,manifest,full,sources):
    data={}
    for n in ('geometry.json.gz','selected.json.gz','baseline.json.gz'):
        p=ex/'videos'/full/n;h=sha(p);assert h==manifest['outputs'][str(p.relative_to(ex))]['sha256'];sources[str(p)]=h;data[n]=load(p)
    return data['geometry.json.gz'],data['selected.json.gz']['selected'],data['baseline.json.gz']

def forward_video(model,ds,utils,ex,manifest,full,sources):
    g,sel,base=source_video(ex,manifest,full,sources)
    if not base:return g,sel,base,[],[]
    ref=manifest['source_files']['cache:'+full];p=Path(ref['path']);assert sha(p)==ref['sha256'];sources[str(p)]=ref['sha256']
    with p.open('rb') as f:raw=pickle.load(f)
    inp=ds._val_getitem(raw);del raw;inp['video_name']=full
    assert len(inp['sids'])==len(g['pairs'])
    for p in g['pairs']:
        i=p['pair_idx'];assert int(inp['sids'][i])==p['sid'] and int(inp['oids'][i])==p['oid']
        assert inp['so_features_list'][i].shape[-1]==p['N_samples'] and int(inp['so_offset'][i])==p['offset']
    gpu=utils.dict_to_device(inp,'cuda:0');del inp
    records,mapping=collect(model,gpu,Capture);del gpu
    positions={pi:(r,row) for r,ids in zip(records,mapping) for row,pi in enumerate(ids)}
    for c,b in zip(sel,base):
        p=g['pairs'][c['pair_idx']];r,row=positions[c['pair_idx']];n=p['N_samples'];x=r['masks'][-1,row,c['query'],:n]
        inds=torch.nonzero(x.sigmoid()>.5).flatten();assert len(inds)>0
        d=[p['pair_start']+p['offset']+int(inds[0])*4,p['pair_start']+p['offset']+int(inds[-1])*4+1]
        assert d==c['duration']==b['duration'],'Current source mask hull differs from frozen baseline'
    assert rebuild(base,sel,g,[b['duration'] for b in base])==base,'Original trajectory reconstruction mismatch'
    return g,sel,base,records,mapping

def predict_video(model,heads,data):
    g,sel,base,records,mapping=data;preds={a:[] for a in ('baseline','mask','boundary')};lookup={};fallbacks=0
    with torch.no_grad():
        for r,ids in zip(records,mapping):
            mask=project(heads['mask'],r)[-1];start,end=heads['boundary'](r);start,end=start[-1],end[-1]
            for row,pi in enumerate(ids):
                p=g['pairs'][pi];n=p['N_samples']
                if n<2:continue
                valid=r['valid'][row].expand(9,-1);mf,me=masks_intervals(mask[row],valid);bf,be=best_interval(start[row],end[row],valid)
                lookup[pi]=(mf.cpu().tolist(),me.cpu().tolist(),bf.cpu().tolist(),be.cpu().tolist())
        for c,b in zip(sel,base):
            pi,q=c['pair_idx'],c['query'];p=g['pairs'][pi];mf,me,bf,be=lookup[pi];origin=p['pair_start']+p['offset']
            md=[origin+4*mf[q],origin+4*me[q]+1]
            if me[q]<=mf[q]:md=b['duration'];fallbacks+=1
            bd=[origin+4*bf[q],origin+4*be[q]+1]
            for arm,d in [('baseline',b['duration']),('mask',md),('boundary',bd)]:preds[arm].append(d)
    hashes={a:canonical_sha(rebuild(base,sel,g,ds)) for a,ds in preds.items()}
    assert hashes['baseline']==canonical_sha(base)
    return dict(N_candidates=len(base),durations=preds,canonical_prediction_sha256=hashes,mask_fallback_candidates=fallbacks,source_forward_batches=len(records))

def run():
    start=time.monotonic();phase='SOURCE_BINDING';out=JOB/'results_v1';out.mkdir(exist_ok=False);sources={}
    def timeout(*_):raise TimeoutError('049 1800second total budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        for n,h in load(HERE/'software_freeze.json').items():assert sha(HERE/n)==h,n
        cohorts=load(HERE/'cohorts.json');contract=load(HERE/'source_contract043.json');inventory=load(HERE/'CACHE_INVENTORY.json')
        for rel,h in contract['repo_sources'].items():assert sha(REPO/rel)==h;sources[str(REPO/rel)]=h
        cp=REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==contract['checkpoint_sha256'];sources[str(cp)]=sha(cp)
        names=[];manifests={}
        for k,c in cohorts.items():
            ex=Path(c['export']);assert sha(ex/'manifest.json')==c['manifest_sha256'];manifests[k]=load(ex/'manifest.json');names.extend(manifests[k]['video_names'])
        assert len(names)==len(set(names))==128
        cfg=load(Path(cohorts['047']['export'])/'effective_config.json');assert sha(Path(cohorts['047']['export'])/'effective_config.json')==manifests['047']['outputs']['effective_config.json']['sha256']
        os.chdir(REPO);sys.path.insert(0,str(REPO));from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        import utils.misc as utils
        utils.set_seed(49);torch.set_num_threads(2);model=MaskVRD(cfg['model_config'],'cuda:0').eval();model._config_eval(cfg['inference_config'])
        weights=torch.load(cp,map_location='cpu',weights_only=False)[contract['checkpoint_key']];model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True);del weights
        model.cuda().requires_grad_(False);before=tree_sha(model.state_dict())
        heads={'mask':copy.deepcopy(model.predictor.mask_embed).requires_grad_(True),'boundary':EndpointProjection(model.predictor.mask_embed)}
        ds=VidOR.__new__(VidOR)
        for k in ('with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset'):setattr(ds,k,cfg['dataset_config'][k])
        phase='ZERO_UPDATE_NATIVE_GATE';first_name=manifests['047']['video_names'][0]
        first=forward_video(model,ds,utils,Path(cohorts['047']['export']),manifests['047'],first_name,sources)
        assert first[2],'Frozen first video is empty; select protocol update before experiment'
        dump(out/'zero_update_gate.json',dict(video_name=first_name,source_hulls_exact=True,N_candidates=len(first[2]),original_trajectory_reconstruction_exact=True,model_state_sha256=before))
        phase='CACHED_TRAIN_MATCHING';caches=prepare_cache(model,inventory,cohorts,out,sources)
        phase='TRAIN';train(model,heads,caches,out);del caches
        assert tree_sha(model.state_dict())==before
        phase='NATIVE_INFERENCE';audits=[]
        for k,c in cohorts.items():
            ex=Path(c['export']);m=manifests[k]
            for i,full in enumerate(m['video_names']):
                data=first if k=='047' and i==0 else forward_video(model,ds,utils,ex,m,full,sources)
                z=predict_video(model,heads,data);z.update(video_name=full,cohort=k);dump(out/'intervals'/k/(full+'.json.gz'),z)
                audits.append(dict(cohort=k,video_name=full,N_candidates=z['N_candidates'],forward_batches=z['source_forward_batches'],fallback_candidates=z['mask_fallback_candidates']))
                if k=='047' and i==0:first=None
                del data,z
                if (i+1)%8==0:print('NATIVE049',k,i+1,'elapsed',round(time.monotonic()-start,1),flush=True)
        assert tree_sha(model.state_dict())==before
        seal={str(p.relative_to(out)):sha(p) for p in (out/'intervals').rglob('*.gz')};assert len(seal)==128;dump(out/'intervals_seal_before_GT.json',seal);dump(out/'native_audits.json',audits)
        phase='OFFICIAL_EVALUATION';result=evaluate(out,cohorts)
        result.update(complete=True,elapsed_seconds=time.monotonic()-start,training_steps_per_arm=STEPS,seed=49,model_forwards_batches=sum(r['forward_batches'] for r in audits),source_sha256=sources,
          original_model_state_unchanged=True,scope='Small fixed-candidate two-head intervention;128 exposed validation videos;not independent generalization or novelty',independent_review='PENDING')
        dump(out/'summary.json',result);dump(out/'status.json',dict(complete=True,decision=result['decision']));print('COMPLETE049',result['decision'],flush=True)
    except Exception as e:
        dump(out/'status.json',dict(complete=False,phase=phase,error=repr(e),elapsed_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)

if __name__=='__main__':run()
