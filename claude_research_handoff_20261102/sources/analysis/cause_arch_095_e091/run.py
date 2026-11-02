"""E091 frozen geometry contract. Sole experiment; no encoder, matching or class updates."""
import argparse, csv, hashlib, importlib.util, json, multiprocessing as mp
import os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def require(x,m):
    if not x:raise RuntimeError(m)
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

class Head(nn.Module):
    def __init__(self,state):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(256,256,1) for _ in range(3)])
        self.load_state_dict(state)
    def forward(self,hs,f):
        # Preserve the native four-layer batch and contraction layout. Only
        # layer 3 is returned/connected to the loss; no auxiliary supervision.
        B=f.shape[0]
        require(hs.shape==(4*B,256,9),'NATIVE_HS_LAYOUT')
        x=hs
        for i,l in enumerate(self.layers):
            x=l(x)
            if i<2:x=F.gelu(x)
        return torch.einsum('lbqc,bcm->lbqm',
                           x.view(4,B,256,9).permute(0,1,3,2),f)[3]

def orig_loss(z,y,v):
    # Exact original functions imported after source SHA verification.
    nt=z.new_tensor(float(len(z)))
    return 2*LOSS.masked_sigmoid_focal_loss(z,y,nt,v)+5*LOSS.masked_dice_loss(z,y,nt,v)

def geometry(z,y,v,dframes,perm=None):
    logp=F.logsigmoid(z).masked_fill(~v,-torch.inf)
    rho=logp.softmax(-1)
    if perm is not None:rho=torch.gather(rho,1,perm)
    u=y/y.sum(-1,keepdim=True)
    distance=(rho.cumsum(-1)-u.cumsum(-1)).abs()[:,:-1]
    # Only valid adjacent edges; no fictitious padded transport.
    return ((distance*v[:,1:]).sum(-1)*4/torch.maximum(dframes,torch.full_like(dframes,4.))).mean()

def make_perm(y,v,event_ids,seed):
    result=[]
    for target,valid,eid in zip(y.cpu().numpy(),v.cpu().numpy(),event_ids):
        n=len(valid);perm=np.arange(n)
        digest=hashlib.sha256(f'E091/{seed}/{eid}'.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],'big'))
        for flag in [False,True]:
            idx=np.flatnonzero(valid & (target.astype(bool)==flag))
            perm[idx]=rng.permutation(idx)
        require(np.array_equal(np.sort(perm),np.arange(n)),'PERM_BIJECTION')
        require(np.array_equal(target[perm],target) and np.array_equal(valid[perm],valid),'PERM_GT_VALID')
        result.append(perm)
    return torch.as_tensor(np.array(result),device=y.device,dtype=torch.long)

def select_batch(c,entries,device):
    B=c['f'].shape[0];hs=c['hs'].to(device)
    rows=[];qs=[];ys=[];vs=[];ds=[];ids=[]
    for b,m in enumerate(entries):
        for q,t,eid in zip(m['query_indices'],m['target_indices'],m['event_ids']):
            y=c['targets'][b][t];v=c['valid'][b,0]
            require(bool(y[v].sum()>0),'ZERO_TARGET_UNEXPECTED')
            event=c['metadata'][b]['events'][t]
            rows.append(b);qs.append(q);ys.append(y);vs.append(v)
            ds.append(event['end_fid']-event['begin_fid'])
            # Includes video and full pair identity, in addition to stored event ID.
            ids.append(f"{c['metadata'][b]['video_id']}/{c['metadata'][b]['pair_key']}/{eid}")
    return dict(hs=hs,f=c['f'].to(device),rows=torch.tensor(rows,device=device),qs=torch.tensor(qs,device=device),y=torch.stack(ys).to(device),v=torch.stack(vs).to(device),d=torch.tensor(ds,device=device,dtype=torch.float32),ids=ids,original_ids=[eid for m in entries for eid in m['event_ids']])

def calibration(batches,seed,out):
    ev=sorted([(eid,i,j) for i,b in enumerate(batches) for j,eid in enumerate(b['original_ids'])])[:64]
    require(len(ev)==64,'CALIBRATION_LT64')
    base=[];geoB=[];geoS=[];details=[]
    for eid,i,j in ev:
        b=batches[i];z=b['native'][j:j+1].detach().clone().requires_grad_(True)
        y=b['y'][j:j+1];v=b['v'][j:j+1];d=b['d'][j:j+1]
        a=orig_loss(z,y,v);gb=geometry(z,y,v,d);gs=geometry(z,y,v,d,b['perm'][j:j+1])
        norms=[float(torch.autograd.grad(x,z,retain_graph=True)[0][v].norm()) for x in [a,gb,gs]]
        require(all(np.isfinite(norms)),'CALIBRATION_NONFINITE')
        base.append(norms[0]);geoB.append(norms[1]);geoS.append(norms[2])
        details.append(dict(event_id=eid,gradient_L2=norms))
    med=np.median([base,geoB,geoS],axis=1)
    require(np.all(np.isfinite(med)) and np.all(med>0),'CALIBRATION_ZERO_MEDIAN')
    coeff={'A':0.,'B':float(.25*med[0]/med[1]),'S':float(.25*med[0]/med[2])}
    write(out/f'calibration_{seed}.json',dict(seed=seed,coefficients=coeff,rule='.25*median(original valid-logit L2 norm)/median(auxiliary valid-logit L2 norm)',median_norms=med.tolist(),events=details))
    return coeff

def decode_slot(task):
    from decode import project,component
    slot,z,origin,admitted=task
    if not admitted:return slot,dict(P=None,C=None)
    return slot,dict(P=project(z,origin),C=component(z,origin))

def prepare_evaluation(cache,cfg):
    # Fixed slots created from existing F. No GT time supplied to prediction functions.
    locked=list(csv.DictReader((cache/'sources/087/eval_locked.csv').open()))
    members=read(cache/'sources/087/members.json')
    main={(str(v),int(i)) for v,i in members['long_coexist_short_covered']}
    sub={(str(v),int(i)) for v,i in members['long_coexist_subsecond_covered']}
    residual={tuple(x) for x in cfg['residual334']}
    require(len(residual)==334,'RESIDUAL_SUPPORT')
    coords=read(cache/'sources/087/eval_coordinates.json')
    slots={}
    for r in locked:
        key=f"{r['video_name']}/{int(r['pair_idx'])}/{int(r['query'])}"
        slots[key]=dict(video=r['video_name'],pair=int(r['pair_idx']),query=int(r['query']),coord=coords[f"{r['video_name']}/{int(r['pair_idx'])}"])
    from decode import hull_eligible
    pairdata={};native_scores={};admitted={}
    for key,s in slots.items():
        pk=(s['video'],s['pair'])
        if pk not in pairdata:
            c=load(cache/'eval'/f'gt27_{s["video"]}_{s["pair"]:04d}.pt');pairdata[pk]=c
        c=pairdata[pk];n=s['coord']['N_valid']
        native_scores[key]=c['masks'][3,0,s['query'],:n].numpy()
        admitted[key]=hull_eligible(native_scores[key])
    events=[]
    for r in locked:
        key=(r['video_name'],int(r['gt_index']));slot=f"{key[0]}/{int(r['pair_idx'])}/{int(r['query'])}";s=slots[slot]
        events.append(dict(video_name=key[0],gt_index=key[1],begin_fid=int(r['begin_fid']),end_fid=int(r['end_fid']),pair_idx=int(r['pair_idx']),query=int(r['query']),fps=float(r['fps']),relation_sec=float(r['relation_sec']),main400=key in main,sub163=key in sub,residual334=key in residual,long_protection=float(r['relation_sec'])>=3 and float(r['coexist_sec'])>=10,admitted=admitted[slot],origin=s['coord']['origin'],N_valid=s['coord']['N_valid'],slot_id=slot,stride=4))
    require(sum(e['admitted'] for e in events if e['main400'])==381,'MAIN_GATE')
    require(sum(e['long_protection'] for e in events)==1101,'LONG_GROUP')
    require(len(events)==2573 and len(main)==400 and len(sub)==163 and residual<=main,'EVAL_IDENTITIES')
    return slots,pairdata,native_scores,admitted,events

def execute(args):
    global LOSS
    cache=Path(args.cache);out=Path(args.out);code=Path(__file__).parent
    out.mkdir(parents=True,exist_ok=False)
    start=time.time();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    require(torch.cuda.is_available(),'NO_CUDA')
    manifest=read(cache/'manifest.json')
    for f in manifest['files']:require(sha(cache/f['path'])==f['sha256'],'CACHE_SHA '+f['path'])
    cfg=read(code/'config.json')
    freeze=read(code/'software_freeze.json')
    for name,digest in freeze['files'].items():require(sha(code/name)==digest,'SOFTWARE_FREEZE '+name)
    require(cfg['steps']==500 and cfg['seeds']==[90,91,92] and cfg['arms']==['A','B','S'],'CONFIG_CONTRACT')
    for k in ['protocol_sha256','architecture_sha256','review_sha256']:require(cfg[k]==freeze['verified_local_protocols'][k],'PROTOCOL_RECEIPT '+k)
    require(sha(cache/'manifest.json')==cfg['cache_manifest_sha256'],'MANIFEST_SHA')
    require(read(cache/'support_audit.json')['status']=='PASS','CACHE_GATE')
    initfile=load(cache/'initial_mask_embed.pt');state=initfile['state_dict']
    require(initfile['trained'] is False,'INIT_NOT_PRETRAINED')
    source=cache/'sources/repo/models/losses.py'
    require(sha(source)=='204bf7368d978ecbce03bf053fd116c1b7803cc53f0e703612c44c866873c5ae','LOSS_SOURCE')
    spec=importlib.util.spec_from_file_location('original_losses',source)
    LOSS=importlib.util.module_from_spec(spec);spec.loader.exec_module(LOSS)
    files={str(p.name):sha(p) for p in code.glob('*.py')}
    write(out/'execution_freeze.json',dict(code=files,config=cfg,config_sha256=sha(code/'config.json'),cache_manifest_sha256=sha(cache/'manifest.json'),init_sha256=sha(cache/'initial_mask_embed.pt'),matching_sha256=sha(cache/'frozen_matching.json'),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),settings=dict(tf32_matmul=False,tf32_cudnn=True,benchmark=False,torch_threads=2),optim=dict(lr=.0003,weight_decay=0,steps=500,clip=1,seeds=[90,91,92],arms=['A','B','S'])))
    entries=read(cache/'frozen_matching.json')['entries']
    bypath={}
    for e in entries:bypath.setdefault(e['cache_path'],[]).append(e)
    initial=Head(state).cuda().eval()
    batches=[];prechecks=[]
    for role in ['train','adapter_dev','eval']:
        for path in sorted((cache/role).glob('*.pt')):
            c=load(path)
            b=select_batch(c,sorted(bypath[str(path.relative_to(cache))],key=lambda e:e['row']),'cuda') if role!='eval' else dict(hs=c['hs'].cuda(),f=c['f'].cuda())
            with torch.no_grad():
                z=initial(b['hs'],b['f']).masked_fill(~c['valid'].cuda(),-10)
                finite=bool(torch.isfinite(z).all() and torch.isfinite(c['masks'][3]).all())
                error=float((z.cpu()-c['masks'][3]).abs().max()) if finite else None
                bits=torch.equal(z.cpu().sigmoid()>.5,c['masks'][3].sigmoid()>.5)
            row=dict(path=str(path.relative_to(cache)),max_abs=error,finite=finite,bits_equal=bits)
            prechecks.append(row)
            if not finite or error>4e-6 or not bits:
                write(out/'preflight_failure.json',dict(status='TECHNICAL_FAILURE',failed=row,completed_checks=prechecks))
            require(finite and error<=4e-6,'NATIVE_REBUILD '+str(row))
            require(bits,'NATIVE_BITS '+str(row))
            if role=='train':
                b['native']=c['masks'][3].cuda()[b['rows'],b['qs']]
                batches.append(b)
    require(len(batches)==88 and sum(len(b['ids']) for b in batches)==563,'TRAIN_SUPPORT')
    write(out/'preflight.json',dict(status='PASS',checks=prechecks,train_events=563,dev_events=169,encoder_forwards=0,matching_recomputed=False))
    print('PREFLIGHT_PASS',len(prechecks),flush=True)
    # Projection runtime estimate uses synthetic logits only, no evaluation labels.
    from decode import project
    t=time.time()
    for n in [128,256,675]:project(np.linspace(-2,2,n,dtype=np.float32),0)
    bench=time.time()-t
    estimate=bench*cfg['decode_sum_n2']*10/(128**2+256**2+675**2)/cfg['decode_workers']
    write(out/'decode_budget.json',dict(benchmark_seconds=bench,estimated_seconds=estimate,limit_seconds=1800,basis='synthetic fixed-grid shapes, quadratic-log implementation; no GT results'))
    require(estimate<=1800,'PROJECTION_BUDGET_EXCEEDED')
    slots,pairdata,native_scores,admitted,events=prepare_evaluation(cache,cfg)
    write(out/'events.json',events)
    # Three paired seeds; exact same shuffled batch sequence in all arms.
    for seed in [90,91,92]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);order=[]
        while len(order)<500:order.extend(rng.permutation(len(batches)).tolist())
        order=order[:500];write(out/f'order_{seed}.json',order)
        perms={}
        for bi,b in enumerate(batches):
            b['perm']=make_perm(b['y'],b['v'],b['ids'],seed)
            perms[str(bi)]=b['perm'].cpu().numpy()
        np.savez_compressed(out/f'permutations_{seed}.npz',**perms)
        coeff=calibration(batches,seed,out)
        for arm in ['A','B','S']:
            model=Head(state).cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
            log=[]
            for step,bi in enumerate(order,1):
                b=batches[bi];optimizer.zero_grad(set_to_none=True)
                allz=model(b['hs'],b['f']);z=allz[b['rows'],b['qs']]
                orig=orig_loss(z,b['y'],b['v'])
                aux=geometry(z,b['y'],b['v'],b['d'],b['perm'] if arm=='S' else None) if arm!='A' else z.sum()*0
                require(torch.isfinite(orig+aux).item(),'NONFINITE_LOSS')
                go=torch.autograd.grad(orig,z,retain_graph=True)[0].norm()
                gx=torch.autograd.grad(aux,z,retain_graph=True)[0].norm() if arm!='A' else torch.zeros_like(go)
                total=orig+coeff[arm]*aux;total.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                require(torch.isfinite(norm).item(),'NONFINITE_GRADIENT')
                optimizer.step()
                log.append(dict(step=step,batch=bi,base=float(orig.detach()),aux=float(aux.detach()),total=float(total.detach()),base_logit_grad=float(go),scaled_aux_logit_grad=float(gx)*coeff[arm],parameter_grad_preclip=float(norm)))
                if step%100==0:print('TRAIN',seed,arm,step,'loss',round(log[-1]['total'],5),flush=True)
            cp=out/'checkpoints'/f'{arm}{seed}.pt';cp.parent.mkdir(exist_ok=True)
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=500,lambda_geometry=coeff[arm],parent_sha256=sha(cache/'initial_mask_embed.pt')),cp)
            write(out/f'train_{arm}{seed}.json',log)
            del optimizer,model
    variants=['Z']+[f'{a}{seed}' for seed in [90,91,92] for a in ['A','B','S']]
    score_files={};seals={}
    for name in variants:
        if name=='Z':scores=native_scores
        else:
            model=Head(load(out/'checkpoints'/f'{name}.pt')['state_dict']).cuda().eval();scores={}
            with torch.no_grad():
                for pk,c in pairdata.items():
                    z=model(c['hs'].cuda(),c['f'].cuda())[0].cpu().numpy()
                    for key,s in slots.items():
                        if (s['video'],s['pair'])==pk:scores[key]=z[s['query'],:s['coord']['N_valid']]
            del model
        require(all(np.isfinite(z).all() for z in scores.values()),'NONFINITE_INFERENCE')
        path=out/'scores'/f'{name}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**scores)
        score_files[name]=str(path.relative_to(out));seals[score_files[name]]=sha(path)
        print('SCORES_SEALED',name,len(scores),flush=True)
    write(out/'score_seal.json',dict(files=seals,events_sha256=sha(out/'events.json'),GT_time_used_in_model_or_decoder=False))
    # CPU deterministic interval formation, all variants share frozen original slots.
    predictions={}
    with mp.get_context('spawn').Pool(cfg['decode_workers']) as pool:
        for name in variants:
            data=np.load(out/score_files[name])
            tasks=[(key,data[key],int(s['coord']['origin']),admitted[key]) for key,s in slots.items()]
            decoded=dict(pool.imap(decode_slot,tasks,chunksize=8))
            predictions[name]=[decoded[e['slot_id']] for e in events]
            write(out/f'decoded_{name}.json',decoded)
            print('DECODED',name,flush=True)
    write(out/'predictions.json',dict(events=events,predictions=predictions,score_files=score_files))
    write(out/'prediction_seal.json',dict(predictions_sha256=sha(out/'predictions.json'),score_seal_sha256=sha(out/'score_seal.json'),files={p.name:sha(p) for p in out.glob('decoded_*.json')},evaluation_not_started=True))
    write(out/'run_receipt.json',dict(status='TRAINING_AND_PREDICTIONS_COMPLETE',elapsed_seconds=time.time()-start,steps_per_arm_seed=500,training_runs=9,encoder_forwards=0,matching_recomputed=False,trainable_parameter_names=list(state),main400=400,sub163=163,long1101=1101,evaluation_pending=True))
    print('PREDICTIONS_COMPLETE',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    try:execute(args)
    except Exception as e:
        dest=Path(args.out)
        if dest.exists():write(dest/'TECHNICAL_FAILURE.json',dict(status='TECHNICAL_FAILURE',error=repr(e)))
        raise
