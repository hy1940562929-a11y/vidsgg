#!/usr/bin/env python3
"""087: one frozen, fresh-forward readability probe. No remote access or tuning."""
from __future__ import annotations
import argparse,copy,csv,hashlib,importlib,json,os,pickle,random,signal,sys,time
from pathlib import Path
from collections import Counter,defaultdict
from probe_math import position_full,local_points,auroc,aggregate,decision
HERE=Path(__file__).resolve().parent
ARMS=('fine','fused','position');SEED=79;STEPS=500;TOL=4e-6

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def check(condition,message):
    if not condition:raise RuntimeError(message)


def verify_locks(config_file,cfg,preflight_only):
    """Before importing any vendor or torch, verify the actual frozen receipts."""
    base=config_file.parent
    def resolve(value):
        p=Path(value).expanduser();return p if p.is_absolute() else (base/p).resolve()
    def path(name,default):return resolve(cfg.get('paths',{}).get(name,default))
    protocol=path('protocol_freeze','protocol_freeze.json')
    coordinate_receipt=path('coordinates_receipt','coordinates_receipt.json')
    software=path('software_freeze','software_freeze.json')
    coordinates=path('eval_coordinates','eval_coordinates.json')
    measured={}
    def verify(p,expected):
        check(p.is_file(),'required frozen file missing: '+str(p));actual=file_hash(p)
        check(actual==expected,'frozen SHA mismatch: '+str(p));measured[str(p)]=actual
    pr=json.loads(protocol.read_text())
    check(pr['status']=='PROTOCOL_FROZEN_BEFORE_EXECUTION','protocol is not frozen')
    for name,h in pr['frozen_files'].items():verify(protocol.parent/name,h)
    cr=json.loads(coordinate_receipt.read_text());verify(coordinates,cr['output_sha256'])
    sw=json.loads(software.read_text());check(isinstance(sw['files'],dict),'software freeze files map required')
    resolved={}
    for name,h in sw['files'].items():
        p=resolve(name);verify(p,h);resolved[str(p)]=h
    required=[HERE/'run_probe.py',HERE/'probe_math.py',config_file,protocol,coordinate_receipt,coordinates,path('eval_locked','eval_locked.csv'),path('members','members.json')]
    budget_path=path('budget_receipt','budget_receipt.json')
    if not preflight_only:required.append(budget_path)
    check(all(str(p.resolve()) in resolved for p in required),'software freeze omits a required runtime/frozen input')
    for p in (protocol,coordinate_receipt,software):measured[str(p)]=file_hash(p)
    budget=None
    if not preflight_only:
        budget=json.loads(budget_path.read_text());check(budget['approved'] is True and budget['budget_seconds']==1200,'budget receipt is not approved for1200s')
        check(float(budget['estimated_remaining_after_lastdev_seconds'])>=0 and float(budget['reference_lastdev_seconds'])>0 and float(budget['prior_preflight_elapsed_seconds'])>=0,'invalid budget estimate')
    return measured,budget

def runtime_settings(torch,device):
    return dict(device=str(device),GPU=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),cuda_version=str(torch.version.cuda),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),threads=torch.get_num_threads())

def budget_gate(receipt,settings,lastdev_seconds,elapsed):
    check(receipt is not None,'full run requires frozen budget receipt')
    check(settings==receipt['settings'],'runtime settings changed from budget preflight')
    ratio=max(1.,lastdev_seconds/float(receipt['reference_lastdev_seconds']))
    estimated=float(receipt['prior_preflight_elapsed_seconds'])+elapsed+float(receipt['estimated_remaining_after_lastdev_seconds'])*ratio
    return dict(allowed=estimated<=1200,estimated_combined_total_seconds=estimated,limit_seconds=1200,
                runtime_scaling_ratio=ratio,observed_lastdev_seconds=lastdev_seconds,current_elapsed_seconds=elapsed,
                prior_preflight_elapsed_seconds=float(receipt['prior_preflight_elapsed_seconds']),estimation_count=1)

def tensor_hash(x):
    a=x.detach().cpu().contiguous().numpy();return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()

def module_import(folder,name):
    sys.path.insert(0,str(folder));m=importlib.import_module(name)
    check(Path(m.__file__).resolve().parent==Path(folder).resolve(),'wrong imported '+name);return m

def flatten(torch,value,prefix=''):
    if torch.is_tensor(value):return {prefix:value.detach().cpu().clone()}
    if isinstance(value,dict):return {k:v for a,b in value.items() for k,v in flatten(torch,b,prefix+'/'+str(a)).items()}
    if isinstance(value,(list,tuple)):return {k:v for a,b in enumerate(value) for k,v in flatten(torch,b,prefix+'/'+str(a)).items()}
    return {}

def discrete(torch,raw):
    r={}
    def visit(x,prefix=''):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='pred_masks':r[prefix+'/'+k]=(v.sigmoid()>.5).detach().cpu()
                elif k=='pred_logits':
                    r[prefix+'/'+k+'/top6']=v.softmax(-1)[...,1:].topk(6,-1).indices.detach().cpu()
                    r[prefix+'/'+k+'/top1']=v.argmax(-1).detach().cpu()
                else:visit(v,prefix+'/'+k)
        elif isinstance(x,(list,tuple)):
            for i,v in enumerate(x):visit(v,prefix+'/'+str(i))
    visit(raw);return r

def compare(torch,a,b,label):
    check(set(a)==set(b),label+' tensor keys changed');diff={}
    for k in a:
        check(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype,label+' shape/dtype '+k)
        if a[k].is_floating_point():
            check(bool(torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all()),label+' nonfinite '+k)
            v=float((a[k]-b[k]).abs().max()) if a[k].numel() else 0.;diff[k]=v
            check(v<=TOL,f'{label} tolerance {k}: {v} > {TOL}')
        else:check(torch.equal(a[k],b[k]),label+' discrete mismatch '+k)
    return diff

def fresh_forward(torch,model,padded,valid,vendor,out,tag,metadata):
    """Exactly three new forwards at identical batch, tensors, settings and shape."""
    check(not model.training,'encoder must be eval')
    with torch.no_grad():native=model._mask_vrd(padded,valid)
    native_flat=flatten(torch,native);native_disc=discrete(torch,native);del native
    captures=[];output=[];discretes=[]
    for repeat in range(2):
        f={}
        def hook(_module,_args,value):
            check('fine' not in f,'fine hook repeated');f['fine']=value.detach().clone()
        handle=model.neck.lateral_norms[0].register_forward_hook(hook)
        try:
            with torch.no_grad(),vendor['frozen043'].Capture(model) as cap:raw=model._mask_vrd(padded,valid)
        finally:handle.remove()
        check(len(cap.records)==1 and 'fine' in f,'capture unavailable')
        c=cap.records[0];c['fine']=f['fine'];c={k:v.detach().cpu().clone() for k,v in c.items()}
        captures.append(c);output.append(flatten(torch,raw));discretes.append(discrete(torch,raw));del raw,cap
    artifact=dict(inputs=padded.detach().cpu(),valid=valid.detach().cpu(),metadata=metadata,
                  native=native_flat,native_discrete=native_disc,hook=captures[0],hook_output=output[0],
                  repeat=captures[1],repeat_output=output[1],hook_discrete=discretes[0],repeat_discrete=discretes[1])
    path=out/'fresh_raw'/(tag+'.pt');torch.save(artifact,path)
    audit=dict(tag=tag,path=str(path),batch=int(padded.shape[0]),padded_length=int(padded.shape[-1]),input_sha256=tensor_hash(padded),
               native_hook=compare(torch,native_flat,output[0],tag+' native/hook'),
               native_repeat=compare(torch,native_flat,output[1],tag+' native/repeat'),
               capture_repeat=compare(torch,captures[0],captures[1],tag+' capture/repeat'))
    for d in discretes:
        check(set(d)==set(native_disc),tag+' discrete keys')
        check(all(torch.equal(native_disc[k],d[k]) for k in d),tag+' bits/class ordering changed')
    check(torch.equal(captures[0]['valid'],valid.cpu()),tag+' validity changed')
    check(captures[0]['fine'].shape==captures[0]['f'].shape==(padded.shape[0],256,padded.shape[-1]),tag+' feature capacity changed')
    audit.update(bits_equal=True,class_order_equal=True,tolerance=TOL,raw_sha256=file_hash(path))
    return captures[0],audit

class TrainingSource:
    def __init__(self,torch,np,lp,paths,VidOR,sources):
        self.torch=torch;self.np=np;self.lp=lp;self.paths=paths;self.sources=sources;self.raw={};self.annotations={};self.manifests={};self.datasets={};self.videos={}
        for role in ('train','adapter_dev'):
            p=paths.get('train_export_root')/role/'manifest.json';lp.verify_file(p,lp.MANIFEST_SHA[role],sources);m=lp.load_json(p)
            check(m['source_split']=='training' and m['role']==role,'not official training source');self.manifests[role]=m
            ds=VidOR.__new__(VidOR)
            for k in ('with_clip_feature','feat_stride','max_seq_len','cut_max_preds','proposal_max_preds'):setattr(ds,k,m['effective_dataset_config'][k])
            check(ds.feat_stride==4 and ds.max_seq_len==512,'training grid changed');self.datasets[role]=ds
            self.videos[role]=set(m['sampled_video_ids'])
        check(len(self.videos['train'])==32 and len(self.videos['adapter_dev'])==8,'32/8 split required')
        check(not self.videos['train']&self.videos['adapter_dev'],'split overlap')
    def load(self,role,ri):
        torch,np,lp,paths=self.torch,self.np,self.lp,self.paths;m=self.manifests[role];ref=m['records'][ri]
        source=paths.remap(ref['path']);lp.verify_file(source,ref['sha256'],self.sources);old=torch.load(source,map_location='cpu',weights_only=False)
        metadata=old['metadata']['pair_metadata'];valid=old['valid'].bool()
        if valid.ndim==2:valid=valid[:,None,:]
        check(valid.shape==(len(metadata),1,512),'training validity shape changed')
        cfg=m['effective_dataset_config'];loader=paths.remap(cfg['cache_dir'])/(cfg['cache_tag']+'_VidOR_training');xs=[]
        for row,pm in enumerate(metadata):
            v=pm['video_id'];check(v in self.videos[role],'video outside fixed split')
            if v not in self.raw:
                p=loader/(v+'.pkl');self.sources[str(p)]=lp.digest(p)
                with p.open('rb') as f:self.raw[v]=pickle.load(f)
            raw=self.raw[v];idx=[list(k) for k in raw['relation_keys']].index(pm['pair_key'])
            random.seed(pm['sampling_seed']);np.random.seed(pm['sampling_seed']);torch.manual_seed(pm['sampling_seed'])
            original,draws=random.randint,[]
            def traced(a,b):
                v=original(a,b);draws.append([int(a),int(b),int(v)]);return v
            random.randint=traced
            try:item=self.datasets[role]._train_getitem(raw,(idx,idx+1))
            finally:random.randint=original
            check(draws==pm['randint_draws'],'crop draws changed')
            check(len(item['so_features_list'])==1,'crop count changed')
            check(torch.equal(item['preds_list'][0],old['preds_list'][row]) and torch.equal(item['masks_list'][0],old['masks_list'][row]),'crop targets changed')
            x=item['so_features_list'][0];n=int(valid[row].sum());check(x.shape==(2069,n),'crop shape changed')
            check(bool(valid[row,0,:n].all()) and not bool(valid[row,0,n:].any()),'nonprefix valid mask');xs.append(x)
        # Historical forward numbers deliberately never accessed or compared.
        return xs,valid,metadata,old['preds_list'],old['masks_list']
    def ann(self,role,video):
        if video not in self.annotations:
            ref=self.manifests[role]['inputs'][video];p=self.paths.remap(ref['annotation']);self.lp.verify_file(p,ref['annotation_sha256'],self.sources)
            self.annotations[video]=self.lp.load_json(p)
        return self.annotations[video]

def reconstruct(torch,np,model,source,vendor,lmath,pids,out,device,write,audits,preflight_only=False,budget=None,begin=None):
    layers={'train':[],'adapter_dev':[]};exclusions=[];preflight={}
    # Historically problematic final one-pair dev batch must gate the entire job.
    ri=len(source.manifests['adapter_dev']['records'])-1
    xs,valid,meta,preds,targets=source.load('adapter_dev',ri)
    check(len(xs)==1,'historical last-dev batch is no longer a single pair')
    start=time.monotonic()
    def capture(role,ri,xs,valid,meta):
        padded=torch.zeros(len(xs),2069,512,device=device)
        for b,x in enumerate(xs):padded[b,:,:x.shape[-1]]=x.to(device)
        c,a=fresh_forward(torch,model,padded,valid.to(device),vendor,out,f'{role}_{ri:04d}',meta);audits.append(a);write(out/'forward_audit.json',audits);return c
    preflight[('adapter_dev',ri)]=capture('adapter_dev',ri,xs,valid,meta)
    lastdev_seconds=time.monotonic()-start
    write(out/'failfast_lastdev.json',dict(passed=True,batch=ri,pairs=1,seconds=lastdev_seconds,old_output_numbers_compared=False))
    settings=runtime_settings(torch,device)
    write(out/'timing_receipt.json',dict(lastdev_seconds=lastdev_seconds,elapsed_seconds=time.monotonic()-begin,
        settings=settings,fixed_training_batches={r:len(m['records']) for r,m in source.manifests.items()},
        fixed_pair_counts=dict(train=350,adapter_dev=89),training_steps_per_head=500,GT27_scope='all complete native pairs; selected coordinate lock has890 segments',
        workload_estimate_status='AWAITING_FROZEN_BUDGET_RECEIPT' if preflight_only else 'FROZEN_RECEIPT_REQUIRED'))
    if preflight_only:return None,None
    gate=budget_gate(budget,settings,lastdev_seconds,time.monotonic()-begin)
    write(out/'budget_gate.json',gate)
    check(gate['allowed'],'frozen full-workload estimate exceeds1200s budget; stop without export/training')
    pair_counts=Counter();gt_counts=Counter()
    for role in ('train','adapter_dev'):
        for ri in range(len(source.manifests[role]['records'])):
            xs,valid,meta,preds,targets=source.load(role,ri);pair_counts[role]+=len(xs);gt_counts[role]+=sum(len(p) for p in preds)
            c=preflight.pop((role,ri),None)
            if c is None:c=capture(role,ri,xs,valid,meta)
            selected=[]
            for row,pm in enumerate(meta):
                v=pm['video_id'];ann=source.ann(role,v);tids=sorted({int(o['tid']) for frame in ann['trajectories'] for o in frame});sid,oid=[tids[pm['pair_key'][i]] for i in (0,1)]
                origin,stride,n=int(pm['sample_frame_origin']),int(pm['feat_stride']),xs[row].shape[-1]
                episodes=[dict(predicate_id=int(pids[g['predicate']]),begin=int(g['begin_fid']),end=int(g['end_fid'])) for g in ann['relation_instances'] if (g['subject_tid'],g['object_tid'])==(sid,oid)]
                y=lmath.annotation_union(episodes,origin,stride,n);retained=np.zeros_like(y)
                for pid,target in zip(preds[row].tolist(),targets[row]):retained[pid-1]|=target[:n].bool().numpy()
                missing=int((y&~retained).sum());check(not (retained&~y).any(),'retained targets exceed GT union');check(missing==int(pm['censored_positive_cells']),'censor count changed')
                exclusions.append(dict(role=role,batch=ri,row=row,video_name=v,N_GT=len(preds[row]),censored_positive_cells=missing,excluded=missing>0))
                if missing:continue
                selected.append(dict(batch=ri,row=row,video_name=v,origin=origin,stride=stride,N_valid=n,targets=torch.from_numpy(y.astype(np.float32)),features=dict(fine=c['fine'][row,:,:n],fused=c['f'][row,:,:n],position=torch.from_numpy(position_full(n)))))
            layers[role].append(selected)
            if ri%20==0:print('087_CAPTURE',role,ri+1,flush=True)
        torch.save(layers[role],out/(role+'_fresh_layers.pt'))
    check(dict(pair_counts)=={'train':350,'adapter_dev':89},'pair census changed');check(dict(gt_counts)=={'train':563,'adapter_dev':169},'GT census changed')
    write(out/'censoring_audit.json',exclusions);return layers,exclusions

def train(torch,np,layers,layerprobe,vendor,out,device,write):
    stats=layerprobe.fit_train_statistics(torch,layers['train']);torch.save(stats,out/'train_only_statistics.pt')
    write(out/'predicate_support.json',dict(supported_predicate_ids=(torch.where(stats['supported'])[0]+1).tolist(),positive_counts=stats['positive_counts'].tolist(),negative_counts=stats['negative_counts'].tolist(),N_frames=stats['N_frames']))
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);initial=torch.nn.Linear(256,50).to(device);heads={a:copy.deepcopy(initial) for a in ARMS}
    torch.save(initial.state_dict(),out/'head_initial.pt');opt={a:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01) for a,h in heads.items()}
    available=[i for i,b in enumerate(layers['train']) if b];check(available,'no training batches');order=[];rng=np.random.default_rng(SEED)
    while len(order)<STEPS:order.extend(rng.permutation(available).tolist())
    order=order[:STEPS];hashes={a:vendor['frozen043'].tree_sha(h.state_dict()) for a,h in heads.items()};check(len(set(hashes.values()))==1,'initialization differs')
    write(out/'training_recipe.json',dict(seed=SEED,steps=STEPS,schedule=order,lr=.001,weight_decay=.01,clip_norm=1,initialization_sha256=hashes,dev_checkpoint_selection=False))
    support=stats['supported'].to(device);pw=stats['positive_weight'].to(device);normal=stats['normalizer'].to(device);history=[]
    # Cache standardized small probe inputs to avoid thousands of CPU transfers.
    prepared={}
    for bi in available:
        batch=layers['train'][bi];prepared[bi]=dict(target=torch.cat([p['targets'].T for p in batch]).to(device),features={})
        for a in ARMS:
            x=torch.cat([p['features'][a].T for p in batch]).to(device);prepared[bi]['features'][a]=(x-stats['means'][a].to(device))/stats['stds'][a].to(device)
    for step,bi in enumerate(order,1):
        target=prepared[bi]['target']
        for arm,head in heads.items():
            opt[arm].zero_grad(set_to_none=True);logits=head(prepared[bi]['features'][arm])
            element=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=pw,reduction='none')/normal
            loss=element[:,support].mean();check(bool(torch.isfinite(loss)),'nonfinite loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);check(bool(torch.isfinite(norm)),'nonfinite gradient');opt[arm].step()
            history.append(dict(step=step,batch=bi,arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),N_frames=len(target)))
        if step%50==0:write(out/'training_history.json',history);print('087_TRAIN',step,flush=True)
    for a,h in heads.items():h.eval();torch.save(h.state_dict(),out/(a+'_step500.pt'))
    # Exposed dev is an execution diagnostic, never a selection criterion.
    devloss={a:[] for a in ARMS}
    with torch.no_grad():
        for batch in layers['adapter_dev']:
            if not batch:continue
            target=torch.cat([p['targets'].T for p in batch]).to(device)
            for a,h in heads.items():
                x=torch.cat([p['features'][a].T for p in batch]).to(device);x=(x-stats['means'][a].to(device))/stats['stds'][a].to(device)
                e=torch.nn.functional.binary_cross_entropy_with_logits(h(x),target,pos_weight=pw,reduction='none')/normal;devloss[a].append(dict(loss=float(e[:,support].mean()),N_frames=len(target)))
    write(out/'exposed_dev_loss.json',devloss);return heads,stats

def evaluate_gt27(torch,np,model,heads,stats,vendor,paths,source,pids,out,write,sources,audits,lp):
    locked_path=paths.get('eval_locked');member_path=paths.get('members');coord_path=paths.get('eval_coordinates')
    for p in (locked_path,member_path,coord_path):sources[str(p)]=lp.digest(p)
    with locked_path.open(encoding='utf-8-sig',newline='') as f:locked=list(csv.DictReader(f))
    members=lp.load_json(member_path);coords=lp.load_json(coord_path)
    check(len(locked)==2573,'locked GT27 total changed')
    main='long_coexist_short_covered';sub='long_coexist_subsecond_covered'
    groups={k:{(str(v),int(i)) for v,i in values} for k,values in members.items() if isinstance(values,list)}
    check(len(groups.get(main,[]))==400 and len(groups.get(sub,[]))==163,'frozen400/163 members changed')
    keys={(r['video_name'],int(r['gt_index'])) for r in locked};check(len(keys)==2573,'duplicate evaluation keys')
    check(all(s<=keys for s in groups.values()),'membership outside locked GT');check(groups[sub]<=groups[main],'subsecond not subset')
    byvideo=defaultdict(list)
    for r in locked:byvideo[r['video_name']].append(r)
    check(len(byvideo)==27,'GT27 video count changed');check(not set(byvideo)&(source.videos['train']|source.videos['adapter_dev']),'eval/train split overlap')
    h=module_import(paths.get('helper073'),'history_probe');h.REPO=paths.repo;h.DATA=paths.data
    # setup() reads the same effective model dataset config used in the old GT helper.
    h.CFG=paths.get('gt_input_config',str(h.CFG));sources[str(h.CFG)]=lp.digest(h.CFG)
    _,ds,input_cfg,helper,_=h.setup()
    for key in ('ROOT','DATA','GT_PROP','GT_FEAT','CACHE'):
        if hasattr(helper,key):setattr(helper,key,paths.remap(getattr(helper,key)))
    for module in (h,helper):sources[str(Path(module.__file__).resolve())]=lp.digest(module.__file__)
    check(input_cfg['dataset_config']['feat_stride']==4 and not input_cfg['dataset_config']['random_stride'],'GT grid changed')
    supported=stats['supported'].numpy();rows=[];pairaudit=[];score_receipts=[];device=next(heads['fine'].parameters()).device
    for v,rr in sorted(byvideo.items()):
        ann,inp,tids,ap=h.video_input(v,ds,helper);sources[str(ap)]=lp.digest(ap);pp=helper.GT_PROP/(v+'.pkl');sources[str(pp)]=lp.digest(pp)
        related=defaultdict(list)
        for g in ann['relation_instances']:related[(int(g['subject_tid']),int(g['object_tid']),g['predicate'])].append((int(g['begin_fid']),int(g['end_fid'])))
        pairrows=defaultdict(list)
        for r in rr:
            gi=int(r['gt_index']);g=ann['relation_instances'][gi]
            check((g['subject_tid'],g['object_tid'],g['predicate'],g['begin_fid'],g['end_fid'])==(int(r['subject_tid']),int(r['object_tid']),r['predicate'],int(r['begin_fid']),int(r['end_fid'])),'GT identity/interval mismatch')
            check(abs(float(r['fps'])-float(ann['fps']))<1e-9,'FPS mismatch')
            if r['pair_idx']=='':
                rows.append(dict(video_name=v,gt_index=gi,predicate=r['predicate'],predicate_id=int(pids[r['predicate']]),relation_sec=float(r['relation_sec']),eligible=False,reason='ORIGINAL_LOCK_HAS_NO_PAIR',auroc=None,groups=[k for k,s in groups.items() if (v,gi) in s]));continue
            pi=int(r['pair_idx']);check(0<=pi<len(inp['so_features_list']),'locked pair missing');pairrows[pi].append(r)
        for pi,x in enumerate(inp['so_features_list']):
            si,oi=int(inp['sids'][pi]),int(inp['oids'][pi]);sid,oid=int(tids[si]),int(tids[oi]);a=max(int(inp['traj_durations'][si,0]),int(inp['traj_durations'][oi,0]));b=min(int(inp['traj_durations'][si,1]),int(inp['traj_durations'][oi,1]));origin=a+int(inp['so_offset'][pi]);n=x.shape[-1]
            coord=dict(origin=origin,N_valid=n,subject_tid=sid,object_tid=oid,pair_start=a,pair_end=b)
            key=f'{v}/{pi}'
            if pi in pairrows:check(key in coords,'selected coordinate lock missing '+key)
            if key in coords:check(all(int(coords[key][k])==z for k,z in coord.items()),'locked coordinate changed '+key)
            xx,mm,ids=model.preprocessing([x]);branches=[k for k in range(2) if xx[k] is not None]
            check(len(branches)==1 and ids[branches[0]]==[0],'native preprocessing mapping changed');k=branches[0]
            check(int(mm[k].sum())==n,'full sequence truncated')
            c,audit=fresh_forward(torch,model,xx[k],mm[k],vendor,out,f'gt27_{v}_{pi:04d}',dict(video_name=v,pair_idx=pi,**coord));audits.append(audit)
            write(out/'forward_audit.json',audits);pairaudit.append(dict(video_name=v,pair_idx=pi,**coord,input_sha256=tensor_hash(x),padding=int(xx[k].shape[-1])-n))
            features=dict(fine=c['fine'][0,:,:n],fused=c['f'][0,:,:n],position=torch.from_numpy(position_full(n)))
            scores={}
            with torch.no_grad():
                for arm,head in heads.items():
                    z=features[arm].to(device);z=(z-stats['means'][arm].to(device)[:,None])/stats['stds'][arm].to(device)[:,None]
                    scores[arm]=head(z.T).T.cpu().numpy();check(np.isfinite(scores[arm]).all(),'nonfinite probe score')
            frames=origin+4*np.arange(n)
            scorepath=out/'scores'/f'{v}_{pi:04d}.npz'
            np.savez(scorepath,frames=frames,**scores,native_masks=c['masks'][-1,0,:,:n].numpy(),native_classes=c['cls'][-1,0].numpy())
            score_receipts.append(dict(video_name=v,pair_idx=pi,path=str(scorepath),sha256=lp.digest(scorepath),N_valid=n,all_predicate_scores=True))
            for r in pairrows[pi]:
                gi=int(r['gt_index']);pid=int(pids[r['predicate']]);q=int(r['query']);check((int(r['subject_tid']),int(r['object_tid']))==(sid,oid),'locked pair identity changed');check(0<=q<c['masks'].shape[-2],'locked query missing')
                s,e=int(r['begin_fid']),int(r['end_fid']);pos,neg=local_points(frames,s,e,related[(sid,oid,r['predicate'])])
                reasons=[]
                if not supported[pid-1]:reasons.append('NO_TRAIN_POSITIVE_OR_NEGATIVE_SUPPORT')
                if not pos.any():reasons.append('NO_INTERNAL_SAMPLE')
                if not neg.any():reasons.append('NO_UNION_EXCLUDED_NEIGHBOR_SAMPLE')
                values={arm:auroc(z[pid-1],pos,neg) for arm,z in scores.items()} if not reasons else None
                row=dict(video_name=v,gt_index=gi,predicate=r['predicate'],predicate_id=pid,subject_tid=sid,object_tid=oid,pair_idx=pi,query=q,begin_fid=s,end_fid=e,fps=float(r['fps']),relation_sec=float(r['relation_sec']),origin=origin,N_valid=n,N_positive=int(pos.sum()),N_negative=int(neg.sum()),positive_indices=np.flatnonzero(pos).tolist(),negative_indices=np.flatnonzero(neg).tolist(),eligible=not reasons,reason=';'.join(reasons) or None,auroc=values,native_locked_query_auroc=auroc(c['masks'][-1,0,q,:n].numpy(),pos,neg),groups=[k for k,gkeys in groups.items() if (v,gi) in gkeys])
                rows.append(row)
            del c,xx,mm,features,scores
        write(out/'per_gt.json',rows);write(out/'evaluation_pair_audit.json',pairaudit);write(out/'score_receipts.json',score_receipts)
        print('087_EVAL',v,'GT',len(rows),flush=True)
    check(len(rows)==2573 and len({(r['video_name'],r['gt_index']) for r in rows})==2573,'evaluation coverage changed')
    summaries={k:aggregate([r for r in rows if (r['video_name'],r['gt_index']) in gkeys]) for k,gkeys in groups.items()}
    summaries['all']=aggregate(rows);summaries['long']=aggregate([r for r in rows if r['relation_sec']>=3]);summaries['short']=aggregate([r for r in rows if r['relation_sec']<3])
    support={k:dict(N_total=len(rr),N_excluded=sum(not r['eligible'] for r in rr),excluded_videos=sorted({r['video_name'] for r in rr if not r['eligible']}),reasons=dict(Counter(r['reason'] for r in rr if not r['eligible'])),unsupported_predicates=dict(Counter(r['predicate'] for r in rr if r['reason'] and 'NO_TRAIN' in r['reason']))) for k,rr in {'all':rows,'primary400':[r for r in rows if main in r['groups']],'subsecond163':[r for r in rows if sub in r['groups']]}.items()}
    write(out/'coverage_ledger.json',support);write(out/'auroc_groups.json',summaries);write(out/'per_video.json',{k:v['per_video'] for k,v in summaries.items()})
    write(out/'split_audit.json',dict(train=sorted(source.videos['train']),exposed_dev=sorted(source.videos['adapter_dev']),GT27=sorted(byvideo),disjoint=True))
    return summaries,support,decision(summaries[main])

def run(config_file,output,preflight_only=False):
    begin=time.monotonic();config_file=Path(config_file).resolve();out=Path(output).resolve();check(not out.exists(),'fresh output required');out.mkdir(parents=True);(out/'fresh_raw').mkdir();(out/'scores').mkdir()
    phase='CONFIG';write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        cfg=json.loads(config_file.read_text());base=config_file.parent
        frozen_sources,budget=verify_locks(config_file,cfg,preflight_only)
        write(out/'freeze_audit.json',dict(passed=True,checked_before_torch_import=True,sha256=frozen_sources))
        def resolve(x):
            p=Path(x).expanduser();return p if p.is_absolute() else (base/p).resolve()
        vendor079=resolve(cfg['paths']['vendor079']);lp=module_import(vendor079,'loss_probe');layerprobe=module_import(vendor079,'layer_probe');lmath=module_import(vendor079,'layer_math');write=lp.write_json
        paths=lp.Paths(cfg,config_file);sources=dict(frozen_sources);sources[str(config_file)]=lp.digest(config_file)
        for p in [HERE/'run_probe.py',HERE/'probe_math.py']+[vendor079/n for n in ('loss_probe.py','loss_math.py','layer_probe.py','layer_math.py')]:sources[str(p)]=lp.digest(p)
        options=cfg['layer_probe']
        check((int(options['steps']),int(options['seed']),float(options['learning_rate']),int(options['wall_seconds']),int(options['bootstrap_reps']),int(options['bootstrap_seed']))==(500,79,.001,1200,10000,87),'087 frozen recipe changed')
        def alarm(*_):raise TimeoutError('087 1200-second wall budget')
        signal.signal(signal.SIGALRM,alarm)
        prior=float(budget['prior_preflight_elapsed_seconds']) if budget else 0.
        remaining=1200-prior-(time.monotonic()-begin);check(remaining>0,'combined preflight/full budget already exhausted');signal.alarm(max(1,int(remaining)))
        vendorpath=paths.get('vendor078');contract=lp.load_json(vendorpath/'source_contract043.json');sources[str(vendorpath/'source_contract043.json')]=lp.digest(vendorpath/'source_contract043.json')
        runtime=vendorpath/'runtime_manifest.json';sources[str(runtime)]=lp.digest(runtime)
        for n,d in lp.load_json(runtime)['files'].items():lp.verify_file(vendorpath/n,d,sources)
        for n,d in contract['repo_sources'].items():lp.verify_file(paths.repo/n,d,sources)
        checkpoint=paths.get('checkpoint');lp.verify_file(checkpoint,contract['checkpoint_sha256'],sources)
        cohorts=lp.load_json(vendorpath/'cohorts.json');sources[str(vendorpath/'cohorts.json')]=lp.digest(vendorpath/'cohorts.json')
        export=paths.remap(cohorts['047']['export']);lp.verify_file(export/'manifest.json',cohorts['047']['manifest_sha256'],sources)
        manifest=lp.load_json(export/'manifest.json');lp.verify_file(export/'effective_config.json',manifest['outputs']['effective_config.json']['sha256'],sources);modelcfg=lp.load_json(export/'effective_config.json')
        os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
        import numpy as np
        import torch
        device=cfg.get('device','cuda:0');check(str(device).startswith('cuda') and torch.cuda.is_available(),'authorized CUDA required')
        torch.cuda.set_device(torch.device(device));torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        vendor=lp.import_vendor(vendorpath,paths.repo)
        from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        from dataloaders.category import vidor_pred_name_to_id as pids
        model=MaskVRD(modelcfg['model_config'],device).eval();model._config_eval(modelcfg['inference_config'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)[contract['checkpoint_key']];model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
        model.to(device).requires_grad_(False);before=vendor['frozen043'].tree_sha(model.state_dict());audits=[]
        phase='FRESH_RECONSTRUCTION_AND_LASTDEV_FAILFAST';source=TrainingSource(torch,np,lp,paths,VidOR,sources)
        layers,exclusions=reconstruct(torch,np,model,source,vendor,lmath,pids,out,device,write,audits,preflight_only=preflight_only,budget=budget,begin=begin)
        if preflight_only:
            check(vendor['frozen043'].tree_sha(model.state_dict())==before,'encoder changed in preflight')
            timing=lp.load_json(out/'timing_receipt.json');timing['elapsed_seconds']=time.monotonic()-begin;timing['source_sha256']=sources;write(out/'timing_receipt.json',timing)
            write(out/'source_sha256.json',sources);write(out/'status.json',dict(complete=False,status='TECHNICAL_PREFLIGHT_PASSED',scientific_result_available=False,elapsed_seconds=time.monotonic()-begin));return
        check(vendor['frozen043'].tree_sha(model.state_dict())==before,'encoder changed during capture')
        phase='FIXED_500_STEP_READOUTS';heads,stats=train(torch,np,layers,layerprobe,vendor,out,device,write);del layers
        phase='COMPLETE_GT27_FRESH_FORWARD_EVALUATION';groups,coverage,result=evaluate_gt27(torch,np,model,heads,stats,vendor,paths,source,pids,out,write,sources,audits,lp)
        phase='FINAL_HASH_AUDIT';check(vendor['frozen043'].tree_sha(model.state_dict())==before,'frozen encoder changed')
        for p,h in sources.items():check(lp.digest(p)==h,'source changed '+p)
        write(out/'source_sha256.json',sources)
        summary=dict(complete=True,experiment='087',decision=result,groups=groups,coverage=coverage,frozen_model_unchanged=True,model_state_sha256=before,source_sha256=sources,elapsed_seconds=time.monotonic()-begin,old_cached_outputs_numerically_compared=False,training_updates_per_head=500,encoder_training_updates=0,forward_tolerance=TOL,forward_checks=len(audits),independent_review='PENDING',scope='fixed-recipe local annotation boundary readability; previously exposed GT27; not semantic competition or irreversible information loss')
        write(out/'artifact_sha256.json',{str(p.relative_to(out)):lp.digest(p) for p in out.rglob('*') if p.is_file()});write(out/'summary.json',summary);write(out/'status.json',dict(complete=True,status='COMPLETE',decision=result));print('COMPLETE087',result,flush=True)
    except BaseException as e:
        write(out/'status.json',dict(complete=False,status='TECHNICAL_FAILURE',phase=phase,error=repr(e),elapsed_seconds=time.monotonic()-begin));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();run(a.config,a.out,a.preflight_only)
