"""043 fixed native mask projection experiment; no GT at detection inference."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import sys,json,gzip,time,signal,hashlib,pickle,random,copy,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=Path('/root/rebinding_project/VrdONE')
DATA=Path('/root/autodl-tmp/VrdONE-data')
EX=DATA/'commander_native_error_025/export_v1'
OLD=DATA/'commander_architecture_010/export_v1'
JOB=DATA/'commander_mask_training_043'
ARMS=['original','global','valid']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):
    p=Path(p)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt',compresslevel=1) if p.suffix=='.gz' else p.open('w')) as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def tree_sha(d):
    h=hashlib.sha256()
    for k,v in sorted(d.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def alarm(*args):raise TimeoutError('043 fixed 1800 second budget')

class Capture:
    def __init__(self,model):self.model=model;self.handles=[];self.records=[];self.partial={}
    def __enter__(self):
        self.handles=[self.model.neck.register_forward_hook(self.neck),self.model.predictor.mask_embed.register_forward_pre_hook(self.hs),self.model.predictor.register_forward_hook(self.output)]
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
    def neck(self,m,args,out):self.partial['f']=out[0].detach()
    def hs(self,m,args):self.partial['hs']=args[0].detach()
    def output(self,m,args,out):
        import torch
        r=self.partial;self.partial={};r['valid']=out['output_mask'].detach()
        r['cls']=torch.stack([a['pred_logits'] for a in out['aux_outputs']]+[out['pred_logits']]).detach()
        r['masks']=torch.stack([a['pred_masks'] for a in out['aux_outputs']]+[out['pred_masks']]).detach()
        re=project(self.model.predictor.mask_embed,r)
        assert torch.equal(re,r['masks']),'Current-device cached projection not exact'
        self.records.append(r)

def project(head,r):
    import torch
    B=r['f'].shape[0]
    e=head(r['hs']).view(4,B,256,9).permute(0,1,3,2)
    return torch.einsum('lbqc,bcm->lbqm',e,r['f']).masked_fill(~r['valid'][None,...],-10.)

def reconstruct(model,VidOR,out,sources):
    import torch
    allcache={};audit=[];global_weights=[];split_videos={}
    frozen={'train':'39c0555bda3d99e928aef89a9f346a9b2c418a4e051205729a3838927a2306c5','adapter_dev':'4a9dc55807723c69b2f8d10d73ad827ca3b9c9f2007f2e6607603f8a16db7edc'}
    for role,h in frozen.items():
        p=OLD/role/'manifest.json';assert sha(p)==h;sources[str(p)]=h;m=load(p);dc=m['effective_dataset_config'];ds=VidOR.__new__(VidOR)
        for k in ('with_clip_feature','feat_stride','max_seq_len','cut_max_preds','proposal_max_preds'):setattr(ds,k,dc[k])
        split_videos[role]=set(m['sampled_video_ids'])
        loaderpath=Path(dc['cache_dir'])/(dc['cache_tag']+'_VidOR_training');cached={};records=[]
        for ri,ref in enumerate(m['records']):
            assert sha(ref['path'])==ref['sha256'];sources[ref['path']]=ref['sha256'];old=torch.load(ref['path'],map_location='cpu',weights_only=False)
            xs=[];draw_records=[]
            for row,pm in enumerate(old['metadata']['pair_metadata']):
                full=pm['video_id']
                if full not in cached:
                    path=loaderpath/(full+'.pkl');sources[str(path)]=sha(path)
                    with path.open('rb') as f:cached[full]=pickle.load(f)
                raw=cached[full];idx=[list(k) for k in raw['relation_keys']].index(pm['pair_key'])
                random.seed(pm['sampling_seed']);np.random.seed(pm['sampling_seed']);torch.manual_seed(pm['sampling_seed'])
                calls=[];fn=random.randint
                def traced(a,b):
                    v=fn(a,b);calls.append([int(a),int(b),int(v)]);return v
                random.randint=traced
                try:s=ds._train_getitem(raw,(idx,idx+1))
                finally:random.randint=fn
                assert calls==pm['randint_draws'],(role,ri,row,'random draws changed')
                assert len(s['so_features_list'])==1
                x=s['so_features_list'][0];assert x.shape[0]==2069 and x.shape[1]==int(old['valid'][row].sum())
                assert torch.equal(s['preds_list'][0],old['preds_list'][row]) and torch.equal(s['masks_list'][0],old['masks_list'][row]),'GT reconstruction changed'
                xs.append(x);draw_records.append({'video':full,'pair_key':pm['pair_key'],'draws':calls,'N':x.shape[1],'N_GT':len(old['preds_list'][row]),'censored_positive_cells':pm['censored_positive_cells']})
            B=len(xs);x=torch.zeros(B,2069,512,device='cuda');valid=old['valid'].cuda()[:,None,:]
            for row,v in enumerate(xs):x[row,:,:v.shape[-1]]=v.cuda()
            with torch.no_grad(),Capture(model) as cap:model._mask_vrd(x,valid)
            assert len(cap.records)==1;r=cap.records[0]
            assert torch.equal(r['valid'].cpu().squeeze(1),old['valid'])
            historical={'fpn':float((r['f'].cpu()-old['fpn_features']).abs().max()),'class':float((r['cls'][-1].cpu()-old['base_class_logits']).abs().max()),'mask':float((r['masks'][-1].cpu()-old['base_mask_logits']).abs().max())}
            r['preds']=[a.cuda() for a in old['preds_list']];r['targets']=[a.cuda() for a in old['masks_list']];r['metadata']=old['metadata']['pair_metadata']
            if role=='train':
                for row,g in enumerate(r['preds']):global_weights.extend([512/int(valid[row].sum())]*len(g))
            records.append(r);audit.append({'role':role,'batch':ri,'source_batch_sha256':ref['sha256'],'pairs':draw_records,'reconstruction_exact':True,'zero_update_all_layers_exact':True,'historical_max_abs_drift':historical})
            if (ri+1)%20==0:print('CACHE043',role,ri+1,'/',len(m['records']),flush=True)
        allcache[role]=records
        save=[{k:([a.cpu() for a in v] if k in ('preds','targets') else v.cpu() if torch.is_tensor(v) else v) for k,v in r.items()} for r in records]
        path=out/(role+'_cache.pt');torch.save(save,path);sources['output:'+str(path)]=sha(path)
        del save,cached
    assert not split_videos['train']&split_videos['adapter_dev']
    assert not (split_videos['train']|split_videos['adapter_dev'])&set(load(EX/'manifest.json')['video_names'])
    assert sum(len(r['preds']) for r in allcache['train'])==350
    assert sum(len(g) for r in allcache['train'] for g in r['preds'])==563
    assert sum(len(r['preds']) for r in allcache['adapter_dev'])==89
    assert sum(len(g) for r in allcache['adapter_dev'] for g in r['preds'])==169
    dump(out/'cache_audits.json',audit)
    return allcache,float(np.mean(global_weights))

def native_loss(model,masks,r,arm,global_scale,check_original=False):
    import torch
    import torch.nn.functional as F
    from models.losses import masked_sigmoid_focal_loss,masked_dice_loss
    total=masks.sum()*0;components=[];n=sum(len(g) for g in r['preds']);nt=torch.tensor(float(n),device=masks.device)
    for layer in range(4):
        cls=r['cls'][layer];ms=masks[layer]
        indices,lmask=model.bipartite_match(cls,r['preds'],ms,r['targets'],None,r['valid'])
        idx=model._get_src_permutation_idx(indices);z=ms[idx];y=torch.cat([gt[J] for gt,(_,J) in zip(r['targets'],indices)])
        p=z.sigmoid();pt=p*y+(1-p)*(1-y)
        element=F.binary_cross_entropy_with_logits(z,y,reduction='none')*(1-pt)**2*(.25*y+.75*(1-y))*lmask.to(z.dtype)
        original=element.mean(1).sum()/nt
        if arm=='valid':focal=(element.sum(1)/lmask.sum(1)).sum()/nt
        else:focal=original*(global_scale if arm=='global' else 1.)
        dice=masked_dice_loss(z,y,nt,lmask)
        if check_original:
            source_focal=masked_sigmoid_focal_loss(z,y,nt,lmask)
            assert torch.allclose(original,source_focal,rtol=1e-6,atol=1e-7)
            g0=torch.autograd.grad(original,z,retain_graph=True)[0]
            g1=torch.autograd.grad(source_focal,z,retain_graph=True)[0]
            assert torch.allclose(g0,g1,rtol=1e-5,atol=1e-8),'Native focal gradient differs'
            a=model.loss_masks(cls,ms,r['preds'],r['targets'],None,indices,nt,lmask)
            assert torch.allclose(a['loss_mask'],model.loss_factor['loss_mask']*original,rtol=1e-6,atol=1e-7)
            assert torch.equal(a['loss_dice'],model.loss_factor['loss_dice']*dice)
        ce=model.loss_labels(cls,ms,r['preds'],r['targets'],None,indices,nt,lmask)['loss_class']
        total=total+model.loss_factor['loss_mask']*focal+model.loss_factor['loss_dice']*dice+ce
        components.append(torch.stack([original,focal,dice,ce]))
    return total,torch.stack(components)

def diagnostics(head,records):
    """Same-target predicate set availability; GT used only in this diagnostic."""
    import torch
    vals=[]
    with torch.no_grad():
        for ri,r in enumerate(records):
            ms=project(head,r)[-1];top=r['cls'][-1].softmax(-1)[...,1:].topk(6,-1).indices+1
            for b,(preds,targets) in enumerate(zip(r['preds'],r['targets'])):
                bits=(ms[b].sigmoid()>.5)&r['valid'][b]
                for gi,(pid,y) in enumerate(zip(preds,targets)):
                    yy=y.bool();inter=(bits&yy).sum(-1);union=(bits|yy).sum(-1);iou=inter/union.clamp_min(1);eligible=(top[b]==pid).any(-1)
                    best=float(iou[eligible].max()) if eligible.any() else 0.
                    vals.append({'batch':ri,'row':b,'GT_index':gi,'predicate':int(pid),'N_valid':int(r['valid'][b].sum()),'positive_samples':int(yy.sum()),'best_semantic_set_IoU':best})
    return vals

def train(model,cache,scale,out):
    import torch
    torch.manual_seed(43);np.random.seed(43);random.seed(43)
    heads={a:copy.deepcopy(model.predictor.mask_embed) for a in ARMS}
    for head in heads.values():
        for p in head.parameters():p.requires_grad_(True)
    opt={a:torch.optim.AdamW(heads[a].parameters(),lr=1e-5,weight_decay=.05) for a in ARMS}
    rng=np.random.default_rng(43);order=[]
    while len(order)<500:order+=rng.permutation(len(cache['train'])).tolist()
    order=order[:500];dump(out/'training_schedule.json',{'steps_per_arm':500,'seed':43,'order':order,'global_scale_GT_mean_512_over_V':scale,'trainable_parameters':sum(p.numel() for p in heads['original'].parameters()),'lr':1e-5,'weight_decay':.05,'clip_norm':1.})
    # Check native objective and differentiability before any update.
    r=cache['train'][0];probe=project(heads['original'],r)
    check,co=native_loss(model,probe,r,'original',scale,True);check.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads['original'].parameters())
    for a in ARMS:opt[a].zero_grad(set_to_none=True)
    history=[];t=time.monotonic()
    for step,ri in enumerate(order,1):
        r=cache['train'][ri]
        for a in ARMS:
            opt[a].zero_grad(set_to_none=True);ms=project(heads[a],r);loss,co=native_loss(model,ms,r,a,scale)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(heads[a].parameters(),1.)
            assert torch.isfinite(norm);opt[a].step()
            history.append({'step':step,'batch':ri,'arm':a,'total_loss':float(loss.detach()),'grad_norm_before_clip':float(norm),'per_layer_original_focal_modified_focal_dice_class':co.detach().cpu().tolist()})
        if step%50==0:print('TRAIN043',step,'/500','seconds',round(time.monotonic()-t,1),flush=True)
    dump(out/'training_history.json',history)
    for a,h in heads.items():
        torch.save({'mask_embed':h.state_dict(),'step':500,'arm':a,'global_scale':scale,'seed':43},out/(a+'_step500.pt'))
    dev={a:diagnostics(h,cache['adapter_dev']) for a,h in dict(untrained=model.predictor.mask_embed,**heads).items()};dump(out/'dev_diagnostic.json',dev)
    return heads

def predict(model,heads,VidOR,cfg,manifest,out,sources):
    import torch
    import utils.misc as utils
    from utils.evaluate import EvaluationFormatConvertor
    ds=VidOR.__new__(VidOR);dc=cfg['dataset_config']
    for k in ('with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset'):setattr(ds,k,dc[k])
    converter=EvaluationFormatConvertor('vidor');original_fn=model._mask_vrd;audits=[]
    for i,full in enumerate(manifest['video_names']):
        ref=manifest['source_files']['cache:'+full];p=Path(ref['path']);assert sha(p)==ref['sha256'];sources[str(p)]=ref['sha256']
        with p.open('rb') as f:cached=pickle.load(f)
        inp=ds._val_getitem(cached);inp['video_name']=full;del cached
        gpu=utils.dict_to_device(inp,'cuda:0');del inp
        with torch.no_grad(),Capture(model) as cap:baseline=model(gpu)
        folder=out/'predictions'/full;folder.mkdir(parents=True)
        # Full source decoder replay; only mask projection changes, no GT opened.
        for a,head in dict(untrained=model.predictor.mask_embed,**heads).items():
            cursor=[0]
            def replay(self,x,m):
                r=cap.records[cursor[0]];cursor[0]+=1
                assert torch.equal(m,r['valid']) and tuple(x.shape)==(r['f'].shape[0],2069,r['f'].shape[-1])
                ms=project(head,r)
                return {'pred_logits':r['cls'][-1],'pred_masks':ms[-1],'output_mask':r['valid'],'aux_outputs':[{'pred_logits':cl,'pred_masks':mm} for cl,mm in zip(r['cls'][:-1],ms[:-1])]}
            model._mask_vrd=types.MethodType(replay,model)
            try:
                with torch.no_grad():actual=model(gpu)
            finally:model._mask_vrd=original_fn
            assert cursor[0]==len(cap.records)
            if a=='untrained':assert actual==baseline,'Zero update native decoding changed'
            preds=converter.to_eval_format_pr(full,actual)[full.split('_')[1]] if actual else []
            dump(folder/(a+'.json.gz'),preds)
        audits.append({'video_name':full,'all_pair_source_forward_batches':len(cap.records),'all_layer_cached_projection_exact':True,'zero_update_native_exact':True,'N_pairs':len(gpu['sids'])})
        del gpu,cap,baseline
        if (i+1)%8==0:print('NATIVE043',i+1,'/64',flush=True)
    dump(out/'native_audits.json',audits)
    sealed={str(p.relative_to(out)):sha(p) for p in sorted((out/'predictions').rglob('*.gz'))};assert len(sealed)==256
    dump(out/'predictions_seal_before_GT.json',sealed)

def evaluate(out,manifest,sources):
    ns={};exec(compile((EX/'official.py').read_text(),str(EX/'official.py'),'exec'),ns)
    official=ns['eval_detection_scores_v2'];rows=[];pervideo=[];arms=['untrained']+ARMS
    for full in manifest['video_names']:
        vr=EX/'videos'/full
        for n in ('gt.json.gz','metadata.json','baseline.json.gz'):
            p=vr/n;h=manifest['outputs'][str(p.relative_to(EX))]['sha256'];assert sha(p)==h;sources[str(p)]=h
        gt=load(vr/'gt.json.gz');meta=load(vr/'metadata.json');assert len(gt)==len(meta)
        dump(out/'evaluation_sources'/full/'gt.json.gz',gt);dump(out/'evaluation_sources'/full/'metadata.json',meta)
        ranks={};aps={}
        for a in arms+['historical']:
            preds=load(vr/'baseline.json.gz' if a=='historical' else out/'predictions'/full/(a+'.json.gz'))
            prec,rec,_,rank=official(gt,preds,.5);ranks[a]=rank.tolist();aps[a]=float(ns['voc_ap'](rec,prec))
        pervideo.append({'video_name':full,'N_GT':len(gt),'AP':aps,'hit100':{a:sum(0<=r<100 for r in ranks[a]) for a in ranks}})
        for j,m in enumerate(meta):
            rows.append({'video_name':full,'GT_index':j,'predicate':gt[j]['triplet'][1],'duration_sec':float(m['duration_sec']),'original_group':m['original_group'],'input_feasible':m['oracle008']['input_pair_state']=='feasible_input_pair','rank':{a:ranks[a][j] for a in ranks}})
    assert len(rows)==1985
    groups={'all':lambda r:True,'short':lambda r:r['duration_sec']<3,'subsecond':lambda r:r['duration_sec']<1,'long':lambda r:r['duration_sec']>=3,'C':lambda r:r['original_group']=='C','short_C':lambda r:r['duration_sec']<3 and r['original_group']=='C','F':lambda r:r['input_feasible'],'short_C_F':lambda r:r['input_feasible'] and r['duration_sec']<3 and r['original_group']=='C'}
    result={}
    for name,fn in groups.items():
        rs=[r for r in rows if fn(r)];stats={}
        for a in arms+['historical']:
            stats[a]={'N_GT':len(rs),'hits':{str(k):sum(0<=r['rank'][a]<k for r in rs) for k in (50,100,200)},'comparisons':{}}
            for b in arms:
                stats[a]['comparisons'][b]={'new':sum(0<=r['rank'][a]<100 and not 0<=r['rank'][b]<100 for r in rs),'lost':sum(0<=r['rank'][b]<100 and not 0<=r['rank'][a]<100 for r in rs)}
        result[name]=stats
    maps={a:float(np.mean([r['AP'][a] for r in pervideo])) for a in arms+['historical']}
    passed=all(maps['valid']>=maps[b] and result['all']['valid']['hits']['100']>=result['all'][b]['hits']['100'] and result['short_C']['valid']['hits']['100']>result['short_C'][b]['hits']['100'] for b in ('untrained','original','global'))
    dump(out/'per_gt.json',rows);dump(out/'per_video.json',pervideo)
    return {'mAP':maps,'groups':result,'decision':'SMALL_PILOT_CONTINUATION_SIGNAL' if passed else 'DO_NOT_EXPAND_THIS_VERSION','gate_pass':passed}

def run():
    start=time.monotonic();signal.signal(signal.SIGALRM,alarm);signal.alarm(1800);out=JOB/'results_v1';out.mkdir(parents=True,exist_ok=False);phase='SOURCE_BINDING';sources={}
    try:
        contract=load(HERE/'source_contract.json')
        for rel,h in contract['repo_sources'].items():assert sha(REPO/rel)==h;sources[str(REPO/rel)]=h
        cp=REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==contract['checkpoint_sha256'];sources[str(cp)]=contract['checkpoint_sha256']
        assert sha(EX/'manifest.json')==contract['reference025_manifest_sha256'];manifest=load(EX/'manifest.json');assert len(manifest['video_names'])==64
        cfg=load(EX/'effective_config.json');assert sha(EX/'effective_config.json')==manifest['outputs']['effective_config.json']['sha256'];sources[str(EX/'effective_config.json')]=sha(EX/'effective_config.json')
        assert not cfg['model_config'].get('with_fuzzy',False)
        os.chdir(REPO);sys.path.insert(0,str(REPO));import torch
        from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        import utils.misc as utils
        utils.set_seed(43);torch.set_num_threads(2)
        model=MaskVRD(cfg['model_config'],'cuda:0').eval();model._config_eval(cfg['inference_config'])
        weights=torch.load(cp,map_location='cpu',weights_only=False)[contract['checkpoint_key']]
        model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True);del weights
        model.cuda();model.requires_grad_(False);before=tree_sha(model.state_dict())
        dump(out/'software_binding.json',{'code_sha256':{p.name:sha(p) for p in HERE.iterdir() if p.is_file()},'torch':torch.__version__,'numpy':np.__version__,'checkpoint_sha256':contract['checkpoint_sha256'],'source_sha256':sources})
        phase='CACHE';cache,scale=reconstruct(model,VidOR,out,sources)
        phase='TRAIN';heads=train(model,cache,scale,out);del cache
        phase='NATIVE_PREDICT';predict(model,heads,VidOR,cfg,manifest,out,sources)
        assert tree_sha(model.state_dict())==before,'Frozen original changed'
        phase='EVALUATE';result=evaluate(out,manifest,sources)
        result.update(complete=True,seed=43,steps_per_arm=500,global_scale=scale,elapsed_seconds=time.monotonic()-start,type='SMALL_NATIVE_MASK_PROJECTION_INTERVENTION',original_model_state_unchanged=True,source_sha256=sources,independent_review='PENDING',scope='Exposed64;32trainingvideoquota;single seed;train-source dev may be pretrained;not independent generalization;not historical cause proof')
        dump(out/'summary.json',result);dump(out/'status.json',{'complete':True,'decision':result['decision']});print('COMPLETE043',result['decision'],result['mAP'],flush=True)
    except Exception as e:
        dump(out/'status.json',{'complete':False,'phase':phase,'error':repr(e),'elapsed_seconds':time.monotonic()-start});raise
    finally:signal.alarm(0)
if __name__=='__main__':run()
