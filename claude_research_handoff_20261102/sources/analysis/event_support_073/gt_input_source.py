#!/usr/bin/env python3
"""Frozen VrdONE GT-input diagnostic. All writes are under --out.

GT relation times/predicates are never used in feature construction. Only the
ordered entity identity set is used to request oracle pairs. All continuous
co-presence segments of each requested pair are forwarded, not GT time crops.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, hashlib, inspect, json, pickle, shutil, sys, textwrap, time, types
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path('/root/rebinding_project/VrdONE')
DATA = Path('/root/autodl-tmp/VrdONE-data')
GT_PROP = ROOT/'datasets/vidor/features/gt_per_video_val_strict27'
GT_FEAT = ROOT/'datasets/vidor/features/GT_boxfeatures_validation_strict27'
CACHE = DATA/'cache_reproduced_fullval/MEGA_mask_VidOR_validation'

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def load(p):
    with p.open('rb') as f: return pickle.load(f)

def segments(fs):
    fs=sorted(fs)
    if not fs:return []
    out=[];a=prev=fs[0]
    for x in fs[1:]:
        if x!=prev+1:out.append([a,prev+1]);a=x
        prev=x
    return out+[[a,prev+1]]

def gt_cache(full, ann, audit=True):
    from dataloaders.category import vidor_category_name_to_id
    p=load(GT_PROP/(full+'.pkl'))['traj_proposal']
    d=p['traj_durations'].clone();d[:,0]-=1
    tids=list(map(int,p['gt_tids']))
    assert d.tolist()==p['gt_intervals_0based']
    assert len(tids)==len(p['bboxes_list'])==len(p['features_list'])
    assert len(set((tid,*dd) for tid,dd in zip(tids,d.tolist())))==len(tids)
    counts={}; frames=defaultdict(dict)
    cats={int(x['tid']):x['category'] for x in ann['subject/objects']}
    for fid, objs in enumerate(ann['trajectories']):
        for o in objs:
            b=o['bbox'];frames[int(o['tid'])][fid]=[b['xmin'],b['ymin'],b['xmax'],b['ymax']]
    expected={(tid,a,b) for tid, ff in frames.items() for a,b in segments(ff)}
    actual={(tid,*dd) for tid,dd in zip(tids,d.tolist())}
    assert expected==actual, ('missing/extra GT entity segments',full)
    ff=load(GT_FEAT/(full+'.pkl')) if audit else None
    nframes=0
    for i,(tid,(a,b)) in enumerate(zip(tids,d.tolist())):
        expected_b=torch.tensor([frames[tid][t] for t in range(a,b)],dtype=torch.float32)
        assert torch.equal(expected_b,p['bboxes_list'][i]),(full,i,'GT boxes')
        assert int(p['cat_ids'][i])==vidor_category_name_to_id[cats[tid]]
        f=p['features_list'][i]
        assert f.shape==(b-a,1024) and torch.isfinite(f).all()
        assert not (f==0).all(dim=1).any(),(full,i,'empty GT feature')
        if audit:
            for j,t in enumerate(range(a,b)):
                rec=ff[t+1];ix=np.flatnonzero(np.asarray(rec['tids'])==tid)
                assert len(ix)==1,(full,tid,t,'feature tid')
                assert np.array_equal(f[j].numpy(),np.asarray(rec['visual_features'][ix[0]],dtype=np.float32))
                nframes+=1
    # Only identities are used here. No predicate/start/end is passed to loader.
    requested={(int(r['subject_tid']),int(r['object_tid'])) for r in ann['relation_instances']}
    pairs=[]
    for si,st in enumerate(tids):
        for oi,ot in enumerate(tids):
            if (st,ot) not in requested or st==ot:continue
            if max(d[si,0],d[oi,0])<min(d[si,1],d[oi,1]):pairs.append((si,oi))
    cache=dict(sids=torch.tensor([x[0] for x in pairs],dtype=torch.int64),
        oids=torch.tensor([x[1] for x in pairs],dtype=torch.int64),cat_ids=p['cat_ids'].clone(),
        cat_scores=torch.ones(len(tids)),traj_durations=d,bboxes_list=p['bboxes_list'],
        visual_features_list=p['features_list'],video_wh=p['video_wh'])
    stats=dict(GT_entities=len(frames),GT_segments=len(tids),requested_identity_pairs=len(requested),
        all_copresence_segments=len(pairs),aligned_entity_frames=nframes,
        GT_relations=len(ann['relation_instances']),GT_frames=len(ann['trajectories']))
    return cache,tids,stats

def save_input(path,inp,tids=None):
    result={k:v for k,v in inp.items() if k not in ('so_features_list','video_name')}
    if tids is not None:result['gt_tids']=tids
    with path.open('wb') as f:pickle.dump(result,f)

def raw_save(folder,full,raw,inp):
    arrays={'pred_logits':torch.stack(raw['pred_logits']).cpu().numpy()};meta=[]
    for i,(si,oi) in enumerate(zip(inp['sids'],inp['oids'])):
        si=int(si);oi=int(oi);n=int(raw['output_mask'][i].sum())
        arrays[f'mask_{i}']=raw['pred_masks'][i][:,:n].cpu().numpy()
        sd,od=inp['traj_durations'][si],inp['traj_durations'][oi]
        meta.append(dict(pair_idx=i,sid=si,oid=oi,pair_start=int(max(sd[0],od[0])),
                         pair_end=int(min(sd[1],od[1])),offset=int(inp['so_offset'][i])))
    np.savez_compressed(folder/(full+'.npz'),**arrays)
    dump(folder/(full+'.json'),meta)

def source_raw(full,inp):
    roots=[DATA/'commander_common_failure_009'/n/'raw' for n in
        ['remaining731_v5','existing96_v4_raw','speed8_v5_retry','speed8_v5','expanded64_v2','old32_v1']]
    roots+=[DATA/'commander_failure_audit_20260928_v1/pilot_retry/raw']
    for rt in roots:
        p=rt/(full+'.npz')
        if not p.is_file() or not p.with_suffix('.json').is_file():continue
        meta=json.loads(p.with_suffix('.json').read_text())
        assert len(meta)==len(inp['sids']), (full,'native raw number',p)
        with np.load(p) as z:
            for i,r in enumerate(meta):
                assert (r['sid'],r['oid'])==(int(inp['sids'][i]),int(inp['oids'][i]))
                sd,od=inp['traj_durations'][r['sid']],inp['traj_durations'][r['oid']]
                assert r['pair_start']==int(max(sd[0],od[0])) and r['pair_end']==int(min(sd[1],od[1]))
                assert r['offset']==int(inp['so_offset'][i])
                assert z[f'mask_{i}'].shape==(9,inp['so_features_list'][i].shape[-1])
        return p
    return None

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);pa.add_argument('--limit',type=int,default=27)
    pa.add_argument('--preflight-only',action='store_true');pa.add_argument('--wall-seconds',type=float,default=1800)
    pa.add_argument('--native-fresh',action='store_true',help='forward native and GT in the same process; do not reuse native raw')
    args=pa.parse_args();out=Path(args.out);assert not out.exists(),'never overwrite';out.mkdir(parents=True)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    from dataloaders.vidor import VidOR
    from models.maskvrd import MaskVRD
    import utils.misc as utils
    utils.set_seed(42);torch.set_num_threads(2)
    cfg=yaml.safe_load((ROOT/'configs/vidor.yaml').read_text())
    dc=dict(cfg['dataset_config']);dc.update(cfg['test_dataset_config'])
    mc=dict(cfg['model_config']);mc['with_clip_feature']=False
    assert (dc['feat_stride'],dc['stride_offset'],dc['proposal_min_frames'],dc['with_clip_feature'])==(4,0,5,False)
    assert not dc['random_stride']
    ds=VidOR.__new__(VidOR)
    for k in ['with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset']:setattr(ds,k,dc[k])
    selected=sorted(p.stem for p in GT_PROP.glob('*.pkl'))
    assert len(selected)==27
    selected=selected[:args.limit];dump(out/'selection.json',selected)
    dump(out/'effective_config.json',dict(model_config=mc,dataset_config=dc,inference_config=cfg['inference_config']))
    hashes={str(ROOT/'models/maskvrd.py'):sha(ROOT/'models/maskvrd.py'),str(ROOT/'dataloaders/vidor.py'):sha(ROOT/'dataloaders/vidor.py'),str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    assert hashes[str(ROOT/'models/maskvrd.py')]=='52d5ae191d8331679f69e4d478fa0acffee5ec34423e40dda25cc56a37144d27'
    assert hashes[str(ROOT/'dataloaders/vidor.py')]=='329097169bb2772a951b3409952065b26e735fa5cd1f1c5783caf3b458417787'
    model=None;start=time.time();stats=[];baseline_verified=False
    def make_model():
        nonlocal model
        if model is not None:return model
        cp=ROOT/'experiments/vrdone_vidor/ckpt_vidor.pth';hashes[str(cp)]=sha(cp)
        assert hashes[str(cp)]=='4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627'
        model=MaskVRD(mc,device='cuda:0');model.eval();model._config_eval(cfg['inference_config']);model.cuda()
        ck=torch.load(cp,map_location='cpu',weights_only=False);sd=ck['model_state_dict_ema']
        model.load_state_dict({k.removeprefix('module.'):v for k,v in sd.items()},strict=True)
        src=textwrap.dedent(inspect.getsource(MaskVRD.forward_test));needle='    ## remove pairs without interaction\n'
        assert src.count(needle)==1
        src=src.replace(needle,'    return predictions\n'+needle)
        ns=dict(vars(sys.modules[MaskVRD.__module__]));exec(compile(src,'<capture-before-postprocess>','exec'),ns)
        model.forward_test=types.MethodType(ns['forward_test'],model)
        return model
    for cond in ['native','gt']:
        for sub in ['inputs','raw']:(out/cond/sub).mkdir(parents=True)
    for full in selected:
        assert time.time()-start<args.wall_seconds,'wall budget exceeded'
        group,vid=full.split('_');ap=ROOT/'datasets/vidor/annotations/validation'/group/(vid+'.json')
        ann=json.loads(ap.read_text());gt,tids,st=gt_cache(full,ann)
        # Threshold > 1 disables native duplicate-track suppression for true GT entities.
        gi=ds._val_getitem(gt,viou_threshold=1.01)
        original_gt=ds._val_getitem(gt)
        st.update(video=full,gt_valid_segments=len(gi.get('sids',[])),
                  old_loader_gt_valid_segments=len(original_gt.get('sids',[])))
        del gt,original_gt
        native=load(CACHE/(full+'.pkl'));ni=ds._val_getitem(native);del native
        st['native_valid_pairs']=len(ni.get('sids',[]))
        rp=source_raw(full,ni) if ni else None
        st['native_raw_source']=str(rp) if rp else None
        for p in [ap,GT_PROP/(full+'.pkl'),GT_FEAT/(full+'.pkl'),CACHE/(full+'.pkl')]:hashes[str(p)]=sha(p)
        if rp:hashes[str(rp)]=sha(rp);hashes[str(rp.with_suffix('.json'))]=sha(rp.with_suffix('.json'))
        if args.preflight_only:
            stats.append(st);dump(out/'preflight.json',stats);print(json.dumps(st),flush=True);continue
        for cond,inp in [('native',ni),('gt',gi)]:
            inp['video_name']=full
            save_input(out/cond/'inputs'/(full+'.pkl'),inp,tids if cond=='gt' else None)
            if not len(inp.get('sids',[])):
                np.savez_compressed(out/cond/'raw'/(full+'.npz'),pred_logits=np.empty((0,9,51)))
                dump(out/cond/'raw'/(full+'.json'),[]);continue
            # Reuse native outputs only after one exact-input frozen-forward check.
            if cond=='native' and rp is not None and not args.native_fresh:
                if not baseline_verified:
                    net=make_model();gpu=utils.dict_to_device(inp,'cuda:0')
                    with torch.no_grad():rr=net(gpu)
                    maxdiff=0.;flips=0
                    with np.load(rp) as z:
                        for i in range(len(inp['sids'])):
                            n=int(rr['output_mask'][i].sum());m=rr['pred_masks'][i][:,:n].cpu().numpy();old=z[f'mask_{i}']
                            maxdiff=max(maxdiff,float(np.abs(m-old).max()));flips+=int(np.sum((m>0)!=(old>0)))
                        logdiff=float(np.abs(torch.stack(rr['pred_logits']).cpu().numpy()-z['pred_logits']).max())
                    dump(out/'native_frozen_consistency.json',dict(video=full,mask_max_abs_diff=maxdiff,logit_max_abs_diff=logdiff,threshold_flips=flips))
                    assert maxdiff<1e-4 and logdiff<1e-4 and flips==0,'native cache not equivalent'
                    baseline_verified=True;del gpu,rr
                shutil.copyfile(rp,out/cond/'raw'/(full+'.npz'))
                shutil.copyfile(rp.with_suffix('.json'),out/cond/'raw'/(full+'.json'))
            else:
                net=make_model();gpu=utils.dict_to_device(inp,'cuda:0')
                ts=time.time()
                with torch.no_grad():raw=net(gpu)
                torch.cuda.synchronize();st[cond+'_forward_seconds']=time.time()-ts
                raw_save(out/cond/'raw',full,raw,gpu);del raw,gpu
            torch.cuda.empty_cache()
        stats.append(st);dump(out/'preflight.json',stats);dump(out/'source_hashes.json',hashes)
        print(json.dumps(dict(st,elapsed_seconds=time.time()-start)),flush=True)
        del ni,gi
    dump(out/'source_hashes.json',hashes)
    dump(out/'completion.json',dict(status='PREFLIGHT_PASS' if args.preflight_only else 'RAW_INFERENCE_COMPLETE_NOT_EVALUATED',videos=len(selected),elapsed_seconds=time.time()-start,training=False,native_fresh=args.native_fresh))

if __name__=='__main__':main()
