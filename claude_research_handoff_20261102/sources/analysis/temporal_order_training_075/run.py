"""075 bounded three-arm training; frozen candidates and independently sealed output."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='2'
import sys,time,signal,copy,collections,math
from pathlib import Path
import numpy as np
import torch
from readout import project
from frozen043 import load,dump,sha,tree_sha
from frozen049 import cuda_record,matched,forward_video,masks_intervals
from evaluate import evaluate,rebuild,canonical_sha
from losses import ARMS,focal_rows,rank_domain,rank_rows,loss

HERE=Path(__file__).resolve().parent
REPO=Path('/root/rebinding_project/VrdONE')
JOB=Path('/root/autodl-tmp/VrdONE-data/commander_temporal_order_075')
OLD=Path('/root/autodl-tmp/VrdONE-data/commander_architecture_010/export_v1')
STEPS=500


def prepare_cache(model,inventory,cohorts,out,sources):
    """Keep train bit-exact; audited dev B1 numerical exception only.

The old cache remains the fixed Hungarian matching reference. The exception
is grounded in numeric_probe_v1.json and cannot admit train-mask drift.
"""
    cache_root=Path('/root/autodl-tmp/VrdONE-data/commander_mask_training_043/results_v1')
    caches={};audit=[];video_sets={}
    for role in ('train','adapter_dev'):
        p=cache_root/(role+'_cache.pt');assert sha(p)==inventory[role]['sha256'];sources[str(p)]=sha(p)
        cpu=torch.load(p,map_location='cpu',weights_only=False);assert len(cpu)==inventory[role]['records'];records=[];videos=set()
        for ri,rr in enumerate(cpu):
            r=cuda_record(rr)
            with torch.no_grad():
                base=project(model.predictor.mask_embed,r).masked_fill(~r['valid'][None,...],-10.)
                exact=torch.equal(base,r['masks']);diff=(base-r['masks']).abs()
                assert torch.isfinite(diff).all()
                oldmatch=matched(model,r)
                newmatch=matched(model,dict(r,masks=base))
                assignments_equal=all(torch.equal(a['b'],b['b']) and torch.equal(a['q'],b['q']) and
                    all(torch.equal(ai,bi) and torch.equal(aj,bj) for (ai,aj),(bi,bj) in zip(a['indices'],b['indices']))
                    for a,b in zip(oldmatch,newmatch))
                binary_equal=torch.equal(base.sigmoid()>.5,r['masks'].sigmoid()>.5)
                assert assignments_equal and binary_equal
                if not exact:
                    assert role=='adapter_dev' and ri==22 and len(r['targets'])==1
                    assert float(diff.max())<=4e-6,'075 dev numerical exception exceeded'
                r['match']=oldmatch
            records.append(r);videos.update(x['video_id'] for x in r['metadata'])
            audit.append(dict(role=role,batch=ri,N_pairs=len(r['targets']),N_GT=sum(len(x) for x in r['targets']),
                exact_projection=exact,max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                binary_masks_and_hulls_equal=binary_equal,four_layer_match_equal=assignments_equal,
                fixed_matching_source='unchanged original043 cached masks',
                layer_matches=[dict(b=z['b'].cpu().tolist(),q=z['q'].cpu().tolist(),first=z['first'].cpu().tolist(),last=z['last'].cpu().tolist()) for z in oldmatch]))
        assert sum(len(x['targets']) for x in records)==inventory[role]['N_pairs']
        caches[role]=records;video_sets[role]=videos
    assert not video_sets['train']&video_sets['adapter_dev']
    val=set(n for c in cohorts.values() for n in load(Path(c['export'])/'manifest.json')['video_names'])
    assert not val&(video_sets['train']|video_sets['adapter_dev'])
    dump(out/'training_cache_audit.json',dict(records=audit,video_names={k:sorted(v) for k,v in video_sets.items()}))
    return caches


def verify_retained_sources(caches,out,sources):
    """Check every retained target against its source export and full annotation.

Annotation matching is by predicate and exact sampled/cropped target mask; no
validation data enter training. Complete annotated union is cross-checked with
the retained union for non-censored pairs before ranking is allowed.
"""
    from dataloaders.category import vidor_pred_name_to_id
    receipts=[]
    manifests={'train':'39c0555bda3d99e928aef89a9f346a9b2c418a4e051205729a3838927a2306c5',
               'adapter_dev':'4a9dc55807723c69b2f8d10d73ad827ca3b9c9f2007f2e6607603f8a16db7edc'}
    for role,records in caches.items():
        path=OLD/role/'manifest.json'; assert sha(path)==manifests[role]
        sources[str(path)]=sha(path); manifest=load(path); annotations={}
        assert len(records)==len(manifest['records'])
        for ri,(r,ref) in enumerate(zip(records,manifest['records'])):
            p=Path(ref['path']); assert sha(p)==ref['sha256']; sources[str(p)]=sha(p)
            old=torch.load(p,map_location='cpu',weights_only=False)
            assert r['metadata']==old['metadata']['pair_metadata']
            r['gt_metadata']=[]
            for b,pm in enumerate(r['metadata']):
                assert torch.equal(r['preds'][b].cpu(),old['preds_list'][b])
                assert torch.equal(r['targets'][b].cpu(),old['masks_list'][b])
                full=pm['video_id']
                if full not in annotations:
                    refann=manifest['inputs'][full]; ap=Path(refann['annotation'])
                    assert sha(ap)==refann['annotation_sha256']; sources[str(ap)]=sha(ap); annotations[full]=load(ap)
                ann=annotations[full]; fps=float(ann['fps']); tids=sorted({int(x['tid']) for frame in ann['trajectories'] for x in frame})
                sid,oid=tids[pm['pair_key'][0]],tids[pm['pair_key'][1]]
                origin,stride=pm['sample_frame_origin'],pm['feat_stride']; pair0,pair1=pm['pair_frame_interval']
                candidates=[]; all_identity_episodes=[]; T=r['targets'][b].shape[-1]
                for gi,g in enumerate(ann['relation_instances']):
                    if g['subject_tid']!=sid or g['object_tid']!=oid:continue
                    first=max(0,math.ceil((g['begin_fid']-origin)/stride)); last=min(T,math.ceil((g['end_fid']-origin)/stride))
                    mask=torch.zeros(T,dtype=torch.bool,device=r['targets'][b].device)
                    if last>first:mask[first:last]=True
                    item=(gi,g,vidor_pred_name_to_id[g['predicate']],mask)
                    all_identity_episodes.append(item)
                    if pair0<=g['begin_fid']<g['end_fid']<=pair1:candidates.append(item)
                meta=[]
                for j,(pid,y) in enumerate(zip(r['preds'][b],r['targets'][b])):
                    matches=[(gi,g) for gi,g,p,m in candidates if p==int(pid) and torch.equal(m,y.bool())]
                    assert len(matches)==1,('nonunique exact annotation target',role,ri,b,j,len(matches))
                    gi,g=matches[0]
                    union=torch.stack([m for _,_,p,m in all_identity_episodes if p==int(pid)]).any(0)
                    kept=(r['targets'][b][r['preds'][b]==pid]>0).any(0)
                    assert (kept<=union).all()
                    missing=int(((union & ~kept)&r['valid'][b,0]).sum())
                    if int(pm['censored_positive_cells'])==0:
                        assert missing==0,('Unretained positive cells in uncensored pair',role,ri,b,j,missing)
                    meta.append(dict(video_name=full,gt_index=gi,predicate=g['predicate'],subject_tid=sid,object_tid=oid,
                        relation_sec=(g['end_fid']-g['begin_fid'])/fps,pair_support_sec=(pair1-pair0)/fps,
                        original_interval=[g['begin_fid'],g['end_fid']],censored_pair=int(pm['censored_positive_cells'])>0,
                        annotation_union_missing_retained_valid_cells=missing,
                        same_identity_same_predicate_all_annotation_episodes=sum(p==int(pid) for _,_,p,_ in all_identity_episodes),
                        same_identity_same_predicate_pair_contained_episodes=sum(p==int(pid) for _,_,p,_ in candidates)))
                r['gt_metadata'].append(meta)
                receipts.append(dict(role=role,batch=ri,pair_row=b,video_name=full,cache_targets_equal_source=True,
                    N_GT=len(meta),censored_positive_cells=int(pm['censored_positive_cells']),annotations=meta))
            del old
    dump(out/'annotation_alignment.json',receipts)


def attach_domains(caches,out):
    audit=[]
    for role,records in caches.items():
        for ri,r in enumerate(records):
            for layer,m in enumerate(r['match']):
                domains=[]; metadata=[]
                for b,(_,target_indices) in enumerate(m['indices']):
                    for j in target_indices.tolist():
                        pid=r['preds'][b][j]; y=r['targets'][b][j]; valid=r['valid'][b,0]
                        union=(r['targets'][b][r['preds'][b]==pid]>0).any(0)
                        assert ((y>0)<=union).all()
                        d=rank_domain(y,union,valid,int(r['metadata'][b]['censored_positive_cells'])>0)
                        assert not set(d['positive'].tolist()) & set(d['negative'].tolist())
                        assert valid[d['positive']].all() and valid[d['negative']].all()
                        assert len(d['positive'])<=16 and len(d['negative'])<=16
                        domains.append(d); metadata.append(r['gt_metadata'][b][j])
                        if layer==3:
                            audit.append(dict(role=role,batch=ri,pair_row=b,target_index=j,**metadata[-1],
                                reason=d['reason'],positive_cells=d['positive_cells'],negative_cells=d['negative_cells'],
                                sampled_positive=d['positive'].tolist(),sampled_negative=d['negative'].tolist()))
                assert len(domains)==m['N']==len(m['b'])
                m['rank_domains']=domains; m['gt_metadata']=metadata
    summary={}
    for role in caches:
        rows=[r for r in audit if r['role']==role]; eligible=[r for r in rows if r['reason']=='eligible']
        pk=lambda r:(r['batch'],r['pair_row'])
        groups={}
        for name,fn in dict(all=lambda r:True,short=lambda r:r['relation_sec']<3,subsecond=lambda r:r['relation_sec']<1,
            long=lambda r:r['relation_sec']>=3,long_coexist_short=lambda r:r['relation_sec']<3 and r['pair_support_sec']>=10).items():
            rr=[r for r in rows if fn(r)];groups[name]=dict(N_GT=len(rr),eligible=sum(r['reason']=='eligible' for r in rr),reasons=dict(collections.Counter(r['reason'] for r in rr)))
        summary[role]=dict(N_GT=len(rows),N_pairs=len({pk(r) for r in rows}),N_videos=len({r['video_name'] for r in rows}),
            eligible_GT=len(eligible),eligible_pairs=len({pk(r) for r in eligible}),eligible_videos=len({r['video_name'] for r in eligible}),
            censored_pairs=len({pk(r) for r in rows if r['censored_pair']}),censored_pair_GT=sum(r['censored_pair'] for r in rows),groups=groups)
    dump(out/'ranking_domain_audit.json',dict(summary=summary,rows=audit))
    assert summary['train']['groups']['long_coexist_short']['eligible']>0, 'No ranking-eligible target-like training event'
    return summary


def compute_scales(model,records,out):
    sums=np.zeros(5,dtype=np.float64); count=eligible_count=0
    with torch.no_grad():
        for r in records:
            values=project(model.predictor.mask_embed,r).masked_fill(~r['valid'][None,...],-10.)
            for layer,m in enumerate(r['match']):
                z=values[layer,m['b'],m['q']]; a,b=focal_rows(z,m['y'],m['valid']); rank,eligible=rank_rows(z,m['rank_domains'])
                sums+=np.array([a.double().sum().item(),b.double().sum().item(),b[eligible].double().sum().item(),rank[eligible].double().sum().item(),a[eligible].double().sum().item()])
                count+=len(a);eligible_count+=int(eligible.sum())
    assert np.isfinite(sums).all() and min(sums[:4])>0 and eligible_count>0
    kb=sums[0]/sums[1]; kr=kb*sums[2]/sums[3]
    assert math.isfinite(kb) and math.isfinite(kr) and kb>0 and kr>0
    result=dict(k_B=kb,k_R=kr,margin_logit_units=1.,ordered_mix=.5,all_event_layer_count=count,eligible_event_layer_count=eligible_count,
        original_focal_sum=sums[0],balanced_focal_sum=sums[1],eligible_balanced_sum=sums[2],eligible_rank_sum=sums[3],eligible_original_sum=sums[4],
        source='initial complete training cache only; four layers and targets equally weighted; not gradient matching')
    dump(out/'frozen_scales.json',result); return result


def snapshot(model,heads,caches,scales,out,stage):
    report={}
    with torch.no_grad():
        for role,records in caches.items():
            raw=[];rows=[];terms={a:[] for a in ('baseline',)+ARMS}
            for ri,r in enumerate(records):
                saved={};m=r['match'][-1];valid=m['valid'].bool()
                for arm,head in dict(baseline=model.predictor.mask_embed,**heads).items():
                    values=project(head,r).masked_fill(~r['valid'][None,...],-10.)
                    saved[arm]=values.cpu(); z=values[-1,m['b'],m['q']];a,b=focal_rows(z,m['y'],valid);rank,eligible=rank_rows(z,m['rank_domains'])
                    _,comp=loss(model,values,r,'original' if arm=='baseline' else arm,scales)
                    terms[arm].append(dict(N_GT=m['N'],components=comp.cpu().tolist()))
                    first,last=masks_intervals(z,valid)
                    for j,meta in enumerate(m['gt_metadata']):
                        x0,x1=int(first[j]),int(last[j]);g0,g1=int(m['first'][j]),int(m['last'][j])
                        overlap=max(0,min(x1,g1)-max(x0,g0)+1);union=max(x1,g1)-min(x0,g0)+1
                        d=m['rank_domains'][j]
                        margin=float(z[j,d['positive']].mean()-z[j,d['negative']].mean()) if eligible[j] else None
                        rows.append(dict(batch=ri,arm=arm,query=int(m['q'][j]),pair_row=int(m['b'][j]),**meta,
                            sampled_hull=[x0,x1],target_hull=[g0,g1],visible_tIoU=overlap/union if x1>x0 else 0.,
                            original_focal=float(a[j]),balanced_focal=float(b[j]),rank_loss=float(rank[j]),ranking_eligible=bool(eligible[j]),mean_positive_minus_negative_logit=margin))
                raw.append(saved)
            p=out/'fixed_cache_masks'/(role+'_'+stage+'.pt');p.parent.mkdir(exist_ok=True);torch.save(raw,p)
            dump(out/(role+'_'+stage+'_diagnostics.json.gz'),rows)
            report[role]=dict(raw_masks_sha256=sha(p),terms=terms)
    dump(out/('fixed_snapshot_'+stage+'.json'),report)


def initial_gradients(model,heads,records,scales,out):
    # One deterministic full-training aggregate gradient for each arm.
    n=sum(r['match'][0]['N'] for r in records); grads={};values={}
    for arm,h in heads.items():
        h.zero_grad(set_to_none=True); total=0.
        for r in records:
            v=project(h,r).masked_fill(~r['valid'][None,...],-10.)
            l,_=loss(model,v,r,arm,scales,check_original=arm=='original')
            weight=r['match'][0]['N']/n;(weight*l).backward();total+=float(l.detach())*weight
        g=torch.cat([p.grad.detach().flatten() for p in h.parameters()]);grads[arm]=g
        assert torch.isfinite(g).all() and math.isfinite(total) and torch.isfinite(g.norm())
        values[arm]=dict(full_train_mean_loss=total,parameter_gradient_norm=float(g.norm()))
        h.zero_grad(set_to_none=True)
    cos={a:{b:float(torch.nn.functional.cosine_similarity(grads[a],grads[b],dim=0)) for b in ARMS} for a in ARMS}
    assert all(math.isfinite(v) for x in cos.values() for v in x.values())
    dump(out/'initial_gradient_audit.json',dict(arms=values,cosines=cos,N_GT=n,scope='full train event-weighted gradient; not SGD schedule-gradient equivalence'))


def train(model,heads,caches,scales,out):
    opts={a:torch.optim.AdamW(h.parameters(),lr=1e-5,weight_decay=.05) for a,h in heads.items()}
    rng=np.random.default_rng(75);order=[]
    while len(order)<STEPS:order+=rng.permutation(len(caches['train'])).tolist()
    order=order[:STEPS]
    dump(out/'training_schedule.json',dict(seed=75,steps=STEPS,order=order,lr=1e-5,weight_decay=.05,clip_norm=1.,parameter_count={a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()}))
    for a,h in heads.items():torch.save(h.state_dict(),out/(a+'_initial.pt'))
    snapshot(model,heads,caches,scales,out,'initial');initial_gradients(model,heads,caches['train'],scales,out)
    history=[];clips={a:0 for a in ARMS};start=time.monotonic()
    for step,ri in enumerate(order,1):
        r=caches['train'][ri]
        for a,h in heads.items():
            opts[a].zero_grad(set_to_none=True)
            v=project(h,r).masked_fill(~r['valid'][None,...],-10.);total,parts=loss(model,v,r,a,scales)
            assert torch.isfinite(total);total.backward();norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.)
            assert torch.isfinite(norm);clips[a]+=int(norm>1);opts[a].step()
            history.append(dict(step=step,batch=ri,arm=a,loss=float(total.detach()),grad_norm_before_clip=float(norm),components=parts.cpu().tolist()))
        if step%50==0:dump(out/'training_history.json',history);print('TRAIN075',step,round(time.monotonic()-start,1),flush=True)
    for a,h in heads.items():torch.save(dict(state=h.state_dict(),arm=a,seed=75,step=STEPS),out/(a+'_step500.pt'))
    dump(out/'clipping_summary.json',dict(steps_per_arm=STEPS,clip_trigger_steps=clips));snapshot(model,heads,caches,scales,out,'final')


def predict_video(model,heads,data):
    g,selected,base,records,mapping=data
    intervals={a:[] for a in ('baseline',)+ARMS};lookup={a:{} for a in ARMS};fallbacks={a:0 for a in ARMS}
    with torch.no_grad():
        for r,ids in zip(records,mapping):
            for a,h in heads.items():
                values=project(h,r)[-1]
                for row,pi in enumerate(ids):
                    valid=r['valid'][row].expand(values.shape[1],-1);first,last=masks_intervals(values[row],valid)
                    lookup[a][pi]=(first.cpu().tolist(),last.cpu().tolist())
        for c,b in zip(selected,base):
            pi,q=c['pair_idx'],c['query'];p=g['pairs'][pi];origin=p['pair_start']+p['offset']
            intervals['baseline'].append(b['duration'])
            for a in ARMS:
                first,last=lookup[a][pi];d=[origin+4*first[q],origin+4*last[q]+1]
                if last[q]<=first[q]:d=b['duration'];fallbacks[a]+=1
                intervals[a].append(d)
    hashes={a:canonical_sha(rebuild(base,selected,g,v)) for a,v in intervals.items()}
    assert hashes['baseline']==canonical_sha(base)
    return dict(N_candidates=len(base),durations=intervals,canonical_prediction_sha256=hashes,fallback_candidates=fallbacks,source_forward_batches=len(records))


def run():
    start=time.monotonic();phase='SOURCE_BINDING';out=JOB/'results_v2';out.mkdir(parents=True,exist_ok=False);sources={}
    def timeout(*_):raise TimeoutError('075 1800second total workflow budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        for n,h in load(HERE/'software_freeze.json').items():assert sha(HERE/n)==h,n
        cohorts=load(HERE/'cohorts.json');contract=load(HERE/'source_contract043.json');inventory=load(HERE/'CACHE_INVENTORY.json')
        for rel,h in contract['repo_sources'].items():assert sha(REPO/rel)==h;sources[str(REPO/rel)]=h
        cp=REPO/'experiments/vrdone_vidor/ckpt_vidor.pth';assert sha(cp)==contract['checkpoint_sha256'];sources[str(cp)]=sha(cp)
        manifests={k:load(Path(c['export'])/'manifest.json') for k,c in cohorts.items()}
        for k,c in cohorts.items():assert sha(Path(c['export'])/'manifest.json')==c['manifest_sha256']
        names=[n for m in manifests.values() for n in m['video_names']];assert len(names)==len(set(names))==128
        cfgpath=Path(cohorts['047']['export'])/'effective_config.json';assert sha(cfgpath)==manifests['047']['outputs']['effective_config.json']['sha256'];cfg=load(cfgpath)
        os.chdir(REPO);sys.path.insert(0,str(REPO))
        from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        import utils.misc as utils
        utils.set_seed(75);torch.set_num_threads(2);model=MaskVRD(cfg['model_config'],'cuda:0').eval();model._config_eval(cfg['inference_config'])
        state=torch.load(cp,map_location='cpu',weights_only=False)[contract['checkpoint_key']];model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
        model.cuda().requires_grad_(False);before=tree_sha(model.state_dict());heads={a:copy.deepcopy(model.predictor.mask_embed).requires_grad_(True) for a in ARMS}
        initial_hashes={a:tree_sha(h.state_dict()) for a,h in heads.items()}
        parameter_counts={a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()}
        assert len(set(initial_hashes.values()))==1 and len(set(parameter_counts.values()))==1
        ds=VidOR.__new__(VidOR)
        for k in ('with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset'):setattr(ds,k,cfg['dataset_config'][k])
        phase='ZERO_UPDATE_NATIVE_GATE';first_name=manifests['047']['video_names'][0]
        first=forward_video(model,ds,utils,Path(cohorts['047']['export']),manifests['047'],first_name,sources)
        assert first[2];zero=predict_video(model,heads,first)
        assert len(set(zero['canonical_prediction_sha256'].values()))==1
        dump(out/'zero_update_gate.json',dict(video_name=first_name,source_hulls_exact=True,all_three_untrained_heads_exact=True,N_candidates=len(first[2]),model_state_sha256=before,initial_head_sha256=initial_hashes,head_parameter_counts=parameter_counts))
        phase='CACHED_MATCHING_AND_DOMAINS';caches=prepare_cache(model,inventory,cohorts,out,sources)
        verify_retained_sources(caches,out,sources);attach_domains(caches,out);scales=compute_scales(model,caches['train'],out)
        phase='TRAIN';train(model,heads,caches,scales,out);del caches
        assert tree_sha(model.state_dict())==before
        phase='NATIVE_INFERENCE';audits=[]
        for k,c in cohorts.items():
            for i,full in enumerate(manifests[k]['video_names']):
                data=first if k=='047' and i==0 else forward_video(model,ds,utils,Path(c['export']),manifests[k],full,sources)
                z=predict_video(model,heads,data);z.update(video_name=full,cohort=k);dump(out/'intervals'/k/(full+'.json.gz'),z)
                audits.append(dict(cohort=k,video_name=full,N_candidates=z['N_candidates'],forward_batches=z['source_forward_batches'],fallback_candidates=z['fallback_candidates']))
                if k=='047' and i==0:first=None
                del data,z
                if (i+1)%8==0:print('NATIVE075',k,i+1,'elapsed',round(time.monotonic()-start,1),flush=True)
        assert tree_sha(model.state_dict())==before
        seal={str(p.relative_to(out)):sha(p) for p in (out/'intervals').rglob('*.gz')};assert len(seal)==128
        dump(out/'intervals_seal_before_GT.json',seal);dump(out/'native_audits.json',audits)
        phase='OFFICIAL_EVALUATION';result=evaluate(out,cohorts)
        result.update(complete=True,elapsed_seconds=time.monotonic()-start,training_steps_per_arm=STEPS,seed=75,source_sha256=sources,
            original_model_state_unchanged=True,scope='Fixed candidates on 128 exposed validation videos; no independent generalization claim',independent_review='PENDING')
        dump(out/'summary.json',result);dump(out/'status.json',dict(complete=True,decision=result['decision']));print('COMPLETE075',result['decision'],flush=True)
    except Exception as e:
        dump(out/'status.json',dict(complete=False,phase=phase,error=repr(e),elapsed_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)


if __name__=='__main__':run()
