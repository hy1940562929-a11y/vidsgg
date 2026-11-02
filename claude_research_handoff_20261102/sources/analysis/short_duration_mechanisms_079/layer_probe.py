#!/usr/bin/env python3
"""079-B: equal-capacity frozen-layer annotation readouts. No official recall.

Real execution requires the original CUDA environment and source data. This
script never installs packages, edits the repository, or launches full inference.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import time

from loss_probe import (Paths, MANIFEST_SHA, digest, ensure, load_json, write_json,
                        verify_file, import_vendor, prepare_caches, bind_annotations)
from layer_math import position_features, annotation_union, components, iou, average_precision, video_bootstrap

HERE = Path(__file__).resolve().parent
ARMS = ('fine', 'fused', 'position')


def reconstruct(torch, np, model, VidOR, caches, paths, out, sources, device, vendor):
    """One original-loader forward per batch; capture both layers together."""
    audit, exclusions, result = [], [], {}
    for role, records in caches.items():
        mp = paths.get('train_export_root')/role/'manifest.json'
        verify_file(mp, MANIFEST_SHA[role], sources)
        manifest = load_json(mp)
        ensure(manifest.get('source_split') == 'training' and manifest.get('role') == role,
               '079-B requires genuine official-training-source manifests')
        cfg = manifest['effective_dataset_config']
        ds = VidOR.__new__(VidOR)
        for key in ('with_clip_feature', 'feat_stride', 'max_seq_len', 'cut_max_preds', 'proposal_max_preds'):
            setattr(ds, key, cfg[key])
        ensure(ds.feat_stride == 4 and ds.max_seq_len == 512, 'Frozen loader grid changed')
        loader = paths.remap(cfg['cache_dir'])/(cfg['cache_tag']+'_VidOR_training')
        raw_cache, annotations, selected = {}, {}, []
        for ri, (record, ref) in enumerate(zip(records, manifest['records'])):
            source = paths.remap(ref['path'])
            verify_file(source, ref['sha256'], sources)
            old = torch.load(source, map_location='cpu', weights_only=False)
            ensure(record['metadata'] == old['metadata']['pair_metadata'], 'Pair metadata drift')
            xs, draws_all = [], []
            for row, pm in enumerate(record['metadata']):
                video = pm['video_id']
                if video not in raw_cache:
                    p = loader/(video+'.pkl')
                    sources[str(p)] = digest(p)
                    with p.open('rb') as f:
                        raw_cache[video] = pickle.load(f)
                raw = raw_cache[video]
                index = [list(k) for k in raw['relation_keys']].index(pm['pair_key'])
                random.seed(pm['sampling_seed']); np.random.seed(pm['sampling_seed']); torch.manual_seed(pm['sampling_seed'])
                original_randint, draws = random.randint, []
                def traced(a, b):
                    v = original_randint(a, b); draws.append([int(a), int(b), int(v)]); return v
                random.randint = traced
                try:
                    item = ds._train_getitem(raw, (index, index+1))
                finally:
                    random.randint = original_randint
                ensure(draws == pm['randint_draws'], f'Loader random draws changed: {role}/{ri}/{row}')
                ensure(len(item['so_features_list']) == 1, 'Unexpected loader pair count')
                ensure(torch.equal(item['preds_list'][0], old['preds_list'][row]) and
                       torch.equal(item['masks_list'][0], old['masks_list'][row]), 'Reconstructed targets changed')
                feature = item['so_features_list'][0]
                ensure(tuple(feature.shape) == (2069, int(record['valid'][row].sum())), 'Reconstructed input shape changed')
                xs.append(feature); draws_all.append(draws)
            batch = len(xs)
            padded = torch.zeros(batch, 2069, 512, device=device)
            for b, x in enumerate(xs):
                padded[b, :, :x.shape[-1]] = x.to(device)
            captured = {}
            def capture_fine(module, args, output):
                ensure('fine' not in captured, 'Fine lateral hook fired more than once')
                captured['fine'] = output.detach().clone()
            handle = model.neck.lateral_norms[0].register_forward_hook(capture_fine)
            try:
                with torch.no_grad(), vendor['frozen043'].Capture(model) as cap:
                    model._mask_vrd(padded, record['valid'])
            finally:
                handle.remove()
            ensure(len(cap.records) == 1 and 'fine' in captured, 'Expected one native forward with fine hook')
            fresh = cap.records[0]
            ensure(tuple(captured['fine'].shape) == tuple(fresh['f'].shape) == (batch, 256, 512), 'Fine/fused capacity mismatch')
            differences = {k: float((fresh[k]-record[k]).abs().max()) for k in ('f','cls','masks')}
            ensure(all(v <= 4e-6 for v in differences.values()), f'Reconstructed forward exceeds numerical tolerance: {differences}')
            ensure(torch.equal(fresh['valid'], record['valid']), 'Validity changed')
            ensure(torch.equal(fresh['masks'].sigmoid() > .5, record['masks'].sigmoid() > .5), 'Native binary masks changed')
            ensure(torch.equal(fresh['cls'].softmax(-1)[..., 1:].topk(6,-1).indices,
                               record['cls'].softmax(-1)[..., 1:].topk(6,-1).indices), 'Native predicate ordering changed')
            newer = dict(record, masks=fresh['masks'], cls=fresh['cls'])
            matched = vendor['frozen049'].matched(model, newer)
            ensure(all(torch.equal(a['b'],b['b']) and torch.equal(a['q'],b['q']) and
                       all(torch.equal(ai,bi) and torch.equal(aj,bj) for (ai,aj),(bi,bj) in zip(a['indices'],b['indices']))
                       for a,b in zip(record['match'],matched)), 'Reconstructed matching changed')
            picked = []
            for row, pm in enumerate(record['metadata']):
                video, meta = pm['video_id'], record['gt_metadata'][row]
                ensure(bool(meta), 'Pair has no retained target metadata')
                if video not in annotations:
                    ar = manifest['inputs'][video]
                    ap = paths.remap(ar['annotation'])
                    verify_file(ap, ar['annotation_sha256'], sources)
                    annotations[video] = load_json(ap)
                ann = annotations[video]
                sid, oid = meta[0]['subject_tid'], meta[0]['object_tid']
                origin, stride, n = meta[0]['origin'], meta[0]['stride'], meta[0]['N_valid']
                from dataloaders.category import vidor_pred_name_to_id
                episodes = [dict(predicate_id=vidor_pred_name_to_id[g['predicate']], begin=int(g['begin_fid']), end=int(g['end_fid']))
                            for g in ann['relation_instances'] if g['subject_tid']==sid and g['object_tid']==oid]
                y = annotation_union(episodes, origin, stride, n)
                retained = np.zeros_like(y)
                for p, target in zip(record['preds'][row].tolist(), record['targets'][row]):
                    retained[p-1] |= target[:n].bool().cpu().numpy()
                missing = int((y & ~retained).sum())
                ensure(not (retained & ~y).any(), 'Retained target exceeds complete annotation union')
                ensure(missing == int(pm['censored_positive_cells']), 'Censoring count does not reproduce original010')
                excluded = missing > 0
                exclusions.append(dict(role=role,batch=ri,row=row,video_name=video,N_GT=len(meta),
                    censored_positive_cells=missing,excluded=excluded,reason='censored pair' if excluded else None,
                    sample_origin=origin,stride=stride,N_valid=n))
                if excluded:
                    continue
                event_meta = [dict(e, predicate_id=int(p)) for e,p in zip(meta, record['preds'][row].tolist())]
                picked.append(dict(batch=ri,row=row,video_name=video,origin=origin,stride=stride,N_valid=n,
                    targets=torch.from_numpy(y.astype(np.float32)), events=event_meta,
                    features=dict(fine=captured['fine'][row,:,:n].cpu(), fused=fresh['f'][row,:,:n].cpu(),
                                  position=torch.from_numpy(position_features(n)))))
            selected.append(picked)
            audit.append(dict(role=role,batch=ri,N_pairs=batch,N_retained_pairs=len(picked),
                historical_max_abs=differences,valid_exact=True,binary_exact=True,semantic_order_exact=True,
                matching_exact=True,loader_random_draws=draws_all))
            if (ri+1)%20 == 0:
                print('LAYER079_CAPTURE',role,ri+1,'/',len(records),flush=True)
            del old, fresh, padded, cap, captured
        result[role] = selected
        path = out/(role+'_layer_cache.pt')
        torch.save(selected,path)
        write_json(out/(role+'_layer_cache_receipt.json'),dict(path=str(path),sha256=digest(path)))
        del raw_cache
    write_json(out/'layer_capture_audit.json',audit)
    write_json(out/'censoring_audit.json',exclusions)
    return result, exclusions


def fit_train_statistics(torch, batches):
    sums = {a:torch.zeros(256,dtype=torch.float64) for a in ARMS}
    squares = {a:torch.zeros(256,dtype=torch.float64) for a in ARMS}
    positives, total = torch.zeros(50,dtype=torch.float64),0
    for pairs in batches:
        for pair in pairs:
            n = pair['N_valid']; total += n; positives += pair['targets'].double().sum(1)
            for arm in ARMS:
                x = pair['features'][arm].double()
                sums[arm] += x.sum(1); squares[arm] += x.square().sum(1)
    ensure(total>0,'No uncensored training frames')
    means = {a:(sums[a]/total).float() for a in ARMS}
    stds = {a:(squares[a]/total-(sums[a]/total).square()).clamp_min(0).sqrt().clamp_min(1e-5).float() for a in ARMS}
    negative = total-positives
    supported = (positives>0)&(negative>0)
    ensure(bool(supported.any()), 'No train-supported predicates')
    # Class-balanced BCE: expected positive and negative mass each one-half,
    # using training counts only. Unsupported heads never enter train/eval loss.
    positive_weight = torch.where(supported, negative/positives.clamp_min(1), torch.ones_like(positives)).float()
    normalizer = (2*negative/total).clamp_min(1e-8).float()
    return dict(means=means,stds=stds,positive_weight=positive_weight,normalizer=normalizer,
                supported=supported,positive_counts=positives.long(),negative_counts=negative.long(),N_frames=total)


def evaluate(torch,np,heads,batches,stats,out,role,reps):
    supported = stats['supported'].numpy()
    frame_data={a:{p:[] for p in np.flatnonzero(supported)} for a in ARMS}
    rows, excluded_events = [],[]
    device=next(heads['fine'].parameters()).device
    with torch.no_grad():
        for pairs in batches:
            for pair in pairs:
                target=pair['targets'].numpy().astype(bool)
                values={}
                for arm,head in heads.items():
                    x=(pair['features'][arm].to(device)-stats['means'][arm].to(device)[:,None])/stats['stds'][arm].to(device)[:,None]
                    logits=head(x.T).T
                    probs=logits.sigmoid().cpu().numpy(); bits=(logits.sigmoid()>.5).cpu().numpy()
                    values[arm]=(probs,bits)
                    for p in frame_data[arm]:frame_data[arm][p].append((target[p],probs[p],bits[p]))
                for event in pair['events']:
                    pid=event['predicate_id']-1
                    if not supported[pid]:
                        excluded_events.append(dict(role=role,batch=pair['batch'],row=pair['row'],**event,reason='no positive/negative training support'))
                        continue
                    row=dict(role=role,batch=pair['batch'],pair_row=pair['row'],**event,arms={})
                    for arm,(_,bits) in values.items():
                        spans=components(bits[pid],pair['origin'],pair['stride'])
                        truth=event['original_interval']
                        best=max(spans,key=lambda v:iou(v,truth)) if spans else None
                        overlap=iou(best,truth) if best else 0.
                        row['arms'][arm]=dict(hit=overlap>=.5,best_tIoU=overlap,prediction=best,
                            start_abs_frames=abs(best[0]-truth[0]) if best else None,
                            end_abs_frames=abs(best[1]-truth[1]) if best else None,
                            endpoint_abs_seconds=(abs(best[0]-truth[0])+abs(best[1]-truth[1]))/(2*event['fps']) if best else None)
                    rows.append(row)
    predicates={}
    for arm,items in frame_data.items():
        predicates[arm]={}
        for p,parts in items.items():
            y=np.concatenate([v[0] for v in parts]) if parts else np.array([],bool)
            scores=np.concatenate([v[1] for v in parts]) if parts else np.array([])
            bits=np.concatenate([v[2] for v in parts]) if parts else np.array([],bool)
            tp=int((y&bits).sum());fp=int((~y&bits).sum());fn=int((y&~bits).sum());tn=int((~y&~bits).sum())
            predicates[arm][str(p+1)]=dict(N_positive=int(y.sum()),N_negative=int((~y).sum()),TP=tp,FP=fp,FN=fn,TN=tn,
                AP=average_precision(y,scores) if len(y) else None,
                F1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None)
    groups={}
    selectors=dict(all=lambda r:True,subsecond=lambda r:r['relation_sec']<1,short=lambda r:r['relation_sec']<3,
                   one_to_three=lambda r:1<=r['relation_sec']<3,long=lambda r:r['relation_sec']>=3,
                   three_to_ten=lambda r:3<=r['relation_sec']<10,ten_plus=lambda r:r['relation_sec']>=10)
    for name,select in selectors.items():
        chosen=[r for r in rows if select(r)]
        arms={}
        for a in ARMS:
            endpoints=[r['arms'][a]['endpoint_abs_seconds'] for r in chosen if r['arms'][a]['endpoint_abs_seconds'] is not None]
            arms[a]=dict(hits=sum(r['arms'][a]['hit'] for r in chosen),N_GT=len(chosen),
                mean_best_tIoU=float(np.mean([r['arms'][a]['best_tIoU'] for r in chosen])) if chosen else None,
                endpoint_error_mean_seconds=float(np.mean(endpoints)) if endpoints else None,
                endpoint_error_N_predictions=len(endpoints))
        groups[name]=dict(arms=arms,comparisons={a+'_vs_'+b:video_bootstrap(chosen,a,b,reps)
                                              for a,b in (('fine','fused'),('fine','position'),('fused','position'))})
    same_predicate={}
    for p in sorted({r['predicate_id'] for r in rows}):
        same_predicate[str(p)]={}
        for name in ('short','long'):
            chosen=[r for r in rows if r['predicate_id']==p and selectors[name](r)]
            same_predicate[str(p)][name]={a:dict(hits=sum(r['arms'][a]['hit'] for r in chosen),N_GT=len(chosen)) for a in ARMS}
    write_json(out/(role+'_per_gt.json.gz'),rows)
    write_json(out/(role+'_unsupported_events.json'),excluded_events)
    write_json(out/(role+'_frame_metrics.json'),predicates)
    write_json(out/(role+'_same_predicate_duration.json'),same_predicate)
    with (out/(role+'_per_gt.csv')).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['role','batch','pair_row','video_name','gt_index','predicate','relation_sec','arm','hit','best_tIoU','start_abs_frames','end_abs_frames'])
        writer.writeheader()
        for r in rows:
            for arm,v in r['arms'].items():
                writer.writerow({k:arm if k=='arm' else v[k] if k in v else r[k] for k in writer.fieldnames})
    return dict(groups=groups,N_supported_events=len(rows),N_unsupported_events=len(excluded_events),
                event_metric='per retained annotation occurrence; best connected component, diagnostic not one-to-one recall',
                frame_macro_AP={a:float(np.mean([v['AP'] for v in predicates[a].values() if v['AP'] is not None]))
                                if any(v['AP'] is not None for v in predicates[a].values()) else None for a in ARMS})


def run(config_file,output):
    begin=time.monotonic();out=Path(output).expanduser().resolve();ensure(not out.exists(),'Choose a fresh output path');out.mkdir(parents=True)
    phase='CONFIG'
    try:
        cfg=load_json(config_file);options=cfg['layer_probe'];seed=int(options['seed']);steps=int(options['steps']);lr=float(options['learning_rate'])
        ensure((seed,steps,lr)==(79,500,.001),'079-B fixes seed79, 500 steps, lr0.001')
        def alarm(*_):raise TimeoutError('079-B wall budget exceeded')
        signal.signal(signal.SIGALRM,alarm);signal.alarm(int(options['wall_seconds']))
        paths=Paths(cfg,config_file);vendorpath=paths.get('vendor078');contract=load_json(vendorpath/'source_contract043.json')
        sources={str(Path(config_file).resolve()):digest(config_file)}
        for name in ('layer_probe.py','layer_math.py','loss_probe.py','loss_math.py'):
            sources[str(HERE/name)]=digest(HERE/name)
        for name,expected in load_json(vendorpath/'runtime_manifest.json')['files'].items():verify_file(vendorpath/name,expected,sources)
        for rel,expected in contract['repo_sources'].items():verify_file(paths.repo/rel,expected,sources)
        checkpoint=paths.get('checkpoint');verify_file(checkpoint,contract['checkpoint_sha256'],sources)
        cohorts=load_json(vendorpath/'cohorts.json');inventory=load_json(vendorpath/'CACHE_INVENTORY.json')
        export=paths.remap(cohorts['047']['export']);verify_file(export/'manifest.json',cohorts['047']['manifest_sha256'],sources)
        manifest=load_json(export/'manifest.json');verify_file(export/'effective_config.json',manifest['outputs']['effective_config.json']['sha256'],sources)
        modelcfg=load_json(export/'effective_config.json')
        import numpy as np
        import torch
        device=cfg.get('device','cuda:0');ensure(str(device).startswith('cuda') and torch.cuda.is_available(),'079-B real execution requires authorized CUDA')
        torch.cuda.set_device(torch.device(device));torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        vendor=import_vendor(vendorpath,paths.repo)
        from models.maskvrd import MaskVRD
        from dataloaders.vidor import VidOR
        model=MaskVRD(modelcfg['model_config'],device).eval();model._config_eval(modelcfg['inference_config'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)[contract['checkpoint_key']]
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
        model.to(device).requires_grad_(False);initial_model=vendor['frozen043'].tree_sha(model.state_dict())
        phase='SOURCE_CACHE_AND_ANNOTATION_AUDIT'
        caches=prepare_caches(torch,model,inventory,paths,cohorts,vendor,out,sources,device)
        bind_annotations(torch,caches,paths,sources,out)
        phase='CAPTURE_LAYERS'
        layers,excluded=reconstruct(torch,np,model,VidOR,caches,paths,out,sources,device,vendor);del caches
        ensure(vendor['frozen043'].tree_sha(model.state_dict())==initial_model,'Frozen model changed')
        stats=fit_train_statistics(torch,layers['train'])
        torch.save(stats,out/'train_only_statistics.pt')
        write_json(out/'predicate_support.json',dict(N_training_frames=stats['N_frames'],
            supported_predicate_ids=(torch.where(stats['supported'])[0]+1).tolist(),
            training_positive_frames=stats['positive_counts'].tolist(),training_negative_frames=stats['negative_counts'].tolist()))
        # Reset after reconstruction's loader RNG draws so probe initialization
        # does not accidentally depend on the last selected pair.
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        initial=torch.nn.Linear(256,50).to(device)
        heads={a:copy.deepcopy(initial) for a in ARMS}
        hashes={a:vendor['frozen043'].tree_sha(h.state_dict()) for a,h in heads.items()}
        ensure(len(set(hashes.values()))==1,'Heads do not have identical initialization')
        optimizers={a:torch.optim.AdamW(h.parameters(),lr=lr,weight_decay=.01) for a,h in heads.items()}
        available=[i for i,b in enumerate(layers['train']) if b]
        ensure(bool(available),'No uncensored training batch')
        rng=np.random.default_rng(seed);order=[]
        while len(order)<steps:order.extend(rng.permutation(available).tolist())
        order=order[:steps]
        protocol=dict(steps=steps,seed=seed,learning_rate=lr,weight_decay=.01,clip_norm=1.,
            initialization_sha256=hashes,schedule=order,trainable_parameters=12850,
            supported_predicates=(torch.where(stats['supported'])[0]+1).tolist(),
            annotation_negatives_are_semantic_absence=False,full_benchmark=False,
            selection='same frozen043 crops; censored whole pairs excluded equally',
            feature_normalization='training frames only, separately per layer',
            position_basis='constant, normalized crop time, squared time, sin/cos harmonics1..3; remaining channels0',
            evaluation_threshold=.5,minimum_component_original_frames=5,
            event_scope='retained annotation occurrences; any connected component tIoU>=0.5; not official one-to-one detection',
            development_checkpoint_selection=False)
        write_json(out/'effective_protocol.json',protocol);write_json(out/'effective_config.json',cfg);write_json(out/'source_sha256_before.json',sources)
        history=[];support=stats['supported'].to(device);pw=stats['positive_weight'].to(device);normal=stats['normalizer'].to(device)
        phase='TRAIN_EQUAL_CAPACITY_READOUTS'
        for step,bi in enumerate(order,1):
            batch=layers['train'][bi];target=torch.cat([p['targets'].T for p in batch]).to(device)
            for arm,head in heads.items():
                x=torch.cat([p['features'][arm].T for p in batch]).to(device)
                x=(x-stats['means'][arm].to(device))/stats['stds'][arm].to(device)
                optimizers[arm].zero_grad(set_to_none=True)
                logits=head(x)
                element=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=pw,reduction='none')/normal
                loss=element[:,support].mean();ensure(bool(torch.isfinite(loss)),'Nonfinite readout loss');loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);ensure(bool(torch.isfinite(norm)),'Nonfinite readout gradient');optimizers[arm].step()
                history.append(dict(step=step,batch=bi,arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),N_frames=len(target)))
            if step%50==0:
                write_json(out/'training_history.json',history);print('LAYER079_TRAIN',step,'elapsed',round(time.monotonic()-begin,1),flush=True)
        for arm,head in heads.items():torch.save(dict(state=head.state_dict(),arm=arm,protocol=protocol),out/(arm+'_step500.pt'))
        phase='EVALUATE_ANNOTATION_RECOVERABILITY'
        summaries={r:evaluate(torch,np,heads,layers[r],stats,out,r,int(options.get('bootstrap_reps',2000))) for r in layers}
        for path,expected in sources.items():ensure(digest(path)==expected,'Measured input changed: '+path)
        ensure(vendor['frozen043'].tree_sha(model.state_dict())==initial_model,'Frozen model changed')
        d=summaries['adapter_dev']['groups']['short']['comparisons']
        signal_ok=all(d[k]['net']>0 and d[k]['N_videos']>=10 and d[k]['ci95_pp'] and d[k]['ci95_pp'][0]>0 for k in ('fine_vs_fused','fine_vs_position'))
        # Eight dev videos cannot satisfy ten-video strong evidence; retain this
        # explicit limitation rather than elevating a small diagnostic result.
        point_signal=all(d[k]['net']>0 for k in ('fine_vs_fused','fine_vs_position'))
        summary=dict(complete=True,scientific_status='DESCRIPTIVE_PROBE_SIGNAL' if point_signal else 'NO_SUPPORT_FROM_THIS_READOUT',
            strong_support=signal_ok,decision='REQUIRES_A_INTERVENTION_AND_LARGER_HELD_OUT_VERIFICATION' if point_signal else 'DO_NOT_START_ARCHITECTURE_FROM_B',
            results=summaries,scope='Annotation recoverability, fixed training-source crops; pretrained checkpoint may have seen dev; no semantic OFF or official recall claim',
            pair_censoring={r:dict(all_pairs=sum(e['role']==r for e in excluded),excluded_pairs=sum(e['role']==r and e['excluded'] for e in excluded),
                all_retained_GT=sum(e['N_GT'] for e in excluded if e['role']==r),excluded_GT=sum(e['N_GT'] for e in excluded if e['role']==r and e['excluded'])) for r in layers},
            frozen_model_unchanged=True,source_sha256=sources,elapsed_seconds=time.monotonic()-begin,
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),independent_review='PENDING')
        write_json(out/'summary.json',summary);write_json(out/'status.json',dict(complete=True,status='COMPLETE',decision=summary['decision']))
        print('COMPLETE079_B',summary['decision'],flush=True)
    except BaseException as e:
        write_json(out/'status.json',dict(complete=False,status='TECHNICAL_FAILURE',phase=phase,error=repr(e),elapsed_seconds=time.monotonic()-begin));raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.config,a.out)


if __name__=='__main__':main()
