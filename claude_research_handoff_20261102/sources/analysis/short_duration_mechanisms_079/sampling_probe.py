#!/usr/bin/env python3
"""079 frozen-weight sampling and finest-FPN-path diagnostic.

Run only on an authorized server. No checkpoint updates, dataset writes, SSH,
feature extraction or training. GT identities request complete co-presence pairs;
GT relation boundaries/predicates are used only AFTER raw prediction for diagnosis.
Reported tIoU availability is not official vIoU, one-to-one recall or mAP.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import types
import traceback
import numpy as np
from sampling_math import (sampled_frames, visual_interpolation, envelope_intervals,
                           interval_iou, duration_bin, paired_bootstrap, forward_hold_binary_iou)

HERE = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()


def read_config(path):
    with Path(path).open() as f:
        if Path(path).suffix.lower() == '.json': return json.load(f)
        import yaml
        return yaml.safe_load(f)


def resolve(base, value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else base/p


def remap_path(path, mapping):
    text=str(path)
    for source,target in sorted(mapping.items(),key=lambda item:len(item[0]),reverse=True):
        prefix=str(source).rstrip('/')
        if text==prefix or text.startswith(prefix+'/'):
            return Path(str(target).rstrip('/')+text[len(prefix):])
    return Path(text)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def model_state_hash(model):
    import torch
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.shape).encode());h.update(str(value.dtype).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def coarse_gate(neck, gate):
    """Runtime-only intervention: scale coarse addend at idx==0; restore always."""
    if gate is None:
        yield
        return
    if neck.__class__.__name__ != 'FPN1D_Fuse':
        raise RuntimeError('Expected the pinned FPN1D_Fuse class')
    original = neck.forward
    source = textwrap.dedent(inspect.getsource(type(neck).forward))
    needle = 'y = cur_fpn + F.interpolate(y, scale_factor=self.scale_factor, mode="nearest")'
    if source.count(needle) != 1:
        raise RuntimeError('FPN coarse-addition source contract changed')
    # Gate 1 uses original expression (not x*1), allowing exact no-op verification.
    replacement = ('coarse = F.interpolate(y, scale_factor=self.scale_factor, mode="nearest")\n'
                   '            if idx == 0 and self._probe079_coarse_gate != 1.0:\n'
                   '                coarse = coarse * self._probe079_coarse_gate\n'
                   '            y = cur_fpn + coarse')
    source = source.replace(needle, replacement)
    namespace = dict(vars(sys.modules[type(neck).__module__]))
    exec(compile(source, '<079-finest-fpn-coarse-gate>', 'exec'), namespace)
    neck._probe079_coarse_gate = float(gate)
    neck.forward = types.MethodType(namespace['forward'], neck)
    try: yield
    finally:
        neck.forward = original
        delattr(neck, '_probe079_coarse_gate')


def capture_before_postprocess(model):
    source = textwrap.dedent(inspect.getsource(type(model).forward_test))
    needle = '    ## remove pairs without interaction\n'
    if source.count(needle) != 1: raise RuntimeError('forward capture source contract changed')
    source = source.replace(needle, '    return predictions\n'+needle)
    namespace = dict(vars(sys.modules[type(model).__module__]))
    exec(compile(source, '<079-raw-before-postprocess>', 'exec'), namespace)
    model.forward_test = types.MethodType(namespace['forward_test'], model)


def one_forward(model, feature, sid, gate=None):
    import torch
    # Native preprocessing preserves >512 sequences and pads to max_div_factor.
    with torch.no_grad(), coarse_gate(model.neck, gate):
        raw = model({'sids': torch.tensor([sid]), 'so_features_list': [feature]})
    n = int(raw['output_mask'][0].sum())
    if n != feature.shape[-1]: raise RuntimeError('native preprocessing truncated sequence')
    mask_tensor = raw['pred_masks'][0][:,:n].detach()
    # Keep the original Torch sigmoid kernel/dtype, including near-zero rounding.
    # A logit > 0 comparison is mathematically equivalent but not bitwise native.
    binary = (mask_tensor.sigmoid() > 0.5).cpu().numpy()
    masks = mask_tensor.cpu().numpy()
    classes = raw['pred_logits'][0].detach().cpu().numpy()
    if not np.isfinite(masks).all() or not np.isfinite(classes).all():
        raise RuntimeError('nonfinite model output')
    return masks, classes, binary


def direct_forward(model, feature):
    import torch
    (xx, mm, ids) = model.preprocessing([feature])
    nonempty = [i for i in range(2) if xx[i] is not None]
    if len(nonempty) != 1 or ids[nonempty[0]] != [0]: raise RuntimeError('unexpected preprocessing split')
    k=nonempty[0]
    with torch.no_grad(): r=model._mask_vrd(xx[k], mm[k])
    n=int(r['output_mask'][0].sum())
    mask_tensor=r['pred_masks'][0,:,:n]
    return (mask_tensor.cpu().numpy(),r['pred_logits'][0].cpu().numpy(),
            (mask_tensor.sigmoid() > 0.5).cpu().numpy())


def input_pairs(inp, tids, stride):
    pairs={}
    for idx,(si,oi) in enumerate(zip(inp.get('sids',[]),inp.get('oids',[]))):
        si,oi=int(si),int(oi)
        ds,do=inp['traj_durations'][si],inp['traj_durations'][oi]
        start,end=int(max(ds[0],do[0])),int(min(ds[1],do[1]))
        offset=int(inp['so_offset'][idx])
        key=(int(tids[si]),int(tids[oi]),start,end)
        if key in pairs: raise RuntimeError('duplicate identity + co-presence segment key')
        feature=inp['so_features_list'][idx]
        frames=sampled_frames(start,end,stride,offset)
        if feature.shape != (2069,len(frames)): raise RuntimeError('2069-channel input / coordinates mismatch')
        pairs[key]={'feature':feature,'frames':frames,'sid':si,'oid':oi,'offset':offset,'stride':stride}
    return pairs


def annotation_path(root, full):
    group,vid=full.split('_',1)
    p=root/group/(vid+'.json')
    if not p.is_file(): raise FileNotFoundError(p)
    return p


def make_rows(video, ann, pair_cache, tids):
    fps=float(ann['fps'])
    if not fps>0: raise RuntimeError('invalid annotation FPS')
    durations=pair_cache['traj_durations']
    identity_spans={}
    for si,oi in zip(pair_cache['sids'],pair_cache['oids']):
        si,oi=int(si),int(oi)
        a,b=int(max(durations[si,0],durations[oi,0])),int(min(durations[si,1],durations[oi,1]))
        identity_spans.setdefault((tids[si],tids[oi]),[]).append((a,b))
    rows=[]
    for gi,g in enumerate(ann['relation_instances']):
        sid,oid=int(g['subject_tid']),int(g['object_tid'])
        a,b=int(g['begin_fid']),int(g['end_fid'])
        dur=(b-a)/fps
        overlapping=[(x,y) for x,y in identity_spans.get((sid,oid),[]) if min(y,b)>max(x,a)]
        span=max((y-x for x,y in overlapping),default=0)/fps
        contains=any(x<=a and y>=b for x,y in overlapping)
        rows.append({'video':video,'gt_index':gi,'subject_tid':sid,'object_tid':oid,'predicate':g['predicate'],
                     'start':a,'end':b,'fps':fps,'duration_seconds':dur,'duration_bin':duration_bin(dur),
                     'copresence_seconds':span,'full_gt_in_single_copresence':contains,
                     # Diagnostic cohort rule fixed before prediction, not inference routing.
                     'long_copresence_short':dur<3 and span>=10 and contains})
    return rows


def blank_arm():
    return dict(any_query_iou=0.,semantic_top6_iou=0.,top1_class_iou=0.,
                any_query_forward_hold_binary_iou=0.,semantic_top6_forward_hold_binary_iou=0.,
                target_class_highest_query_iou=0.,target_class_highest_query_probability=-1.,
                raw_event_separation_max=None,pair_segments_present=0,candidate_intervals=0)


def update_gt(rows, arm, key, masks, classes, binary, frames, pred_ids):
    spans=envelope_intervals(binary,frames,min_frames=5)
    probabilities=np.exp(classes-classes.max(-1,keepdims=True));probabilities/=probabilities.sum(-1,keepdims=True)
    top1=probabilities.argmax(-1)
    top6=np.argsort(-probabilities[:,1:],axis=-1)[:,:min(6,probabilities.shape[1]-1)]+1
    for row in rows:
        if (row['subject_tid'],row['object_tid'])!=key[:2]: continue
        r=row[arm];r['pair_segments_present']+=1
        r['candidate_intervals']+=sum(s is not None for s in spans)
        target=(row['start'],row['end']);ious=np.asarray([interval_iou(s,target) for s in spans])
        pid=pred_ids[row['predicate']]
        r['any_query_iou']=max(r['any_query_iou'],float(ious.max(initial=0)))
        eligible=(top6==pid).any(-1)
        binary_ious=forward_hold_binary_iou(binary,frames,key[3],target)
        r['any_query_forward_hold_binary_iou']=max(r['any_query_forward_hold_binary_iou'],float(binary_ious.max(initial=0)))
        r['semantic_top6_forward_hold_binary_iou']=max(r['semantic_top6_forward_hold_binary_iou'],float(binary_ious[eligible].max(initial=0)))
        r['semantic_top6_iou']=max(r['semantic_top6_iou'],float(ious[eligible].max(initial=0)))
        r['top1_class_iou']=max(r['top1_class_iou'],float(ious[top1==pid].max(initial=0)))
        q=int(probabilities[:,pid].argmax())
        if float(probabilities[q,pid])>r['target_class_highest_query_probability']:
            r['target_class_highest_query_probability']=float(probabilities[q,pid])
            r['target_class_highest_query_iou']=float(ious[q])
        # Descriptive raw-logit separation only. Outside GT is not certified semantic absence.
        inside=(frames>=row['start'])&(frames<row['end'])
        outside=~inside
        if inside.any() and outside.any():
            sep=float((masks[:,inside].mean(-1)-masks[:,outside].mean(-1)).max())
            prior=r['raw_event_separation_max']
            r['raw_event_separation_max']=sep if prior is None else max(prior,sep)


def finish_rows(rows,arms):
    for r in rows:
        for arm in arms:
            for metric in ('any_query','semantic_top6','top1_class','target_class_highest_query',
                           'any_query_forward_hold_binary','semantic_top6_forward_hold_binary'):
                r[arm][metric+'_hit']=r[arm][metric+'_iou']>=.5


def summarize(rows,arms,opt,selected,excluded,complete,elapsed):
    groups={'all':rows,'short_lt3':[r for r in rows if r['duration_seconds']<3],
            'long_ge3':[r for r in rows if r['duration_seconds']>=3],
            'long_copresence_short':[r for r in rows if r['long_copresence_short']]}
    for name in ('sub1','1to3','3to10','10plus'):groups[name]=[r for r in rows if r['duration_bin']==name]
    for name,rr in list(groups.items()):
        groups[name+'_common_visual_support']=[r for r in rr if r.get('common_visual_pair_segments_overlapping_gt',0)>0]
    comparisons=[('baseline',a) for a in arms if a!='baseline' and not a.endswith('_common')]
    if 'dense2' in arms and 'visual_interp2' in arms:comparisons.append(('visual_interp2','dense2'))
    if 'dense2_common' in arms and 'visual_interp2_common' in arms:
        comparisons.append(('visual_interp2_common','dense2_common'))
    results={}
    metrics=['any_query_hit','semantic_top6_hit','top1_class_hit','target_class_highest_query_hit',
             'any_query_forward_hold_binary_hit','semantic_top6_forward_hold_binary_hit']
    for name,rr in groups.items():
        results[name]={'N_GT':len(rr),'N_videos':len({r['video'] for r in rr}),'arms':{},'comparisons':{}}
        for arm in arms:
            results[name]['arms'][arm]={m:{'hit':sum(int(r[arm][m]) for r in rr),'GT':len(rr)} for m in metrics}
        for left,right in comparisons:
            entry={}
            for m in metrics:
                added=sum(not r[left][m] and r[right][m] for r in rr)
                lost=sum(r[left][m] and not r[right][m] for r in rr)
                video_nets={v:sum(int(r[right][m])-int(r[left][m]) for r in rr if r['video']==v) for v in {r['video'] for r in rr}}
                entry[m]={'added':added,'lost':lost,'net':added-lost,
                          'positive_net_videos':sum(v>0 for v in video_nets.values()),
                          'negative_net_videos':sum(v<0 for v in video_nets.values()),
                          **paired_bootstrap(rr,left,right,m,int(opt.get('bootstrap_reps',2000)),int(opt.get('seed',79)))}
            results[name]['comparisons'][right+'__minus__'+left]=entry
    return {'complete':complete,'status':'COMPLETE_DIAGNOSTIC' if complete else 'INCOMPLETE_RESOURCE_OR_ERROR',
            'training':False,'official_benchmark_metrics':False,'GT_input':True,
            'interpretation':'Per-GT temporal candidate availability on previously exposed GT-input diagnostic videos. Not one-to-one R@100 or mAP; oracle identities and GT predicate used in diagnostic selection only.',
            'primary_contrast':'dense2_common minus visual_interp2_common, any_query_hit, short_lt3_common_visual_support',
            'common_visual_support':'Both outputs aggregated ONLY over identical entity/co-presence segments with baseline interpolation source. Conditional GT subset requires at least one such segment overlapping GT. Full-denominator and unrestricted comparisons are also disclosed.',
            'unrestricted_dense_visual_contrast':'Includes missing-baseline-source availability differences and is NOT a pure added-visual-observation effect.',
            'binary_metric':'forward-hold occupancy IoU in original-frame coordinates; hold each positive token until next sample, last token to pair end, no min-5 filter. Distinct from native first/last hull tIoU and historical GT27 binary metrics.',
            'structural_contrasts':'coarse_gate_half/zero minus baseline; causal intervention on finest FPN coarse addend, not proof of long-predicate semantic suppression',
            'long_copresence_short_definition':'GT relation <3s, continuous GT pair co-presence >=10s and contains complete GT relation',
            'selected_videos':selected,'processed_videos':sorted({r['video'] for r in rows}),
            'excluded_or_pending_videos':excluded,'elapsed_seconds':elapsed,'results':results}


def advance_decisions(summary, criteria):
    """Prespecified diagnostic gates only; never launch a follow-up automatically."""
    minimum=float(criteria.get('sampling_min_net_short_pp',1.))
    lower_gt=float(criteria.get('sampling_video_ci_lower_gt',0.))
    def check(group,comparison):
        item=summary['results'].get(group,{}).get('comparisons',{}).get(comparison,{}).get('any_query_hit',{})
        ci=item.get('ci95_pp')
        reasons=[]
        if not summary['complete']:reasons.append('incomplete_or_technical_failure')
        if item.get('videos',0)<10:reasons.append('fewer_than_10_cohort_support_videos')
        if item.get('net_pp',float('-inf'))<minimum:reasons.append('short_net_below_prespecified_pp')
        if ci is None or ci[0]<=lower_gt:reasons.append('video_bootstrap_lower_bound_not_positive')
        return {'passed':not reasons,'reasons':reasons,'measurement':item}
    dense=check('short_lt3_common_visual_support','dense2_common__minus__visual_interp2_common')
    baseline=check('short_lt3','dense2__minus__baseline')
    gates={name:check('short_lt3',name+'__minus__baseline') for name in ('coarse_gate_half','coarse_gate_zero')}
    return {'automatically_expand':False,
            'thresholds':{'minimum_short_net_pp':minimum,'video_ci_lower_strictly_above':lower_gt,
                          'minimum_cohort_support_videos':10,'support_means':'videos containing evaluated GT in the contrast cohort, not only videos with gains'},
            'added_visual_observation':{'status':'STRONG_DIAGNOSTIC_SUPPORT' if dense['passed'] and baseline['passed'] else 'INCONCLUSIVE_NO_EXPANSION',
                                        'common_pair_visual_comparison':dense,'native_baseline_comparison':baseline},
            'coarse_fusion_path':{'status':'STRONG_DIAGNOSTIC_SUPPORT' if all(x['passed'] for x in gates.values()) else 'INCONCLUSIVE_NO_EXPANSION',
                                  'both_gate_results':gates},
            'next_action':'Do not automatically run stride1, search gates, train a new architecture or expand the benchmark. Negative and mixed results remain in the output.'}


def predicate_duration_summary(rows, arms):
    """Descriptive all-predicate table; no selective reporting or extra tests."""
    results=[]
    for predicate in sorted({r['predicate'] for r in rows}):
        for name in ('short_lt3','long_ge3','sub1','1to3','3to10','10plus'):
            rr=[r for r in rows if r['predicate']==predicate and
                ((name=='short_lt3' and r['duration_seconds']<3) or
                 (name=='long_ge3' and r['duration_seconds']>=3) or r['duration_bin']==name)]
            if not rr:continue
            counts={}
            for arm in arms:
                if arm.endswith('_common'):continue
                counts[arm]={}
                for metric in ('any_query_hit','semantic_top6_hit','any_query_forward_hold_binary_hit'):
                    added=sum(not r['baseline'][metric] and r[arm][metric] for r in rr)
                    lost=sum(r['baseline'][metric] and not r[arm][metric] for r in rr)
                    counts[arm][metric]={'hit':sum(r[arm][metric] for r in rr),'GT':len(rr),
                                        'added_vs_baseline':added,'lost_vs_baseline':lost,'net_vs_baseline':added-lost}
            results.append({'predicate':predicate,'duration_group':name,'GT':len(rr),
                            'videos':len({r['video'] for r in rr}),'arms':counts})
    return {'scope':'All observed predicates; descriptive only. Same-predicate short/long counts, not adjusted causal duration effects or an independent test. Exact common-pair comparisons remain available in per_gt.csv.',
            'results':results}


def write_rows(path,rows,arms):
    flat=[]
    for r in rows:
        record={k:v for k,v in r.items() if k not in arms}
        for a in arms:
            for k,v in r[a].items():record[a+'__'+k]=v
        flat.append(record)
    if not flat: path.write_text('video,gt_index,status\n');return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)


def run(config_path,out):
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
    import torch
    cfg=read_config(config_path);base=Path(config_path).resolve().parent
    opt=cfg.get('sampling',{});paths={k:remap_path(resolve(base,v),cfg.get('path_remap',{})) for k,v in cfg['paths'].items()}
    contract_path=resolve(base,cfg.get('source_contract','vendor078/source_contract043.json'))
    contract=read_config(contract_path)
    repo=paths['repo_root'];sys.path.insert(0,str(repo))
    hashes={str(contract_path):sha(contract_path),str(Path(config_path)):sha(config_path),
            str(Path(__file__)):sha(__file__),str(HERE/'sampling_math.py'):sha(HERE/'sampling_math.py'),
            str(HERE/'vendor_gt'/'run_probe.py'):sha(HERE/'vendor_gt'/'run_probe.py')}
    # Pinned source fingerprints protect channel order, preprocessing and FPN intervention.
    for rel,expected in contract['repo_sources'].items():
        p=repo/rel;observed=sha(p);hashes[str(p)]=observed
        if observed!=expected:raise RuntimeError('Source SHA256 mismatch: '+str(p))
    cp=paths['checkpoint'];hashes[str(cp)]=sha(cp)
    if hashes[str(cp)]!=contract['checkpoint_sha256']:raise RuntimeError('Checkpoint SHA256 mismatch')
    from dataloaders.vidor import VidOR
    from dataloaders.category import vidor_pred_name_to_id
    from models.maskvrd import MaskVRD
    import utils.misc as utils
    ef_path=resolve(base,cfg.get('model_config',cfg.get('files',{}).get('effective_config','effective_config.json')))
    effective=read_config(ef_path);hashes[str(ef_path)]=sha(ef_path)
    model_config=dict(effective['model_config']);model_config['with_clip_feature']=False
    if (model_config.get('visual_dim'),model_config.get('bbox_entity_dim'),model_config.get('bbox_so_dim')) != (1024,8,5):
        raise RuntimeError('Channel prefix requires 2 x 1024 ROI then 5+8+8 geometry')
    inference=dict(effective['inference_config'])
    ds_config=dict(effective['dataset_config'])
    if (inference.get('topk'),inference.get('feat_stride'),inference.get('pred_min_frames'))!=(6,4,5):
        raise RuntimeError('Frozen baseline requires top6 predicates, stride4, minimum5 physical frames')
    if (ds_config.get('feat_stride'),ds_config.get('stride_offset'),ds_config.get('proposal_min_frames'),ds_config.get('random_stride'))!=(4,0,5,False):
        raise RuntimeError('Frozen baseline loader contract changed')
    if ds_config.get('with_clip_feature',False):raise RuntimeError('Probe requires non-CLIP 2069 channel input')
    torch.set_num_threads(2);utils.set_seed(int(opt.get('seed',79)))
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no silent CPU scientific run')
    device=cfg.get('device','cuda:0')
    model=MaskVRD(model_config,device=device).to(device).eval();model._config_eval(inference);model.max_so_pair=1
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in ck[contract.get('checkpoint_key','model_state_dict_ema')].items()},strict=True)
    del ck
    for p in model.parameters():p.requires_grad_(False)
    initial_state_hash=model_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    capture_before_postprocess(model)
    vendor=import_file(HERE/'vendor_gt'/'run_probe.py','probe079_vendor_gt')
    vendor.ROOT=repo;vendor.DATA=paths['data_root'];vendor.GT_PROP=paths['gt_proposals'];vendor.GT_FEAT=paths['gt_features']
    available=sorted(p.stem for p in paths['gt_proposals'].glob('*.pkl'))
    if not opt.get('video_ids') and len(available)!=27:
        raise RuntimeError('Expected exactly the existing strict27 GT proposal files; refuse silent cohort replacement or missing videos')
    selected=opt.get('video_ids') or available
    if len(selected)!=len(set(selected)):raise RuntimeError('Duplicate diagnostic video IDs')
    if opt.get('max_videos') is not None:selected=selected[:int(opt['max_videos'])]
    if not selected:raise RuntimeError('No GT proposal videos available; no feature regeneration is attempted')
    selected=list(selected)
    arms={'baseline':(4,0,None),'phase1':(4,1,None),'phase2':(4,2,None),'phase3':(4,3,None),
          'dense2':(2,0,None),'visual_interp2':(2,0,None)}
    if opt.get('full_interp2',False):arms['full_interp2']=(2,0,None)
    if opt.get('stride1',False):arms['dense1']=(1,0,None)
    gates=opt.get('coarse_gates',[.5,0.])
    for gate in gates:
        name='coarse_gate_half' if gate==.5 else 'coarse_gate_zero' if gate==0 else 'coarse_gate_'+str(gate).replace('.','p')
        arms[name]=(4,0,float(gate))
    evaluation_arms=list(arms)+['dense2_common','visual_interp2_common']
    dump(out/'effective_config.json',{'requested_config':cfg,'resolved_paths':{k:str(v) for k,v in paths.items()},
         'model_config':model_config,'inference_config':inference,'dataset_config':ds_config,'arms':arms,
         'physical_min_prediction_frames':5,'mask_threshold':'native Torch mask_tensor.sigmoid() > 0.5; preserve dtype and kernel',
         'pair_microbatch':1,'padding':'native preprocessing; no 512 truncation',
         'interpolation':'linear on absolute sampled frames; clamp nearest value beyond observed edges',
         'visual_interp2':'only first 2048 ROI channels replaced; final 21 geometry channels exact dense2',
         'pair_matching_key':['subject_GT_tid','object_GT_tid','copresence_start','copresence_end']})
    dump(out/'selection.json',selected);dump(out/'source_hashes.json',hashes)
    rows=[];excluded=[];video_audit=[];start=time.monotonic();first_verified=False;failed=None
    def budget():
        if time.monotonic()-start>float(opt.get('wall_seconds',2400)):raise TimeoutError('079 sampling wall budget reached')
    try:
        for vi,full in enumerate(selected):
            budget()
            ap=annotation_path(paths['annotations'],full);ann=read_config(ap)
            for p in (ap,paths['gt_proposals']/(full+'.pkl'),paths['gt_features']/(full+'.pkl')):hashes[str(p)]=sha(p)
            raw,tids,st=vendor.gt_cache(full,ann,audit=True)
            vr=make_rows(full,ann,raw,tids)
            for row in vr:
                for arm in evaluation_arms:row[arm]=blank_arm()
            # Reject whole videos BEFORE any arm, never silently truncate a long pair.
            frame_lengths=[int(min(raw['traj_durations'][s,1],raw['traj_durations'][o,1])-max(raw['traj_durations'][s,0],raw['traj_durations'][o,0])) for s,o in zip(raw['sids'],raw['oids'])]
            shortest_stride=min(v[0] for v in arms.values())
            max_frames=max(frame_lengths,default=0);max_tokens=(max_frames+shortest_stride-1)//shortest_stride
            if max_frames>int(opt.get('max_pair_frames',32768)) or max_tokens>int(opt.get('max_tokens',8192)):
                excluded.append({'video':full,'N_GT':len(vr),'reason':'resource_guard_whole_video_all_arms','max_pair_frames':max_frames,'max_tokens':max_tokens})
                print(json.dumps(excluded[-1]),flush=True);continue
            inputs={}
            for stride,offset in sorted({(v[0],v[1]) for v in arms.values()}):
                ds=VidOR.__new__(VidOR)
                for k,v in dict(with_clip_feature=False,feat_stride=stride,proposal_min_frames=5,random_stride=False,stride_offset=offset).items():setattr(ds,k,v)
                inp=ds._val_getitem(raw,viou_threshold=1.01)
                inputs[(stride,offset)]=input_pairs(inp,tids,stride)
            common_visual_keys=set(inputs[(4,0)]) & set(inputs[(2,0)])
            for row in vr:
                row['common_visual_pair_segments_overlapping_gt']=sum(
                    key[:2]==(row['subject_tid'],row['object_tid']) and min(key[3],row['end'])>max(key[2],row['start'])
                    for key in common_visual_keys)
            audit={'video':full,'GT':len(vr),'cache_stats':st,'arms':{},'max_pair_frames':max_frames,'max_tokens':max_tokens}
            audit['common_visual_pair_keys']=sorted(common_visual_keys)
            for arm,(stride,offset,gate) in arms.items():
                pairs=inputs[(stride,offset)];saved=[]
                for pi,(key,entry) in enumerate(sorted(pairs.items())):
                    budget();x=entry['feature']
                    if arm in ('visual_interp2','full_interp2'):
                        source=inputs[(4,0)].get(key)
                        if source is None:
                            # A >=5-frame pair with >=2 dense samples but no baseline support
                            # has no legitimate interpolation input. Retain as absent, not delete GT.
                            saved.append({'key':key,'status':'absent_baseline_interpolation_source'});continue
                        z=visual_interpolation(source['feature'].numpy(),x.numpy(),source['frames'],entry['frames'],full=arm=='full_interp2')
                        x=torch.from_numpy(z)
                    masks,classes,binary=one_forward(model,x,entry['sid'],gate)
                    if not first_verified and arm=='baseline':
                        repeat=one_forward(model,x,entry['sid'])
                        direct=direct_forward(model,x)
                        noop=one_forward(model,x,entry['sid'],1.)
                        for label,values in [('repeat',repeat),('native_direct_capture',direct),('coarse_gate_one',noop)]:
                            if not (np.array_equal(masks,values[0]) and np.array_equal(classes,values[1]) and np.array_equal(binary,values[2])):
                                raise RuntimeError('Exact baseline verification failed: '+label)
                        dump(out/'baseline_exact_verification.json',{'video':full,'pair_key':key,'sha256':arrays_hash(masks,classes,binary),
                             'repeat_exact':True,'native_direct_capture_exact':True,'gate_one_exact':True,
                             'native_binary_sigmoid_exact':True,'binary_threshold':'Torch sigmoid > 0.5, original dtype',
                             'mask_shape':list(masks.shape),'class_shape':list(classes.shape),'tolerance':0.})
                        first_verified=True
                    update_gt(vr,arm,key,masks,classes,binary,entry['frames'],vidor_pred_name_to_id)
                    if arm in ('dense2','visual_interp2') and key in common_visual_keys:
                        update_gt(vr,arm+'_common',key,masks,classes,binary,entry['frames'],vidor_pred_name_to_id)
                    meta={'key':key,'sid_row':entry['sid'],'oid_row':entry['oid'],'stride':stride,'offset':offset,
                          'tokens':x.shape[-1],'common_visual_pair_support':key in common_visual_keys,
                          'output_sha256':arrays_hash(masks,classes,binary),'input_feature_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
                    if opt.get('save_raw',True):
                        dest=out/'raw'/full/arm;dest.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(dest/(str(pi)+'.npz'),mask_logits=masks,native_binary_mask=binary,predicate_logits=classes,frames=entry['frames'])
                        meta['raw_file']=str((dest/(str(pi)+'.npz')).relative_to(out))
                    saved.append(meta)
                audit['arms'][arm]={'pair_segments':len(pairs),'records':saved}
            finish_rows(vr,evaluation_arms);rows.extend(vr);video_audit.append(audit)
            dump(out/'video_input_audit.json',video_audit);dump(out/'source_hashes.json',hashes)
            write_rows(out/'per_gt.csv',rows,evaluation_arms)
            print(json.dumps({'video':full,'completed':vi+1,'selected':len(selected),'GT_evaluated':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
            del inputs,raw,vr
            # No forced empty_cache per pair: retain allocator reuse and avoid GPU idle gaps.
    except Exception as exc:
        failed={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(out/'error.json',failed)
    # Re-read every consumed file and all model buffers/parameters, even after a
    # resource interruption. A changed input or weight is a technical failure.
    final_audit={'files':{},'model_state_before':initial_state_hash,'all_unchanged':True}
    try:
        for path,before in hashes.items():
            after=sha(path)
            final_audit['files'][path]={'before':before,'after':after,'unchanged':after==before}
            final_audit['all_unchanged'] &= after==before
        final_audit['model_state_after']=model_state_hash(model)
        final_audit['model_state_unchanged']=final_audit['model_state_after']==initial_state_hash
        final_audit['all_unchanged'] &= final_audit['model_state_unchanged']
        if not final_audit['all_unchanged']:raise RuntimeError('Source, input or model state changed during frozen diagnostic')
    except Exception as exc:
        final_audit['all_unchanged']=False
        final_audit['audit_error']=str(exc)
        if failed is None:
            failed={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            dump(out/'error.json',failed)
    dump(out/'final_immutability_audit.json',final_audit)
    done={r['video'] for r in rows};known={e['video'] for e in excluded}
    for full in selected:
        if full not in done and full not in known:
            ap=annotation_path(paths['annotations'],full)
            excluded.append({'video':full,'N_GT':len(read_config(ap)['relation_instances']),'reason':'not_completed_all_arms'})
    complete=not excluded and failed is None and first_verified
    dump(out/'video_input_audit.json',video_audit);dump(out/'source_hashes.json',hashes)
    write_rows(out/'per_gt.csv',rows,evaluation_arms)
    summary=summarize(rows,evaluation_arms,opt,selected,excluded,complete,time.monotonic()-start)
    summary['N_GT_selected']=len(rows)+sum(e['N_GT'] for e in excluded)
    summary['N_GT_evaluated_all_arms']=len(rows)
    summary['N_GT_not_evaluated_all_arms']=sum(e['N_GT'] for e in excluded)
    summary['baseline_exact_verified']=first_verified
    summary['source_and_model_unchanged']=final_audit['all_unchanged']
    summary['peak_cuda_allocated_MiB']=torch.cuda.max_memory_allocated(device)/(1024**2)
    summary['peak_cuda_reserved_MiB']=torch.cuda.max_memory_reserved(device)/(1024**2)
    summary['advance_decisions']=advance_decisions(summary,cfg.get('advance_criteria',{}))
    dump(out/'advance_decisions.json',summary['advance_decisions'])
    dump(out/'predicate_duration_summary.json',predicate_duration_summary(rows,evaluation_arms))
    dump(out/'summary.json',summary);dump(out/'completion.json',{'complete':complete,'status':summary['status'],'elapsed_seconds':summary['elapsed_seconds']})
    print(json.dumps({'status':summary['status'],'GT_evaluated':len(rows),'GT_not_evaluated':summary['N_GT_not_evaluated_all_arms']}),flush=True)
    if failed:raise RuntimeError(failed['type']+': '+failed['message'])
    return 0 if complete else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--out')
    parser.add_argument('--self-test',action='store_true',help='Run NumPy synthetic tests only; no dataset, GPU or model')
    args=parser.parse_args()
    if args.self_test:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_sampling_math.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.out:parser.error('--out is required unless --self-test')
    out=Path(args.out).resolve()
    if out.exists():raise RuntimeError('Refusing to overwrite any existing output directory')
    out.mkdir(parents=True)
    try:return run(Path(args.config).resolve(),out)
    except Exception as exc:
        if not (out/'error.json').exists():dump(out/'error.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        if not (out/'completion.json').exists():dump(out/'completion.json',{'complete':False,'status':'FAILED_PREFLIGHT_OR_EXECUTION'})
        raise


if __name__=='__main__':raise SystemExit(main())
