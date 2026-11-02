"""E100 admission only. Never performs optimizer steps or old gradient probes."""
import json,sys,time,traceback,hashlib,types,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from native_runtime import NativeRuntime
import capacity_tools as cap
R=Path('/root/autodl-tmp/VrdONE-data'); HERE=R/'commander_cause_arch_100'
CACHE=R/'commander_cache_import_094/cache_v4'; OUT=HERE/'admission_fix_v3'

def req(v,s):
 if not v:raise RuntimeError(s)
def save(n,v):
 (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def captured_matches(rt,o,b):
 import models.maskvrd as mv
 original=mv.linear_sum_assignment; mats=[]
 def capture(c):
  mats.append(np.asarray(c).copy());return original(c)
 mv.linear_sum_assignment=capture
 try: matches=rt.matches(o,b)
 finally:mv.linear_sum_assignment=original
 B=len(b['preds']);req(len(mats)==4*B,'MATCHER_COST_CAPTURE')
 return matches,[[mats[l*B+i] for i in range(B)] for l in range(4)]
def joint(mats,native,route):
 B=len(mats[0]);idx=[[] for _ in range(4)]
 for b in range(B):
  total=sum(mats[l][b][route[l]] for l in range(4))
  q,g=linear_sum_assignment(total)
  for l in range(4):idx[l].append((torch.as_tensor(route[l][q],dtype=torch.int64),torch.as_tensor(g,dtype=torch.int64)))
 return [(idx[l],native[l][1]) for l in range(4)]
def equiv(b,row,t):
 if t is None:return 'BACKGROUND'
 valid=b['valid'][row,0].cpu();y=b['targets'][row][t].detach().cpu()[valid]
 return str(int(b['preds'][row][t]))+':'+hashlib.sha256(y.numpy().tobytes()).hexdigest()
def costvalue(mats,m):
 return math.fsum(float(c[int(qi),int(gi)]) for l in range(4) for c,(q,g) in zip(mats[l],m[l][0]) for qi,gi in zip(q,g))
def total_loss(rt,o,b,m):
 v=rt.losses(o,b,m);return v['Lc']+v['Lm']
def native_loss_direct(rt,o,b,matches):
 # Bypass runtime.losses; invoke original label/focal/Dice layer criteria directly.
 ce=[];mask=[];num=o['pred_masks'].new_tensor(float(max(sum(len(x) for x in b['preds']),1)))
 for layer,(indices,lm) in zip(rt.layers(o),matches):
  args=(layer['pred_logits'],layer['pred_masks'],b['preds'],b['targets'],None,indices,num,lm)
  ce.append(rt.model.loss_labels(*args)['loss_class'])
  mask.append(sum(rt.model.loss_masks(*args).values()))
 return sum(ce)+sum(mask)

def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();phase='MATERIAL';current=None
 try:
  torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
  args=types.SimpleNamespace(cache=str(CACHE),events=str(R/'commander_cause_arch_095/results_v2/events.json'),preflight=str(HERE/'preflight_v1'))
  events,c0,pool,feasible,identity=cap.material(args);save('material_identity.json',identity)
  phase='JOINT_CAPACITY_RECHECK'
  wp=R/'commander_cause_arch_099/fix_rerun_v2/capacity/witness_candidates.json'
  w=json.loads(wp.read_text());sets={(r['video_name'],r['pair_idx']):r['candidates'] for r in w['pairs']}
  req(set(sets)==set(pool),'WITNESS_PAIR_SET')
  for pair,cs in sets.items():
   req(len(cs)<=9,'WITNESS_SLOT_COUNT')
   allowed={(c['start'],c['end'],c['predicate_id']) for c in pool[pair]}
   req(all((c['start'],c['end'],c['predicate_id']) in allowed for c in cs),'WITNESS_LEGAL_CANDIDATES')
  base=cap.evaluate(c0,events);value=cap.evaluate(sets,events);checks=cap.witness_checks(value,base)
  req(base['residual334']['time_hits']==29 and base['main400']['time_hits']==46,'C0_REPRODUCTION')
  req(checks['pass'],'JOINT_CAPACITY_UNRESOLVED')
  save('capacity_receipt.json',dict(source=str(wp),sha256=cap.sha(wp),checks=checks,baseline=base,witness=value))
  phase='PREDICTOR_NATIVE_AND_ASSIGNMENT'
  rt=NativeRuntime(CACHE,HERE/'preflight_v1/decoder_inputs','/root/rebinding_project/VrdONE',R/'commander_native_error_025/export_v1/effective_config.json','/root/rebinding_project/VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth')
  routes={'T':[np.arange(9) for _ in range(4)],'X':[(np.arange(9)+k)%9 for k in [1,3,5,0]]}
  rows=[]; tests=[]; maxerr=0.; q0={}; missing=[]; c0_outputs={}
  paths=[p for role in ['train','adapter_dev','eval'] for p in sorted((CACHE/role).glob('*.pt'))]
  req(len(paths)==1725,'BATCH_COUNT')
  for i,p in enumerate(paths):
   current=p.name;b=rt.load_batch(p)
   with torch.no_grad():o=rt.forward(b);layers=rt.layers(o)
   err=0.
   for l,op in enumerate(layers):
    for field,ref in [('pred_masks',b['c']['masks'][l]),('pred_logits',b['c']['cls'][l])]:
     a=op[field].cpu();req(torch.isfinite(a).all().item(),'NONFINITE');err=max(err,float((a-ref).abs().max()))
     if field=='pred_masks':req(torch.equal(a.sigmoid()>.5,ref.sigmoid()>.5),'MASK_BITS')
     else:
      req(torch.equal(a.argmax(-1),ref.argmax(-1)),'CLASS_TOP1')
      req(torch.equal(a.softmax(-1)[...,1:].topk(6,-1).indices,ref.softmax(-1)[...,1:].topk(6,-1).indices),'CLASS_TOP6')
   req(err<=4e-6,'PREDICTOR_NATIVE '+p.name+' '+str(err));maxerr=max(maxerr,err)
   with (OUT/'predictor_rebuild.jsonl').open('a') as f:f.write(json.dumps(dict(batch=p.name,max_abs=err))+'\n')
   if p.parent.name in ['train','adapter_dev']:
    with torch.no_grad():
     nm,mats=captured_matches(rt,o,b)
     jm={k:joint(mats,nm,v) for k,v in routes.items()}
     losses={k:float(total_loss(rt,o,b,m)) for k,m in [('N',nm),*jm.items()]}
     req(all(np.isfinite(v) for v in losses.values()),'LOSS_FINITE')
     # N is the exact unmodified native matcher and Tensor criterion.
     nc=rt.matches(o,b)
     req(all(torch.equal(q,r) and torch.equal(g,s) for (a,_),(c,_) in zip(nm,nc) for (q,g),(r,s) in zip(a,c)),'N_NATIVE_MATCH')
     # FIX_RERUN: absolute 4e-6, relative delta logged but never used to widen tolerance.
     second=float(total_loss(rt,o,b,nc))
     first=losses['N']; raw_delta=abs(first-second)
     audit=dict(batch=p.name,native_loss_first=first,native_loss_second=second,
                abs_delta=raw_delta,rel_delta=raw_delta/max(abs(first),abs(second),1e-30),
                absolute_tolerance=4e-6,relative_tolerance=0.,deterministic_fallback=False)
     # Also compare wrapper against an independent direct invocation of native layer criteria.
     direct=float(native_loss_direct(rt,o,b,nc))
     audit.update(native_direct_loss=direct,wrapper_direct_abs_delta=abs(second-direct))
     passed=np.isfinite(first) and np.isfinite(second) and np.isfinite(direct) and raw_delta<=4e-6 and abs(second-direct)<=4e-6
     if not passed:
      # No forward/matching/parameter changes: deterministic loss-only recalculation.
      enabled=torch.are_deterministic_algorithms_enabled()
      try:
       torch.use_deterministic_algorithms(True)
       d1=float(total_loss(rt,o,b,nm));d2=float(total_loss(rt,o,b,nc));dn=float(native_loss_direct(rt,o,b,nc))
       delta=abs(d1-d2);drift=abs(d2-dn)
       audit.update(deterministic_fallback=True,deterministic_first=d1,deterministic_second=d2,
                    deterministic_native_direct=dn,deterministic_abs_delta=delta,
                    deterministic_rel_delta=delta/max(abs(d1),abs(d2),1e-30),deterministic_wrapper_direct_abs_delta=drift)
       passed=all(np.isfinite(v) for v in [d1,d2,dn]) and delta<=4e-6 and drift<=4e-6
      finally:torch.use_deterministic_algorithms(enabled)
     audit['pass']=bool(passed)
     with (OUT/'native_loss_consistency.jsonl').open('a') as f:f.write(json.dumps(audit,allow_nan=False)+'\n')
     req(passed,'N_NATIVE_LOSS '+json.dumps(audit))
     for k,m in jm.items():
      for l in range(4):
       for row,(q,g) in enumerate(m[l][0]):
        req(len(q)==min(9,len(b['preds'][row])) and len(q.unique())==len(q) and len(g.unique())==len(g),'ASSIGNMENT_CAPACITY')
     # Global renaming: index new=sigma(old), route conjugated; no forward rewire.
     sigma=np.array([2,7,0,8,3,1,6,4,5]);inv=np.argsort(sigma)
     pm=[[c[inv] for c in layer] for layer in mats]
     po={'pred_logits':o['pred_logits'][:,inv], 'pred_masks':o['pred_masks'][:,inv], 'aux_outputs':[{'pred_logits':x['pred_logits'][:,inv],'pred_masks':x['pred_masks'][:,inv]} for x in o['aux_outputs']]}
     for k,route in routes.items():
      conjug=[sigma[x[inv]] for x in route]; ren=joint(pm,nm,conjug)
      ca,cb=costvalue(mats,jm[k]),costvalue(pm,ren)
      la,lb=losses[k],float(total_loss(rt,po,b,ren))
      record=dict(batch=p.name,arm=k,cost_first=ca,cost_second=cb,cost_abs_delta=abs(ca-cb),
                  loss_first=la,loss_second=lb,loss_abs_delta=abs(la-lb),absolute_tolerance=4e-6,
                  cost_accumulator='math.fsum of unchanged float32 selected cost cells')
      with (OUT/'rename_consistency.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
      req(abs(ca-cb)<=4e-6,'RENAME_COST '+json.dumps(record))
      req(abs(la-lb)<=4e-6,'RENAME_LOSS '+json.dumps(record))
     tests.append(dict(batch=p.name,losses=losses,rename_pass=True,num_masks_type='Tensor'))
    if p.parent.name=='train':
     for bi,meta in enumerate(b['c']['metadata']):
      changed=False
      for l in range(4):
       am={q:g for q,g in zip(*[t.tolist() for t in nm[l][0][bi]])};bm={q:g for q,g in zip(*[t.tolist() for t in jm['T'][l][0][bi]])}
       for q in range(9):
        a,t=am.get(q),bm.get(q)
        if equiv(b,bi,a)==equiv(b,bi,t):continue
        changed=True
        kind='foreground_background' if a is None or t is None else ('same_predicate_event' if int(b['preds'][bi][a])==int(b['preds'][bi][t]) else 'different_predicate')
        rows.append(dict(batch=p.name,video=meta['video_id'],row=bi,layer=l,query=q,n_target=a,t_target=t,kind=kind))
    else:
     c0_outputs[p.name]=[{k:v.cpu() for k,v in layer.items() if k in ['pred_logits','pred_masks']} for layer in layers]
     for bi,meta in enumerate(b['c']['metadata']):
      im={g:q for q,g in zip(*[t.tolist() for t in nm[-1][0][bi]])}
      for e in meta['events']:
       if e['target_index'] in im:q0[e['event_id']]=im[e['target_index']]
       else:missing.append(e['event_id'])
   if (i+1)%300==0:print(json.dumps(dict(phase=phase,checked=i+1,max_error=maxerr)),flush=True)
  save('assignment_tests.json',tests);save('assignment_exposure.json',dict(changes=rows,changed_slots=len(rows),changed_batches=len({r['batch'] for r in rows}),total_batches=88,changed_videos=len({r['video'] for r in rows}),total_videos=32))
  save('q0.json',dict(queries=q0,unmatched=missing));torch.save(c0_outputs,OUT/'C0_dev_outputs.pt')
  if not rows:
   save('status.json',dict(status='REJECT',subreason='NO_INITIAL_ASSIGNMENT_EXPOSURE',training_runs=0));return 0
  phase='MECHANISM_SUPPORT'
  sm=json.loads((CACHE/'sources/splits/adapter_dev.json').read_text());anncache={};support=[]
  for p in sorted((CACHE/'adapter_dev').glob('*.pt')):
   current=p.name;c=torch.load(p,map_location='cpu',weights_only=False)
   for bi,meta in enumerate(c['metadata']):
    vid=meta['video_id']; ref=sm['inputs'][vid]
    if vid not in anncache:
     a=Path(ref['annotation']);a=CACHE/'sources/annotations'/(a.parent.name+'_'+a.name)
     req(cap.sha(a)==ref['annotation_sha256'],'ANNOTATION_SHA');anncache[vid]=json.loads(a.read_text())
    ann=anncache[vid];valid=c['valid'][bi,0].bool();frames=meta['sample_frame_origin']+meta['feat_stride']*torch.arange(valid.numel())
    start,end=meta['pair_frame_interval']; valid=valid&(frames>=start)&(frames<end)
    for e in meta['events']:
     pos=valid&(frames>=e['begin_fid'])&(frames<e['end_fid'])
     req(torch.equal(pos,c['targets'][bi][e['target_index']].bool()),'TARGET_COORDINATES')
     d=e['end_fid']-e['begin_fid']; neg=valid&(frames>=e['begin_fid']-d)&(frames<e['end_fid']+d)&~pos
     for gi,g in enumerate(ann['relation_instances']):
      if gi==e['annotation_gt_index'] or (g['subject_tid'],g['object_tid'])!=(meta['subject_tid'],meta['object_tid']):continue
      neg &= ~((frames>=g['begin_fid'])&(frames<g['end_fid']))
     reason='UNMATCHED' if e['event_id'] not in q0 else ('NO_POSITIVE' if not pos.any() else ('NO_NEGATIVE' if not neg.any() else None))
     support.append(dict(**e,video=vid,batch=p.name,row=bi,q0=q0.get(e['event_id']),positive_indices=torch.where(pos)[0].tolist(),negative_indices=torch.where(neg)[0].tolist(),eligible=reason is None,missing_reason=reason))
  req(len(support)==169,'DEV_N')
  groups=defaultdict(list)
  for e in support:
   if e['eligible']:groups[(e['video'],e['predicate_id'])].append(e)
  common={k:v for k,v in groups.items() if any(e['duration_seconds']<3 for e in v) and any(e['duration_seconds']>=3 for e in v)}
  ids={e['event_id'] for es in common.values() for e in es}
  for e in support:e['common_support']=e['event_id'] in ids
  ns=sum(e['duration_seconds']<3 for es in common.values() for e in es);nl=sum(e['duration_seconds']>=3 for es in common.values() for e in es);nv=len({k[0] for k in common})
  stats=dict(N_dev=169,N_eligible=sum(e['eligible'] for e in support),N_common_short=ns,N_common_long=nl,N_common_videos=nv,minimum=dict(videos=4,short=20,long=20),missing=dict(Counter(e['missing_reason'] for e in support if e['missing_reason'])),pass_gate=nv>=4 and ns>=20 and nl>=20)
  save('dev_support.json',dict(summary=stats,events=support));save('status.json',dict(status='ADMISSION_PASS' if stats['pass_gate'] else 'REJECT',subreason=None if stats['pass_gate'] else 'MECHANISM_UNIDENTIFIABLE',training_runs=0,max_native_error=maxerr,checked_batches=1725,assignment_changes=len(rows),support=stats,elapsed_seconds=time.monotonic()-start))
  print((OUT/'status.json').read_text(),flush=True);return 0
 except Exception as exc:
  save('status.json',dict(status='TECHNICAL_FAILURE',phase=phase,batch=current,error=str(exc),traceback=traceback.format_exc(),training_runs=0,elapsed_seconds=time.monotonic()-start));print((OUT/'status.json').read_text(),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
