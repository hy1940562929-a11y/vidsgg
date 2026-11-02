"""094 cache assembly + frozen-branch verification ONLY. No optimizer/encoder/evaluation."""
import ast, csv, hashlib, importlib.util, json, os, shutil, time, types
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT=Path('/root/autodl-tmp/VrdONE-data')
SRC=ROOT/'commander_mech_minifalsify_087/results_v1'
CODE=ROOT/'commander_mech_minifalsify_087/code_v1'
REPO=Path('/root/rebinding_project/VrdONE')
OLD=ROOT/'commander_architecture_010/export_v1'
OUT=ROOT/'commander_cache_import_094/cache_v1'
TOL=4e-6
HEAD_SHA='4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627'
SPLIT_SHA={'train':'39c0555bda3d99e928aef89a9f346a9b2c418a4e051205729a3838927a2306c5','adapter_dev':'4a9dc55807723c69b2f8d10d73ad827ca3b9c9f2007f2e6607603f8a16db7edc'}
files=[]; sources={}; records=[]; checks=[]; matching=[]; annotation_cache={}; phase='START'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def verify(p,expected=None):
 actual=sha(p);require(expected is None or actual==expected,'SOURCE_SHA '+str(p));sources[str(p)]=actual;return actual
def add(p,origin,field_map=None):
 files.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size,source=origin,field_map=field_map))
def dump(rel,obj):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');return p
def copy(p,rel,expected=None):
 h=verify(p,expected);dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);add(dest,str(p));return h
def load_module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def annotation(p,expected):
 if str(p) not in annotation_cache:
  copy(p,'sources/annotations/'+p.parent.name+'_'+p.name,expected)
  annotation_cache[str(p)]=json.loads(p.read_text())
 return annotation_cache[str(p)]

def native_check(c,raw):
 B=c['f'].shape[0];hs=c['hs'].cuda();valid=c['valid'].cuda()
 with torch.no_grad():
  z=hs
  for i in range(3):
   z=F.conv1d(z,head[f'layers.{i}.weight'],head[f'layers.{i}.bias'])
   if i<2:z=F.gelu(z)
  pred=torch.einsum('lbqc,bcm->lbqm',z.view(4,B,256,9).permute(0,1,3,2),c['f'].cuda()).masked_fill(~valid[None,...],-10.).cpu()
  cls=F.conv1d(hs,classhead['weight'],classhead['bias']).view(4,B,51,9).permute(0,1,3,2).cpu()
 errors={}
 for name,a,b in [('projection_mask',pred,c['masks']),('projection_class',cls,c['cls']),('hook_native_mask',c['masks'][-1],raw['native']['/pred_masks']),('hook_native_class',c['cls'][-1],raw['native']['/pred_logits'])]:
  require(a.shape==b.shape and a.dtype==b.dtype,'SHAPE_DTYPE '+name)
  require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(),'NONFINITE '+name)
  error=float((a-b).abs().max());errors[name]=error;require(error<=TOL,'NUMERICAL '+name+' '+str(error))
 require(torch.equal(pred.sigmoid()>.5,c['masks'].sigmoid()>.5),'MASK_BITS')
 require(torch.equal(cls.argmax(-1),c['cls'].argmax(-1)),'CLASS_TOP1')
 require(torch.equal(cls.softmax(-1)[...,1:].topk(6,-1).indices,c['cls'].softmax(-1)[...,1:].topk(6,-1).indices),'CLASS_TOP6')
 return dict(max_abs=errors,projection_exact=torch.equal(pred,c['masks']),discrete_exact=True)

start=time.time();OUT.mkdir(parents=True,exist_ok=False)
try:
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
 cfg=json.loads((CODE/'config.json').read_text());contract=json.loads((ROOT/'commander_mechanisms_079/code_v3/vendor078/source_contract043.json').read_text())
 artifact_hashes=json.loads((SRC/'artifact_sha256.json').read_text());old_sources=json.loads((SRC/'source_sha256.json').read_text())
 for rel in ['forward_audit.json','split_audit.json','freeze_audit.json','timing_receipt.json','source_sha256.json']:
  copy(SRC/rel,'sources/087/'+rel,artifact_hashes.get(rel))
 for rel in ['eval_locked.csv','members.json','eval_coordinates.json','config.json']:
  copy(CODE/rel,'sources/087/'+rel,old_sources.get(str(CODE/rel)))
 for rel in ['models/losses.py','models/maskvrd.py','models/blocks.py','models/predictor.py','models/transformer.py','dataloaders/category.py','dataloaders/vidor.py']:
  copy(REPO/rel,'sources/repo/'+rel,contract['repo_sources'][rel])
 cp=Path(cfg['paths']['checkpoint']);verify(cp,HEAD_SHA)
 weights=torch.load(cp,map_location='cpu',weights_only=False)[contract['checkpoint_key']];weights={k.removeprefix('module.'):v for k,v in weights.items()}
 init={k.removeprefix('predictor.mask_embed.'):v.clone() for k,v in weights.items() if k.startswith('predictor.mask_embed.')}
 p=OUT/'initial_mask_embed.pt';torch.save(dict(state_dict=init,parent_checkpoint_sha256=HEAD_SHA,checkpoint_key=contract['checkpoint_key'],trained=False),p);add(p,str(cp),'model_state_dict_ema.predictor.mask_embed.*')
 p=OUT/'initial_class_embed.pt';ci={k.removeprefix('predictor.class_embed.'):v.clone() for k,v in weights.items() if k.startswith('predictor.class_embed.')};torch.save(ci,p);add(p,str(cp),'model_state_dict_ema.predictor.class_embed.*')
 head={k:v.cuda() for k,v in init.items()};classhead={k:v.cuda() for k,v in ci.items()};del weights
 losses=load_module('cache094_losses',REPO/'models/losses.py')
 tree=ast.parse((REPO/'models/maskvrd.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MaskVRD');fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='bipartite_match')
 env=dict(torch=torch,F=F,linear_sum_assignment=linear_sum_assignment,batch_masked_sigmoid_focal_loss=losses.batch_masked_sigmoid_focal_loss,batch_masked_dice_loss=losses.batch_masked_dice_loss)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(REPO/'models/maskvrd.py'),'exec'),env)
 matcher=types.SimpleNamespace(with_fuzzy=False,cost_factor={'cost_class':2.,'cost_mask':2.,'cost_dice':5.})
 pids=load_module('cache094_categories',REPO/'dataloaders/category.py').vidor_pred_name_to_id
 splitsets={};counts={};evalmeta={}
 for role in ['train','adapter_dev']:
  phase=role;mfile=OLD/role/'manifest.json';copy(mfile,'sources/splits/'+role+'.json',SPLIT_SHA[role]);m=json.loads(mfile.read_text());splitsets[role]=set(m['sampled_video_ids']);counts[role]={'pairs':0,'targets':0,'batches':len(m['records'])}
  for ri,ref in enumerate(m['records']):
   path=SRC/'fresh_raw'/f'{role}_{ri:04d}.pt';verify(path,artifact_hashes['fresh_raw/'+path.name]);raw=torch.load(path,map_location='cpu',weights_only=False);c=raw['hook']
   targetfile=Path(ref['path']);verify(targetfile,ref['sha256']);old=torch.load(targetfile,map_location='cpu',weights_only=False)
   require(raw['metadata']==old['metadata']['pair_metadata'],'TARGET_METADATA')
   require(torch.equal(c['valid'].squeeze(1),old['valid']),'TARGET_VALID')
   check=native_check(c,raw);check.update(role=role,batch=ri);checks.append(check)
   preds=old['preds_list'];targets=old['masks_list'];B=len(preds)
   with torch.no_grad():
    indices,_=env['bipartite_match'](matcher,c['cls'][-1].cuda(),[t.cuda() for t in preds],c['masks'][-1].cuda(),[t.cuda() for t in targets],None,c['valid'].cuda())
   enriched=[];eventrows=[]
   for b,pm in enumerate(raw['metadata']):
    v=pm['video_id'];ar=m['inputs'][v];ann=annotation(Path(ar['annotation']),ar['annotation_sha256']);tids=sorted({int(o['tid']) for f in ann['trajectories'] for o in f});sid,oid=[tids[pm['pair_key'][i]] for i in (0,1)]
    n=int(c['valid'][b].sum());frames=int(pm['sample_frame_origin'])+int(pm['feat_stride'])*torch.arange(n)
    er=[]
    for ti,(pid,target) in enumerate(zip(preds[b],targets[b])):
     require(target.shape[-1]==512 and bool(((target==0)|(target==1)).all()),'EVENT_TARGET')
     candidates=[]
     for gi,g in enumerate(ann['relation_instances']):
      if (g['subject_tid'],g['object_tid'],pids[g['predicate']])!=(sid,oid,int(pid)):continue
      sampled=(frames>=g['begin_fid'])&(frames<g['end_fid'])
      if torch.equal(sampled,target[:n].bool()):candidates.append(gi)
     require(len(candidates)==1,'ANNOTATION_EVENT_ID_AMBIGUOUS '+str((role,ri,b,ti,candidates)))
     gi=candidates[0];g=ann['relation_instances'][gi]
     er.append(dict(event_id=f'{role}/{ri}/{b}/{ti}',target_index=ti,annotation_gt_index=gi,predicate_id=int(pid),begin_fid=g['begin_fid'],end_fid=g['end_fid'],duration_seconds=(g['end_fid']-g['begin_fid'])/ann['fps'],positive_samples=int(target[:n].sum())))
    enriched.append(dict(**pm,subject_tid=sid,object_tid=oid,fps=ann['fps'],N_valid=n,events=er));eventrows.append(er)
   dest=OUT/role/path.name;dest.parent.mkdir(exist_ok=True)
   payload={k:c[k].clone() for k in ['f','hs','valid','cls','masks']};payload.update(preds=preds,targets=targets,metadata=enriched,source_raw_sha256=sources[str(path)],target_source_sha256=ref['sha256'],layer_contract='shared mask_embed; decoder layer3 only for E091 updates; other outputs cached')
   torch.save(payload,dest);add(dest,[str(path),str(targetfile)],'hook.f/hs/valid/cls/masks + source preds_list/masks_list; no numerical conversion')
   for b,(q,g) in enumerate(indices):
    matching.append(dict(role=role,batch=ri,row=b,video_id=enriched[b]['video_id'],pair_key=enriched[b]['pair_key'],layer=3,query_indices=q.tolist(),target_indices=g.tolist(),event_ids=[eventrows[b][j]['event_id'] for j in g.tolist()],cache_path=str(dest.relative_to(OUT)),cache_sha256=files[-1]['sha256']))
   counts[role]['pairs']+=B;counts[role]['targets']+=sum(len(p) for p in preds)
   records.append(dict(role=role,path=str(dest.relative_to(OUT)),pairs=B,targets=sum(len(p) for p in preds)))
   if ri%20==0: print('ASSEMBLE',role,ri,flush=True)
 require(counts['train']['pairs']==350 and counts['train']['targets']==563,'TRAIN_CENSUS')
 require(counts['adapter_dev']['pairs']==89 and counts['adapter_dev']['targets']==169,'DEV_CENSUS')
 require(len(splitsets['train'])==32 and len(splitsets['adapter_dev'])==8 and not splitsets['train']&splitsets['adapter_dev'],'SPLITS')
 phase='eval';locked=list(csv.DictReader((CODE/'eval_locked.csv').open()));members=json.loads((CODE/'members.json').read_text());coords=json.loads((CODE/'eval_coordinates.json').read_text())
 main={tuple(a) for a in members['long_coexist_short_covered']};sub={tuple(a) for a in members['long_coexist_subsecond_covered']};require(len(main)==400 and len(sub)==163,'MEMBERS')
 evideos={r['video_name'] for r in locked};require(len(locked)==2573 and len(evideos)==27 and not evideos&(splitsets['train']|splitsets['adapter_dev']),'EVAL_SPLIT')
 for v in sorted(evideos):
  ap=Path(cfg['paths']['annotations'])/v.split('_')[0]/(v.split('_')[1]+'.json');annotation(ap,old_sources[str(ap)])
 selected={(r['video_name'],int(r['pair_idx'])) for r in locked if r['pair_idx']!=''}
 rawfiles=sorted((SRC/'fresh_raw').glob('gt27_*.pt'));main_gate=[]
 for ri,path in enumerate(rawfiles):
  # retain all087 native pairs so the evaluable GT27 universe is not reduced
  verify(path,artifact_hashes['fresh_raw/'+path.name]);raw=torch.load(path,map_location='cpu',weights_only=False);c=raw['hook'];meta=raw['metadata'];key=(meta['video_name'],int(meta['pair_idx']))
  check=native_check(c,raw);check.update(role='eval',key=list(key));checks.append(check)
  if key in selected:
   expected=coords[f'{key[0]}/{key[1]}'];require(all(int(meta[k])==int(expected[k]) for k in ['origin','N_valid','subject_tid','object_tid','pair_start','pair_end']),'EVAL_COORDINATES')
  dest=OUT/'eval'/path.name;dest.parent.mkdir(exist_ok=True);payload={k:c[k].clone() for k in ['f','hs','valid','cls','masks']};payload.update(metadata=meta,source_raw_sha256=sources[str(path)])
  torch.save(payload,dest);add(dest,str(path),'hook.f/hs/valid/cls/masks + metadata; no numerical conversion');evalmeta[key]=dict(path=str(dest.relative_to(OUT)),meta=meta)
  for row in locked:
   if row['pair_idx']=='' or (row['video_name'],int(row['pair_idx']))!=key:continue
   q=int(row['query']);pid=pids[row['predicate']];n=int(meta['N_valid']);require(int(c['cls'][-1,0,:,pid].argmax())==q,'FIXED_QUERY_CHANGED')
   if (row['video_name'],int(row['gt_index'])) in main:
    indices=torch.where(c['masks'][-1,0,q,:n].sigmoid()>.5)[0];eligible=bool(len(indices) and int(indices[-1]-indices[0])*4+1>=5);main_gate.append(eligible)
  if ri%200==0:print('ASSEMBLE eval',ri,'/',len(rawfiles),flush=True)
 require(selected<=set(evalmeta),'MISSING_EVAL_PAIR');require(len(main_gate)==400 and sum(main_gate)==381,'FIXED_H_GATE')
 counts['eval']=dict(native_pairs=len(rawfiles),locked_GT=len(locked),selected_pairs=len(selected),main400=len(main),subsecond163=len(sub),main_H_eligible=sum(main_gate),no_pair_GT=sum(r['pair_idx']=='' for r in locked),long_protection_GT=sum(float(r['relation_sec'])>=3 and float(r['coexist_sec'])>=10 and r['pair_idx']!='' for r in locked))
 p=dump('frozen_matching.json',dict(status='FROZEN_ONCE',source_function='MaskVRD.bipartite_match',costs=[2,2,5],layer=3,initial_mask_embed_sha256=next(f['sha256'] for f in files if f['path']=='initial_mask_embed.pt'),entries=matching));add(p,'same087 native logits + original independent targets; baseline Hungarian')
 p=dump('forward_consistency.json',dict(status='PASS',tolerance=TOL,settings=dict(device=torch.cuda.get_device_name(0),torch=str(torch.__version__),tf32_matmul=False,tf32_cudnn=True,cudnn_benchmark=False,threads=2),checks=checks,encoder_forwards=0,optimizer_steps=0,E091_evaluation=False,scope='frozen mask/class branch reconstruction and input consistency only; no time performance metrics',frozen=['encoder','query','class','trajectory','matching','early decoder cached outputs'],trainable_future='shared mask_embed applied ONLY to decoder layer3'));add(p,'094 cached branch checks')
 p=dump('source_sha256.json',sources);add(p,'verified source files')
 dump('manifest.json',dict(status='REMOTE_ASSEMBLY_PASS_PENDING_LOCAL_IMPORT',counts=counts,split_videos={k:sorted(v) for k,v in splitsets.items()},eval_videos=sorted(evideos),files=files,records=records,source_root=str(SRC),bytes=sum(f['bytes'] for f in files),training=False,E091_evaluation=False,elapsed_seconds=time.time()-start))
 print('ASSEMBLY_PASS',json.dumps(counts),'bytes',sum(f['bytes'] for f in files),flush=True)
except Exception as e:
 dump('forward_consistency.json',dict(status='TECHNICAL_FAILURE',phase=phase,error=repr(e),tolerance=TOL,completed_checks=checks,optimizer_steps=0,E091_evaluation=False))
 dump('manifest.json',dict(status='TECHNICAL_FAILURE',phase=phase,error=repr(e),files=files,sources=sources,training=False))
 print('TECHNICAL_FAILURE',phase,repr(e),flush=True)
 raise
