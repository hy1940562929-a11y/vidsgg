#!/usr/bin/env python3
"""Read-only annotation vocabulary coverage. Not effective sampler exposure."""
import argparse
import collections
import csv
import hashlib
import json
import zipfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def extract(ann):
    cats={int(x['tid']):x['category'] for x in ann['subject/objects']}
    return [(cats[int(r['subject_tid'])],r['predicate'],cats[int(r['object_tid'])]) for r in ann['relation_instances']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-zip',type=Path,required=True);p.add_argument('--val-dir',type=Path,required=True);p.add_argument('--per-gt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    train=collections.Counter();tp=collections.Counter();tn=set()
    with zipfile.ZipFile(a.train_zip) as z:
        names=sorted(n for n in z.namelist() if n.startswith('training/') and n.endswith('.json'))
        for i,n in enumerate(names):
            ann=json.loads(z.read(n));rs=extract(ann);train.update(rs);tp.update(t[1] for t in rs);tn.add(Path(n).parent.name+'_'+Path(n).stem)
            if (i+1)%1000==0:print('training_annotations_read',i+1,flush=True)
    val=collections.Counter();vp=collections.Counter();vn=set();vr={};val_hashes={}
    for f in sorted(a.val_dir.glob('*/*.json')):
        full=f.parent.name+'_'+f.stem;vn.add(full);rs=extract(json.loads(f.read_text()));val.update(rs);vp.update(t[1] for t in rs);val_hashes[str(f)]=digest(f)
        for gi,t in enumerate(rs):vr[full,gi]=t
    assert not tn&vn
    with a.per_gt.open(encoding='utf-8-sig') as f: old=list(csv.DictReader(f))
    assert len(old)==len(vr),('evaluation_vs_annotation_GT_count',len(old),len(vr))
    rows=[]
    for r in old:
        k=(r['video_name'],int(r['gt_index']));t=(r['subject_class'],r['predicate'],r['object_class']);assert vr[k]==t
        rows.append(dict(video_name=k[0],gt_index=k[1],subject=t[0],predicate=t[1],object=t[2],training_predicate_instances=tp[t[1]],training_triplet_instances=train[t],unseen_predicate=int(tp[t[1]]==0),unseen_triplet=int(train[t]==0),relation_sec=float(r['relation_sec']),VrdONE_hit=int(r['VrdONE_hit']),BIG_hit=int(r['BIG_hit'])))
    novel=[t for t in val if t not in train]
    groups={}
    for label,rr in [('seen_triplet',[r for r in rows if not r['unseen_triplet']]),('unseen_triplet',[r for r in rows if r['unseen_triplet']])]:
        groups[label]={'N_GT':len(rr),'VrdONE_hit':sum(r['VrdONE_hit'] for r in rr),'BIG_hit':sum(r['BIG_hit'] for r in rr),'short_GT':sum(r['relation_sec']<3 for r in rr)}
    summary={'complete':True,'definition':'Raw official-split annotation presence; not proof of effective sampled or checkpoint-specific exposure','training_videos':len(tn),'validation_videos':len(vn),'video_overlap':len(tn&vn),'training_relation_instances':sum(train.values()),'validation_relation_instances':sum(val.values()),'training_predicates':len(tp),'validation_predicates':len(vp),'unseen_validation_predicates':sorted(set(vp)-set(tp)),'unseen_predicate_validation_instances':sum(v for k,v in vp.items() if k not in tp),'training_unique_class_triplets':len(train),'validation_unique_class_triplets':len(val),'unseen_validation_class_triplets':len(novel),'unseen_class_triplet_validation_instances':sum(val[t] for t in novel),'unseen_triplet_examples':[{'subject':t[0],'predicate':t[1],'object':t[2],'val_instances':val[t]} for t in sorted(novel,key=lambda t:(-val[t],t))[:20]],'groups':groups,'predicate_counts':{'training':dict(tp),'validation':dict(vp)},'source_sha256':{str(a.train_zip):digest(a.train_zip),str(a.per_gt):digest(a.per_gt),str(Path(__file__).resolve()):digest(Path(__file__))},'validation_annotation_sha256':val_hashes,'guard':'All30142 evaluation GT keys/class triples match annotation indices; no train/validation video overlap'}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (a.out/'per_gt_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('predicate_counts','source_sha256','validation_annotation_sha256')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
