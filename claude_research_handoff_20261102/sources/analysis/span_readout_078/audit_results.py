"""Independent stdlib-only recomputation of imported078 numeric summaries."""
import argparse,json,gzip,hashlib,statistics
from pathlib import Path

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('import_root');ap.add_argument('out');a=ap.parse_args()
    root=Path(a.import_root);r=root/'results_v1';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    s=read(r/'summary.json');assert s['complete'];rows=read(r/'per_gt.json.gz');vs=read(r/'per_video.json')
    assert len(rows)==4739 and len(vs)==128
    arms=tuple(s['cohorts']['combined']['mAP'])
    groups={'all':lambda x:True,'short':lambda x:x['relation_sec']<3,
        'subsecond':lambda x:x['relation_sec']<1,'one_to_three':lambda x:1<=x['relation_sec']<3,
        'long':lambda x:x['relation_sec']>=3,'C':lambda x:x['original_group']=='C',
        'short_C':lambda x:x['original_group']=='C' and x['relation_sec']<3,
        'long_coexist_short':lambda x:x['relation_sec']<3 and x['joint_local_continuous_sec']>=10,
        'long_coexist_persistent':lambda x:x['relation_sec']>=3 and x['joint_local_continuous_sec']>=10}
    compact={};changes=[]
    for cohort,stored in s['cohorts'].items():
        rr=rows if cohort=='combined' else [x for x in rows if x['cohort']==cohort]
        vv=vs if cohort=='combined' else [x for x in vs if x['cohort']==cohort]
        recomputed={'N_GT':len(rr),'N_videos':len(vv),'mAP':{},'groups':{}}
        for arm in arms:
            m=statistics.mean(x['AP'][arm] for x in vv if x['N_GT'])
            assert abs(m-stored['mAP'][arm])<1e-12
            recomputed['mAP'][arm]=m
        for group,fn in groups.items():
            subset=[x for x in rr if fn(x)];record={'N_GT':len(subset),'arms':{}}
            assert len(subset)==stored['groups'][group]['N_GT']
            for arm in arms:
                d={}
                for k in (50,100,200):
                    hit=lambda x,aa:0<=x['ranks'][aa]<k
                    h=sum(hit(x,arm) for x in subset)
                    new=sum(hit(x,arm) and not hit(x,'baseline') for x in subset)
                    lost=sum(hit(x,'baseline') and not hit(x,arm) for x in subset)
                    original=stored['groups'][group]['arms'][arm]
                    assert h==original['hits'][str(k)]
                    assert {'new':new,'lost':lost,'net':new-lost}==original['comparisons']['baseline'][str(k)]
                    d[str(k)]={'hits':h,'N_GT':len(subset),'new':new,'lost':lost,'net':new-lost}
                record['arms'][arm]=d
            recomputed['groups'][group]=record
        compact[cohort]=recomputed
    for row in rows:
        for arm in arms:
            if arm=='baseline':continue
            old=0<=row['ranks']['baseline']<100;new=0<=row['ranks'][arm]<100
            if old!=new:changes.append(dict(row,arm=arm,change='gained' if new else 'lost'))
    seal=read(r/'intervals_seal_before_GT.json');assert len(seal)==128
    for p,h in seal.items():assert digest(r/p)==h
    for p,h in read(root/'code_v1'/'software_freeze.json').items():assert digest(root/'code_v1'/p)==h
    c=compact['combined'];assert c['groups']['all']['arms']['baseline']['100']['hits']==588
    assert c['groups']['long_coexist_short']['arms']['baseline']['100']['hits']==6
    strict={}
    for arm in ('joint','joint_hard'):
        controls=('baseline','mask','independent') if arm=='joint' else ('baseline','mask','independent','joint','joint_random')
        hit=lambda aa,g='long_coexist_short':c['groups'][g]['arms'][aa]['100']['hits']
        strict[arm]={
            'target_exceeds_controls':all(hit(arm)>hit(x) for x in controls),
            'overall_noninferior':all(c['mAP'][arm]>=c['mAP'][x] and hit(arm,'all')>=hit(x,'all') for x in ('baseline','independent')),
            'persistent_no_net_loss':hit(arm,'long_coexist_persistent')>=hit('baseline','long_coexist_persistent'),
            'target_cohorts_no_loss':all(compact[k]['groups']['long_coexist_short']['arms'][arm]['100']['net']>=0 for k in ('047','048'))}
    result=dict(numeric_recomputation='PASS',prediction_seal='PASS',code_freeze='PASS',
        strict_preregistered_gates=strict,decision='SMALL_PILOT_CONTINUATION_SIGNAL' if any(all(x.values()) for x in strict.values()) else 'DO_NOT_EXPAND_THIS_VERSION',
        elapsed_seconds=s['elapsed_seconds'],cohorts=compact)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (out/'changed_cases.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
    print('VERIFICATION PASS',result['decision'],'changed GT/arm rows',len(changes))
    for arm in arms:
        print(arm,'mAP',round(c['mAP'][arm]*100,4),'overall',c['groups']['all']['arms'][arm]['100'],
            'short',c['groups']['short']['arms'][arm]['100'],'target',c['groups']['long_coexist_short']['arms'][arm]['100'])

if __name__=='__main__':main()
