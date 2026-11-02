#!/usr/bin/env python3
"""Frozen batch of observational contrasts; existing artifacts only."""
import argparse,csv,gzip,json,time
from pathlib import Path
from controlled import compare,load_csv
from run import sha,write_csv

def configurations():
    plans=[]
    contrasts=[('H1',False,False),('H2',False,False),('H2',True,False),('H5',False,False),('H5',False,True)]
    for model in ('VrdONE','BIG'):
        for population in ('all','native_F'):
            for h,no_comp,load in contrasts:
                plans.append(dict(dataset='native',hypothesis=h,outcome=model+'_official_miss',population=population,
                                  h2_no_comp_control=no_comp,load_control=load))
    for h,no_comp,load in contrasts:
        plans.append(dict(dataset='GT27',hypothesis=h,outcome='miss_tiou05',population='all',h2_no_comp_control=no_comp,load_control=load))
    for h,no_comp,load in [('H1',False,False),('H2',True,False),('H5',False,True)]:
        for outcome in ('miss_tiou03','miss_tiou07'):
            plans.append(dict(dataset='GT27',hypothesis=h,outcome=outcome,population='all',h2_no_comp_control=no_comp,load_control=load))
    for population in ('all','chosen_pair_covers_gt','chosen_pair_covers_gt_unique_annotation'):
        for outcome in ('stretch','overlap_stretch','no_overlap_stretch'):
            plans.append(dict(dataset='GT27',hypothesis='H1',outcome=outcome,population=population,h2_no_comp_control=False,load_control=False))
        if population!='all':
            plans.append(dict(dataset='GT27',hypothesis='H1',outcome='miss_tiou05',population=population,h2_no_comp_control=False,load_control=False))
    return plans

def run(inputs,out):
    inputs=Path(inputs);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plans=configurations()
    config=dict(definition='overlap_other_pred_ge3',repetitions=2000,seed=80,
                bootstrap_methods=['resample','exp_multiplier'],plans=plans)
    (out/'frozen_batch.json').write_text(json.dumps(config,indent=2)+'\n')
    data={'native':load_csv(inputs/'per_gt_slices.csv'),'GT27':load_csv(inputs/'per_gt_boundary_joined.csv')}
    for r in data['native']:
        r.update(source_scope='VidOR_val_30142GT_832_GT_bearing_videos',readout='official_Top100_vIoU50')
    summary=[];started=time.monotonic()
    for i,plan in enumerate(plans):
        for method in config['bootstrap_methods']:
            args={k:v for k,v in plan.items() if k!='dataset'}
            result=compare(data[plan['dataset']],**args,definition=config['definition'],repetitions=2000,seed=80,bootstrap_method=method)
            name=f'{i:03d}_{method}.json.gz'
            with gzip.open(out/name,'wt',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
            flat={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
            flat.update({f'bootstrap_{k}':v for k,v in result['bootstrap'].items()})
            flat.update({f'judgment_{k}':v for k,v in result['judgment'].items()})
            flat['detail_file']=name;summary.append(flat)
        print(json.dumps(dict(done=i+1,total=len(plans),plan=plan,gap_pp=result['standardized_gap_pp'],
                              N=result['N_common_support'],elapsed=round(time.monotonic()-started,1)),ensure_ascii=False),flush=True)
    write_csv(out/'controlled_summary.csv',summary)
    receipt=dict(complete=True,model_or_training_executed=False,N_comparisons=len(summary),
        sources={p.name:dict(path=str(p.resolve()),sha256=sha(p)) for p in (inputs/'per_gt_slices.csv',inputs/'per_gt_boundary_joined.csv')},
        code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('controlled.py'))},
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.inputs,a.out)
