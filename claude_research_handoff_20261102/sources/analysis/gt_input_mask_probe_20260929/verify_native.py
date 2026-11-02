#!/usr/bin/env python3
"""Verify every reused native raw output against a fresh frozen forward."""
import argparse, inspect, json, os, sys, textwrap, types, time
from pathlib import Path
import numpy as np
import torch
from run_probe import ROOT, CACHE, load, dump, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    target=run/'native_all_video_consistency.json';assert not target.exists()
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    from dataloaders.vidor import VidOR
    from models.maskvrd import MaskVRD
    import utils.misc as utils
    utils.set_seed(42);torch.set_num_threads(2);start=time.time()
    cfg=json.loads((run/'effective_config.json').read_text());ds=VidOR.__new__(VidOR)
    for k in ['with_clip_feature','feat_stride','proposal_min_frames','random_stride','stride_offset']:setattr(ds,k,cfg['dataset_config'][k])
    net=MaskVRD(cfg['model_config'],device='cuda:0');net.eval();net._config_eval(cfg['inference_config']);net.cuda()
    cp=ROOT/'experiments/vrdone_vidor/ckpt_vidor.pth'
    assert sha(cp)=='4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627'
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    net.load_state_dict({k.removeprefix('module.'):v for k,v in ck['model_state_dict_ema'].items()},strict=True);del ck
    src=textwrap.dedent(inspect.getsource(MaskVRD.forward_test));needle='    ## remove pairs without interaction\n';assert src.count(needle)==1
    src=src.replace(needle,'    return predictions\n'+needle);ns=dict(vars(sys.modules[MaskVRD.__module__]))
    exec(compile(src,'<verify-raw-native>','exec'),ns);net.forward_test=types.MethodType(ns['forward_test'],net)
    rows=[]
    for full in json.loads((run/'selection.json').read_text()):
        inp=ds._val_getitem(load(CACHE/(full+'.pkl')));saved=load(run/'native/inputs'/(full+'.pkl'))
        for k in ['sids','oids','traj_durations','cat_ids','cat_scores','so_offset']:assert torch.equal(inp[k],saved[k])
        for x,y in zip(inp['bboxes_list'],saved['bboxes_list']):assert torch.equal(x,y)
        inp['video_name']=full;gpu=utils.dict_to_device(inp,'cuda:0')
        with torch.no_grad():raw=net(gpu)
        md=0.;flips=0
        with np.load(run/'native/raw'/(full+'.npz')) as z:
            for i in range(len(inp['sids'])):
                n=int(raw['output_mask'][i].sum());m=raw['pred_masks'][i][:,:n].cpu().numpy();old=z[f'mask_{i}']
                md=max(md,float(np.abs(m-old).max()));flips+=int(np.sum((m>0)!=(old>0)))
            ld=float(np.abs(torch.stack(raw['pred_logits']).cpu().numpy()-z['pred_logits']).max())
        row=dict(video=full,mask_max_abs_diff=md,logit_max_abs_diff=ld,threshold_flips=flips,passed=md<1e-4 and ld<1e-4 and flips==0)
        rows.append(row);dump(target,dict(complete=False,videos=rows));print(json.dumps(row),flush=True)
        assert row['passed'],('native equivalence failed',row)
        del inp,gpu,raw,saved;torch.cuda.empty_cache()
    dump(target,dict(complete=True,videos=rows,all_pass=True,elapsed_seconds=time.time()-start,script_sha256=sha(Path(__file__).resolve())))

if __name__=='__main__':main()
