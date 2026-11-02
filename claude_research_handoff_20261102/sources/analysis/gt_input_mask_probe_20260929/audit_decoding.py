#!/usr/bin/env python3
"""Check actual saved logits against the original GPU sigmoid threshold."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from evaluate_masks import sigmoid_positive, sha

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--run',type=Path,required=True)
    pa.add_argument('--out',type=Path,required=True);args=pa.parse_args()
    assert not args.out.exists();start=time.time();checks=[]
    for condition in ['native','gt']:
        for full in json.loads((args.run/'selection.json').read_text()):
            count=0;diff=0;minabs=float('inf')
            with np.load(args.run/condition/'raw'/(full+'.npz')) as z:
                for key in z.files:
                    if not key.startswith('mask_'):continue
                    x=z[key];n= x[x!=0]
                    if n.size:minabs=min(minabs,float(np.abs(n).min()))
                    original=(torch.from_numpy(x).cuda().sigmoid()>.5).cpu().numpy()
                    decoded=sigmoid_positive(x)
                    count+=x.size;diff+=int(np.sum(original!=decoded))
            checks.append(dict(condition=condition,video=full,mask_logit_values=count,
                               numpy_vs_original_cuda_sigmoid_mismatches=diff,min_nonzero_abs_logit=minabs))
    total=sum(r['mask_logit_values'] for r in checks);bad=sum(r['numpy_vs_original_cuda_sigmoid_mismatches'] for r in checks)
    result=dict(complete=True,passed=bad==0,compared_mask_values=total,mismatches=bad,
                checks=checks,elapsed_seconds=time.time()-start,script_sha256=sha(Path(__file__)))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    assert bad==0,'CPU diagnostic decode differs from original GPU sigmoid'

if __name__=='__main__':main()
