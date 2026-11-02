"""Read-only native branch alignment audit; no optimizer or performance evaluation."""
import argparse, json, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from run import Head

p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
a=p.parse_args();cache=Path(a.cache);out=Path(a.out)
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=False
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
init=load(cache/'initial_mask_embed.pt')
head=Head(init['state_dict']).cuda().eval()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'init_sha256':sha(cache/'initial_mask_embed.pt'),'trained':init['trained'],
        'settings':dict(torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
                        matmul_tf32=False,cudnn_tf32=True,benchmark=False,threads=2),
        'checks':[]}
def check(z,ref):
    z=z.cpu();ref=ref.cpu();finite=bool(torch.isfinite(z).all() and torch.isfinite(ref).all())
    return dict(max_abs=float((z-ref).abs().max()) if finite else None,
                finite=finite,exact=torch.equal(z,ref),
                bits_equal=torch.equal(z.sigmoid()>.5,ref.sigmoid()>.5))
with torch.no_grad():
    for role in ['train','adapter_dev','eval']:
        for path in sorted((cache/role).glob('*.pt')):
            c=load(path);B=c['f'].shape[0];hs=c['hs'].cuda();f=c['f'].cuda()
            valid=c['valid'].cuda();last=hs.reshape(4,B,256,9)[3]
            # Frozen failed 095 path.
            old=head(last,f).masked_fill(~valid,-10)
            # Exact 094 path: preserve four-layer batch and einsum layout.
            x=hs
            for i,l in enumerate(head.layers):
                x=F.conv1d(x,l.weight,l.bias)
                if i<2:x=F.gelu(x)
            all094=torch.einsum('lbqc,bcm->lbqm',
                x.view(4,B,256,9).permute(0,1,3,2),f).masked_fill(~valid[None,...],-10)
            # Isolate contraction vs convolution batch layout.
            final_contract=torch.einsum('bcq,bct->bqt',
                x.view(4,B,256,9)[3],f).masked_fill(~valid,-10)
            lx=last
            for i,l in enumerate(head.layers):
                lx=F.conv1d(lx,l.weight,l.bias)
                if i<2:lx=F.gelu(lx)
            row=dict(path=str(path.relative_to(cache)),B=B,
                hs_shape=list(hs.shape),f_shape=list(f.shape),
                hs_dtype=str(hs.dtype),f_dtype=str(f.dtype),device=str(hs.device),
                original095=check(old,c['masks'][3]),
                exact094=check(all094,c['masks']),
                full_conv_final_einsum=check(final_contract,c['masks'][3]),
                final_vs_full_conv=check(lx,x.view(4,B,256,9)[3]))
            result['checks'].append(row)
            if len(result['checks'])<=2 or len(result['checks'])%200==0:
                print(json.dumps(row),flush=True)
            if not row['exact094']['finite'] or row['exact094']['max_abs']>4e-6 or not row['exact094']['bits_equal']:
                result['status']='TECHNICAL_FAILURE_094_PATH';break
        if result.get('status'):break
result.setdefault('status','AUDIT_COMPLETE')
(out/'rebuild_diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('AUDIT',result['status'],len(result['checks']),flush=True)
