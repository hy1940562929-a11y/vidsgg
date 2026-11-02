"""Synthetic checks only; no optimization or video inference."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import run

cache=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('source_losses',cache/'sources/repo/models/losses.py')
run.LOSS=importlib.util.module_from_spec(spec);spec.loader.exec_module(run.LOSS)
torch.manual_seed(95)
z=torch.randn(2,512,requires_grad=True)
v=torch.zeros(2,512,dtype=torch.bool);v[0,:30]=True;v[1,:55]=True
y=torch.zeros_like(z);y[0,4:8]=1;y[1,30:35]=1
p=z.sigmoid();pt=p*y+(1-p)*(1-y)
direct=2*(torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')*(1-pt)**2*(.25*y+.75*(1-y))*v).mean(1).mean()+5*(1-(2*(p*y*v).sum(-1)+1)/((p*v).sum(-1)+(y*v).sum(-1)+1)).mean()
assert torch.equal(run.orig_loss(z,y,v),direct)
perm=run.make_perm(y,v,['v/p/e0','v/p/e1'],90)
assert torch.equal(perm,run.make_perm(y,v,['v/p/e0','v/p/e1'],90))
assert torch.allclose(run.orig_loss(z.gather(1,perm),y,v),run.orig_loss(z,y,v),atol=1e-6,rtol=0)
d=torch.tensor([16.,20.])
for perm_ in [None,perm]:
    loss=run.geometry(z,y,v,d,perm_)
    grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[~v])==0
    rho=torch.nn.functional.logsigmoid(z).masked_fill(~v,-torch.inf).softmax(-1)
    if perm_ is not None:rho=rho.gather(1,perm_)
    ref=0.
    for i in range(2):
        n=int(v[i].sum())
        ref+=4*(rho[i,:n].cumsum(0)-(y[i,:n]/y[i].sum()).cumsum(0)).abs()[:-1].sum()/d[i]/2
    assert torch.allclose(loss,ref,atol=1e-6,rtol=0)
print('TRAINING_SYNTHETIC_CHECKS_PASS; no optimizer constructed')
