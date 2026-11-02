"""Wiring-only regression: original layout and layer-3-only loss connection."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from run import Head

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=False
cache=Path(sys.argv[1])
state=torch.load(cache/'initial_mask_embed.pt',map_location='cpu',weights_only=False)['state_dict']
model=Head(state).cuda()
torch.manual_seed(95)
hs=torch.randn(4,256,9,device='cuda',requires_grad=True)
f=torch.randn(1,256,32,device='cuda')
z=model(hs,f)
x=hs
for i,l in enumerate(model.layers):
    x=F.conv1d(x,l.weight,l.bias)
    if i<2:x=F.gelu(x)
ref=torch.einsum('lbqc,bcm->lbqm',x.view(4,1,256,9).permute(0,1,3,2),f)[3]
assert torch.equal(z,ref)
z.square().sum().backward()
assert torch.count_nonzero(hs.grad[:3])==0
assert torch.count_nonzero(hs.grad[3])>0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('ALIGNMENT_PASS; exact native layout; layers0-2 input gradients zero; no optimizer')
