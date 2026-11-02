# M113-SPAN-CXW：Moment-DETR 风格辅助跨度头

```text
decoder hs → SpanHead MLP → (cx, w) ∈ (0,1)² per query
Loss_T = L_native + λ L1(cxw, gt_cxw_matched)
Loss_X = L_native + λ L1(cxw, shuffle_gt_cxw)
Loss_N = L_native
Eval_T/X: interval from cxw on valid length → frames (min 5)
Eval_N/C0: native sigmoid>0.5 hull
```
λ=2.0；gt_cxw from target mask support.
