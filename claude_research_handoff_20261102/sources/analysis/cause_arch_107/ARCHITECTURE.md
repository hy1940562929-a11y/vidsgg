# M107-MASKFEAT-ADAPTER：对 `f` 的残差时间适配

```text
x (unchanged) ─┐
               ├─ predictor → class, mask_embed
f (B,256,512) → TemporalAdapter → mask_features ─┘ einsum → pred_masks
X: adapter(flip(f))
N: raw f
```

Adapter：2×Conv1d(256→256,k=3)+GELU，第二层零初始化残差。
