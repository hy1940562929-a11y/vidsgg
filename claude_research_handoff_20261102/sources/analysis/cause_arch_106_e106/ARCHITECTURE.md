# M106-TEMPORAL-ADAPTER：缓存特征上的残差时间适配器

```text
cached x (B,C,T=64) → TemporalAdapter (depthwise/grouped Conv1d residual) → predictor
X: flip(x, time) → same adapter → predictor
N: x → predictor (no adapter)
```

近恒等初始化（残差支路零初始化），避免起步破坏 C0。文献：ActionFormer 时序建模；adapter 残差。
