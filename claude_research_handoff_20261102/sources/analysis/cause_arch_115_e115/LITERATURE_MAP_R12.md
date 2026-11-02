# Literature map R12 — multi-scale temporal features for short actions

| Work | Venue | Link | Why it motivates H115 |
|---|---|---|---|
| **ActionFormer** (Zhang et al.) | ECCV 2022 | https://arxiv.org/abs/2202.07925 | Multi-scale transformer on temporal pyramid; short actions detected on finer scales |
| **TriDet** (Shi et al.) | CVPR 2023 | https://arxiv.org/abs/2303.07347 | Trident-head on multi-scale features; argues single-scale hurts boundary precision |
| **VSGN** (Zhao et al.) | ICCV 2021 | https://arxiv.org/abs/2108.11155 | Pyramid grouping for short vs long proposals |
| **TadTR** (Liu et al.) | TIP 2022 | https://arxiv.org/html/2106.10271 | Temporal deformable attn still benefits from multi-resolution memory |
| **Video-Swin / MViT** | CVPR/ICCV | multi-scale video backbones | Scale hierarchy is default for spatiotemporal recognition |
| VrdONE | MM 2024 | https://arxiv.org/html/2408.09408v2 | Single-scale `f` into MaskVRD decoder — **no temporal FPN** |

**Contrast:** H106/107 = residual Conv adapter (not pyramid). H105 soft decode. H114 match GIoU (assignment, not representation).

**X control:** same FPN params, **time-reversed `f`** before pyramid (capacity matched; temporal order destroyed).
