# Literature map R11 — temporal GIoU / tIoU in set matching

| Work | Venue | Link | Why it motivates H114 |
|---|---|---|---|
| **DETR** (Carion et al.) | ECCV 2020 | https://arxiv.org/abs/2005.12872 | Matching cost = λ_cls·CE + λ_L1·‖b−b̂‖ + **λ_giou·GIoU**; localization geometry is first-class in assignment |
| **Moment-DETR** (Lei et al.) | NeurIPS 2021 | https://arxiv.org/abs/2107.09609 | Same recipe on **(center,width)** spans; GIoU on time is what made set-prediction work for moments |
| **DINO** (Zhang et al.) | ICLR 2023 | https://arxiv.org/abs/2203.03605 | Denoising + look-forward-twice still keeps **GIoU** in the matching/loss stack — geometry cost is not optional |
| **ActionFormer** (Zhang et al.) | ECCV 2022 | https://arxiv.org/abs/2202.07925 | TAL ranking/NMS driven by **tIoU**; short actions need explicit temporal-overlap signal |
| **TadTR** (Liu et al.) | TIP 2022 | https://arxiv.org/html/2106.10271 | Temporal DETR variant; segment IoU central to refine queries |
| VrdONE MaskVRD | MM 2024 | https://arxiv.org/html/2408.09408v2 | `bipartite_match`: class+focal-mask+dice **only** — `gt_segs` unused in non-fuzzy geometric cost |

**Contrast with closed labs:** H110 = duration **reweight** of mask/dice/class (no new geometric term). H113 = replace hull with cxw **decode**. H114 = add **cost_tgiou** between pred-mask-induced span and GT seg at matching time; train/eval decode stays mask→hull.

**X control:** same GIoU cost weight, but GT seg endpoints **time-shuffled within clip** when building cost_tgiou (capacity-matched; temporal geometry destroyed).
