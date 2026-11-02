# Literature map R10 — continuous span supervision

| Work | Venue | Link | Why it motivates H113 |
|---|---|---|---|
| **Moment-DETR** (Lei et al.) | NeurIPS 2021 | https://arxiv.org/abs/2107.09609 · [code](https://github.com/jayleicn/moment_detr) | Direct set prediction of **normalized (center, width)** spans + L1/GIoU; treats localization as DETR boxes on time |
| **ActionFormer** (Zhang et al.) | ECCV 2022 | https://arxiv.org/abs/2202.07925 | Moment-wise classify + **boundary distance regression** from multi-scale features — argues against sole hard-binarized support |
| **TadTR** (Liu et al.) | TIP 2022 | https://arxiv.org/html/2106.10271 | Iterative temporal span refinement in DETR-like decoder |
| **TE-TAD** | CVPR 2024 | https://arxiv.org/html/2404.02405 | Time-aligned coordinate decoding for action detection |
| VrdONE | MM 2024 | https://arxiv.org/html/2408.09408v2 | Baseline: mask logits → sigmoid>0.5 hull — **no continuous span head** |

**Contrast with closed labs:** H105 = soft mass **decode-only** (same weights); 078 = start–end heads without this Moment-DETR cxw+GIoU framing / joint native mask path; H106/107 = feature adapters not span outputs.

**X control:** same span head, **time-shuffled GT (cx,w)** targets — capacity matched, temporal span signal destroyed.
