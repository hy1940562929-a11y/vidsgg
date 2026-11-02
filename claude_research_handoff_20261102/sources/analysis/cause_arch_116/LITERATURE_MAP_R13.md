# Literature map R13 — span aux without decode swap

| Work | Venue | Link | Why H116 |
|---|---|---|---|
| **Moment-DETR** | NeurIPS 2021 | https://arxiv.org/abs/2107.09609 | L1+GIoU on (cx,w) is the training signal; decode is separate |
| **DETR** | ECCV 2020 | https://arxiv.org/abs/2005.12872 | Box L1+GIoU loss alongside class; masks optional parallel head |
| **Mask DINO** | CVPR 2023 | https://arxiv.org/abs/2206.02777 | Parallel mask + box heads; box aux helps localization without replacing mask |
| **ActionFormer** | ECCV 2022 | https://arxiv.org/abs/2202.07925 | Boundary regression aux with classification — not decode replacement |
| VrdONE | MM 2024 | mask→hull eval | H113 swapped decode; H116 keeps hull, adds span aux only |

**Contrast:** H113 = aux **and** span-as-eval. H114 = match GIoU only. H116 = train-time span L1+GIoU on matched queries; eval hull.
**X:** shuffle GT (cx,w).
