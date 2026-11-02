# Literature map for remaining short-interval residual (2026-10-21)

Grace/supervisor. Builds on [091 survey](../../literature/short_interval_cause_arch_091_survey.md) and H100–H104 outcomes. **No fabricated citations.** Links are arXiv/CVF/ECVA/official pages checked this turn or already F-read in 091.

Reading tags: **F** = prior lab F-read / key sections; **A** = abstract+metadata verified this turn; **R** = reused 091 row.

---

## Cluster → status vs this project

### C1. Video relation / VidSGG set prediction

| Work | Source | Status |
|---|---|---|
| VrdONE (MM2024) | https://arxiv.org/html/2408.09408v2 · [code](https://github.com/lucaspk512/vrdone) | **R/F** Baseline; mask+Hungarian already present — not an untested cause |
| BIG (CVPR2022) | https://arxiv.org/pdf/2112.04222 | **R/F** Multi-instance grounding; capacity not shown as our main residual (E097 old protocol stopped) |
| DETR (ECCV2020) | https://arxiv.org/html/2005.12872 | **R/F** Matching prior; H100 tested depth assignment |

**Open gap:** not “invent set prediction”; gap is **short interval under long coexistence on VrdONE masks**.

### C2. DETR matching / deep supervision

| Work | Source | Status |
|---|---|---|
| Stable-DINO (ICCV2023) | https://arxiv.org/html/2304.04742 | **R/M** Cross-layer match instability — **tested as H100 → REJECT** |
| DN-DETR (CVPR2022) | https://arxiv.org/html/2203.01305 | **R/A** Denoising queries — **not tried**; would add queries (different family) |
| RTD-Net (ICCV2021) | https://arxiv.org/pdf/2102.01894 | **R/F** Relaxed temporal matching — not H100; open but overlaps capacity/query |

**Do not retry:** Stable-DINO-style path-joint assignment alone.

### C3. Temporal action localization (TAL)

| Work | Source | Status |
|---|---|---|
| ActionFormer (ECCV2022) | https://arxiv.org/abs/2202.07925 · HTML https://arxiv.org/html/2202.07925v2 | **A this turn** Moment-wise classify + **boundary regression** from features — argues against sole reliance on hard binarized support |
| TadTR (TIP2022) | https://arxiv.org/html/2106.10271 | **R/F** Iterative span refinement |
| TE-TAD (CVPR2024) | https://arxiv.org/html/2404.02405 | **R/F** Time-aligned coordinates |
| TriDet / DisTime / BMA | see 091 §4 | **R** Distribution/OT neighbors — **E091 aux + H103 primary W1 both failed under hull** |

**Open gap:** TAL often **decodes spans continuously** (center/width or distances). VrdONE uses **mask→hard hull**. That decode gap is **still open** after H103 (loss changed, hull kept).

### C4. Temporal grounding / moment retrieval

| Work | Source | Status |
|---|---|---|
| Moment-DETR / QVHighlights (NeurIPS2021) | https://arxiv.org/abs/2107.09609 · [code](https://github.com/jayleicn/moment_detr) | **A this turn** Direct **span** set prediction + saliency — continuous interval outputs |
| related QD-DETR etc. | not re-fetched | Treat as same cluster: span heads, not mask-hull |

**Open gap:** continuous span decode vs mask-hull — aligns with residual family A.

### C5. Mask / segmentation losses

| Work | Source | Status |
|---|---|---|
| RetinaNet Focal (ICCV2017) | standard | Used natively; **H101/H102 reweight REJECT** |
| Mask2Former-style dice+focal | used in VrdONE | Event-equal dice already; **not** the residual fix via reweight |
| Soft-argmax / expectation landmarks (pose lit.) | classic (e.g. integral pose regression lineage) | **Not tested here** as VidOR interval decode |

**Do not retry:** another focal/Dice reweight or W1-primary+hull.

### C6. Project-internal decode attempts

| Work | Status |
|---|---|
| 078 start–end heads | **REJECT / no short gain** |
| 088 split-hull | **NOT_SUPPORTED** as main remedy |
| 090 unit-mass + moment proposal | Design only; H103 tested **W1 half** with hull — failed; **moment/soft decode half still open** |

---

## Map summary

| Residual family | Lit support | Project status | Next? |
|---|---|---|---|
| Matching identity | Stable-DINO | **Ruled out (H100)** | No |
| Mask loss reweight / OT primary+hull | Focal, TriDet, 091 | **Ruled out (H101–H103, E091)** | No |
| Depth mask agg | deep supervision folklore | **Ruled out (H104)** | No |
| Continuous / soft interval **decode** from scores | ActionFormer, Moment-DETR, soft-argmax | **OPEN — highest ID now** | **Yes → H105** |
| New queries / DN | DN-DETR, RTD-Net | Open, lower priority (confounds capacity) | Later |
| Unfreeze encoder | — | Open, expensive | Later |

**Selection rule for one hyp:** choose the open cell with (i) residual consistency, (ii) lit precedent for the *decode* gap, (iii) falsifiable X control, (iv) not on the do-not-retry list.
