# Literature map R4 — open families after H107

Closed: matching identity, mask reweight/OT+hull, depth agg, soft decode, adapter-on-x, adapter-on-f.

## O2 — Denoising / extra queries (selected)

| Work | Source | Relevance |
|---|---|---|
| DN-DETR (CVPR2022) | https://arxiv.org/html/2203.01305 | Train-only noisy GT queries + exclusive match; **eval keeps content queries** — isolates binding/supervision from eval capacity |
| RTD-Net (ICCV2021) | https://arxiv.org/pdf/2102.01894 | Relaxed temporal matching / query structure for action |

**Project fit:** 089 empty bank + H106/H107 feature-adapter failure → test whether **short-biased DN supervision** teaches content queries to form short intervals. X = same DN slots with **time-shuffled** GT spans (capacity+aux matched; temporal DN signal destroyed).

## O3 — Conditioning (deferred)

Coexistence FiLM / gating — still open; not chosen (080/085 negative; easier to confound with f-side edits).

## O4 — Matching-cost duration bias (deferred)

Weak when G=1 (majority of train pairs).
