# Literature map R2 — remaining open families only (2026-10-21)

Closed clusters (matching identity, mask reweight/OT+hull, depth agg, soft decode) omitted; see R1 map.

## Open cluster O1 — Temporal adapters / re-encoding mid features

| Work | Source | Relevance |
|---|---|---|
| ActionFormer (ECCV2022) | https://arxiv.org/abs/2202.07925 | Multiscale temporal modeling before localize — motivates **adapting temporal features**, not only mask loss/decode |
| Temporal convolutional networks for action (Lea et al. / related TCN line) | classic TAL | Local temporal conv on feature sequences |
| Transformer adapters / residual adapters (NLP→vision adapters) | general | Extra capacity with residual path; use **time-rev X** to separate capacity from structure |

**Project:** VrdONE train uses **cached decoder inputs** `(B,512,64)`; backbone frozen/bypassed. Untested: trainable temporal adapter on `x` before predictor.

## Open cluster O2 — Extra / denoising queries

| Work | Source | Status |
|---|---|---|
| DN-DETR | https://arxiv.org/html/2203.01305 | Open; not chosen this round (capacity confound) |
| RTD-Net relaxed matching | https://arxiv.org/pdf/2102.01894 | Open |

## Open cluster O3 — Conditioning / gating

Coexistence gating: no clean positive project evidence (080/085 failed semantic long-suppress). Lower priority than O1.

## Selection

**H106** maps to **O1**: residual after H105 points to missing short peaks in features→mask, not decode.
