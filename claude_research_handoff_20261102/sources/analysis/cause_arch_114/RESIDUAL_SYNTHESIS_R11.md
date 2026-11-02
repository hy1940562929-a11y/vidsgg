# Residual synthesis R11 (after H113) — 2026-10-21

## Closed this round
**H113 REJECT / DIRECTION_WRONG_OR_HARMFUL** — Moment-DETR cxw span decode+L1: eta≈−0.068; T≪N all seeds. Close span-as-eval-decode family.

Still closed: E091/097–099, H100–H113(+H112-SCALE). No underpowered+ pending scale.

## Failure localization
089 empty interval bank; **mask→hull readout must stay**. Soft decode (H105), adapters (H106/107), DN (H108), FiLM (H109), duration match bias (H110), query bank (H111), oversample (H112), **span decode (H113)** all insufficient/harmful.

## Lit-driven open gap
DETR / Moment-DETR / DINO matching uses **class + L1 + GIoU** on boxes/spans. VrdONE `bipartite_match` has only **cost_class + cost_mask + cost_dice** — **no temporal GIoU/tIoU localization cost**, despite `gt_segs` already threaded for fuzzy loss. H110 only reweighted existing costs by duration; it did **not** add a DETR-style geometric localization term. ActionFormer / TadTR / RTD-Net treat tIoU as the primary localization signal for short actions.

## Select H114
Family: **temporal GIoU matching cost** (DETR box GIoU × Moment-DETR span GIoU × TAL tIoU lineage), keep native mask/CE loss coefficients and **hull decode**.
