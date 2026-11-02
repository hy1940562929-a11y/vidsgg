# Residual synthesis R12 (after H114) — 2026-10-21

## Closed / open
H114 REJECT underpowered+ (eta≈0.034, all T>N, TX weak) — **scale-eligible**, not falsified.
Still closed (falsified): H100–H113, H112-SCALE. H113 closed span-as-decode.

## Failure localization
089 empty bank; hull stays. Match geometry (H114) helps residual334 vs N but not vs X and far below η=0.20.

## Lit-driven open gap
ActionFormer (ECCV’22) / TriDet / Temporal Max Pooling pyramids: short actions need **multi-scale temporal features**. VrdONE feeds a single-scale cached `f` into the decoder. Untested: **ActionFormer-style temporal FPN on `f`** (1×/2×/4× pool → upsample → fuse) before native predictor; eval stays hull.

## Select H115
Family: multi-scale temporal feature pyramid on `f` (ActionFormer / TAL FPN lineage). Keep native match+loss+hull.
