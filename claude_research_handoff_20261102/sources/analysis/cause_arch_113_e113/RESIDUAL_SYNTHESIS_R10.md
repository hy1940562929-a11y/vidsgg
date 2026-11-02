# Residual synthesis R10 (after H112-SCALE) — 2026-10-21

## Closed this round
**H112-SCALE REJECT / SCALE_FALSIFIED** — short oversample sole lever fails at 5×2000 (eta≈0.001; T often <X). Permanently close oversample-only.

Still closed: E091/097–099, H100–H111, H112(+scale). Underpowered small positives that **survived scale gate**: none currently (H109/H110 still eligible for scale but weaker T−X than E112 had; prefer lit jump).

## Failure localization (unchanged)
089 empty interval bank; mask→hull readout. Soft decode (H105), feature adapters (H106/107), DN (H108), FiLM (H109), match-cost (H110), query-bank (H111), sampling (H112) all insufficient.

## Lit-driven open gap
TAL / moment retrieval often **regresses continuous spans** (center-width) rather than only binarized masks. VrdONE is mask+hull. Untested under takeover: **auxiliary Moment-DETR-style (cx,w) head** jointly trained with native mask, evaluated via span→interval (not soft-mass hull of H105).

## Select H113
Family: continuous span set-prediction aux (Moment-DETR / ActionFormer boundary lineage).
