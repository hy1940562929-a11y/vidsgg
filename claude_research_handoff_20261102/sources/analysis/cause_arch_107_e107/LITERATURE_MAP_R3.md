# Literature map R3 — open families after H106

## O1′ — Fine temporal mask features (selected)

Mask classification via dot-product with high-res mask features is standard (MaskFormer/Mask2Former lineage; VrdONE `einsum(mask_embed, f)`). TAL multi-scale work (ActionFormer) stresses **resolution** for short actions. H106 adapted only coarse `x`; **`f` at T=512 remains the open, on-architecture lever**.

## O2 — DN queries (deferred)

DN-DETR https://arxiv.org/html/2203.01305 — still open; not chosen (invasive vs clear `f` gap).

## O3 — Conditioning (deferred)

Coexistence FiLM — no positive project evidence yet.
