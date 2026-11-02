# Residual synthesis R3 (after H106) — 2026-10-21

Grace/supervisor. Builds on [R2](../cause_arch_residual_synthesis_20261021_r2/).

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H106** | Residual temporal adapter on cached **`x`** (B,512,64) | REJECT; T>X(time-rev) but not stably >N; eta≈0.003 | Light re-encoding of **coarse query stream** is not the residual fix. Time structure matters (T>X) but predictor-only already matches adapter(x). |

Still closed: E091, E097–099, H100–H105 (incl. decode isolation).

### Do-not-retry (additions)

14. Another **adapter-on-x** / deeper conv-on-x variant as sole lever.  
15. Claiming “frozen features irreparable” without testing the **mask-feature stream `f`**.

## 2. Architectural fact unlocked this round

Native forward: `predictor(x, f, key_mask, valid)` with  
`pred_masks = einsum(mask_embed, mask_features=f)`.  
Cached shapes: **`x`(B,512,64)** vs **`f`(B,256,512)**.  
H106 only touched `x`. **Fine-grid mask features `f` were never adapted** under the takeover experiments. Short (<3s) events occupy few of 512 bins — adapting `f` is the natural remaining representation lever that is **not** H106.

Train still has G≤9 always (no GT dropped by Hungarian capacity).

## 3. Open families (re-ranked)

| Rank | Family | Status |
|---|---|---|
| **1** | **B″. Adapt / refine mask features `f` (T=512)** | **Open — select H107** |
| 2 | C. DN / extra short queries | Open; heavier |
| 3 | D. Coexistence FiLM | Open |
| 4 | E. Duration-biased matching | Weak (G≤9 always; often G=1) |
| — | B′ adapter-on-x | **Closed H106** |
| — | Decode / loss reweight / depth agg / assignment | **Closed** |

## 4. Deliverable

Root cause still **not** validated.
