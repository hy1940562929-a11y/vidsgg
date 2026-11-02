# Residual synthesis R2 (after H105) — 2026-10-21

Grace/supervisor. Supersedes open-family ranking in [R1](../cause_arch_residual_synthesis_20261021/RESIDUAL_SYNTHESIS.md) for decision-making; R1 inventory remains valid.

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H105** | Soft mass-quantile decode vs hull (shared native weights) | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE`; **T≡X** on 334 | **Decode-isolation family closed.** Soft mass is not a better interval than hull; reverse control matched T → no specific soft-structure benefit. C0 soft 27 < hull 29. |

Still closed (do not reopen): E091, E097–E099, H100–H104 — see R1 §2.

### Do-not-retry (updated)

All R1 items, plus:

11. **Soft mass / quantile / expectation-only decode** as sole lever (H105).  
12. Any “same weights, only change hull↔soft decode” variant (thresholds, other quantiles) without new representation evidence.  
13. Rapid decode/loss micro-sweeps.

## 2. What 334 + H105 jointly imply

- 089: most residual shorts **lack a matching interval in the native candidate bank**.  
- H105: **re-reading the same masks softly does not create that interval**.  
- Therefore the residual is unlikely to be “hull threw away a good soft peak.” More likely: **mask peaks / features never form the short support** under the current frozen-input + predictor-only recipe.

C0 residual334 time hits remain ~29/334.

## 3. Open families (re-ranked)

| Rank | Family | Why open | ID power | Feasibility on cached AutoDL recipe |
|---|---|---|---|---|
| **1** | **B′. Temporal re-encoding of frozen decoder inputs** | Encoder/FPN bypassed (cached `x`); predictor-only may be insufficient to form short peaks | High if T>N and T>X(time-rev inputs) | **High** — adapter on `x` |
| 2 | C. New queries / DN short queries | Lit DN-DETR/RTD; 089 bank empty | Medium–high; confounds capacity | Medium |
| 3 | D. Coexistence FiLM / gating | Long-suppress semantic failed; feature gate untested | Medium | Medium |
| 4 | E. Duration-biased **matching costs** (not focal) | Distinct from H100/H101 | Medium; looks “tweakish” | High |
| 5 | B. Full encoder unfreeze | Strongest representation claim | High | **Low** (cache bypasses encoder) |

**Selected for one hyp:** Family **B′** → H106 temporal adapter on cached `x`.

## 4. Hard deliverable

Root cause **still not validated**. Architecture still conditional.
