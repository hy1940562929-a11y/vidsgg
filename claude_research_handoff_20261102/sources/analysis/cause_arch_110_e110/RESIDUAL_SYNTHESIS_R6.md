# Residual synthesis R6 (after H109) — 2026-10-21

Grace/supervisor continuous loop. Builds on [R5](../cause_arch_residual_synthesis_20261021_r5/).

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H109** | Coexist-length FiLM on content queries | REJECT; all T≥N but eta≈0.012; T−X unstable | Pair-length query conditioning is not a sufficient root-cause lever. |

Still closed: E091, E097–099, H100–H108.

### Do-not-retry (additions)

21. Coexist-length **FiLM on query embeds** (or trivial scalar→γβ variants).  
22. Claiming long-coexistence is solved by **query-side length FiLM alone**.

## 2. Re-rank

H109’s weak but directionally positive T−N suggests coexistence structure matters slightly, but not via query FiLM. On coexistence pairs **G≥2** (short+long same pair), **Hungarian cost duration bias** can change which GT a query binds to during training — distinct from H100 (cross-layer identity) and H101/H102 (focal loss reweight). Prior “weak ID” concern (G=1 majority) understates the residual mechanism-dev pairs where G>1 is exactly the coexistence case.

## 3. Open families

| Rank | Family | Status |
|---|---|---|
| **1** | **E. Duration-biased matching costs** | **Select H110** |
| 2 | Permanent eval-time extra short queries | Capacity confound; next if H110 fails |
| 3 | Encoder unfreeze | Cache-hostile |
| — | FiLM/DN/adapters/decode/loss/depth/assignment | **Closed** |
