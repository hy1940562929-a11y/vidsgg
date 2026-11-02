# Residual synthesis R7 (after H110) — 2026-10-21

Continuous loop. Builds on [R6](../cause_arch_residual_synthesis_20261021_r6/).

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H110** | Duration-biased Hungarian costs | REJECT; all T>N but eta≈0.018; T−X unstable | Matching-cost short preference is not the residual root cause. |

Do-not-retry: duration-biased matching-cost-only variants.

## 2. Pattern note

H109–H110 show small positive T−N (eta≪0.20). 089 empty bank (U≤~13% with 9 queries) remains the strongest structural clue. Next: **permanent extra queries at eval** with short vs long prior (capacity-matched across N/T/X at Q=12).

## 3. Open families

| Rank | Family | Status |
|---|---|---|
| **1** | **Permanent eval-time extra short-prior queries** | **Select H111** |
| 2 | Encoder/backbone unfreeze | Cache-hostile last resort |
| 3 | Train sampling / short oversampling | Methodish |
| — | matching costs, FiLM, DN, adapters, decode, loss | **Closed** |
