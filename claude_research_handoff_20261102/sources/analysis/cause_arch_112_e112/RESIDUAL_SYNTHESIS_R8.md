# Residual synthesis R8 (after H111) — 2026-10-21

## Newly closed
H111 permanent short-prior query bank REJECT (eta≈−0.002).

Do-not-retry: Q-expand + short/long prior alone.

## Open → select
**Train sampling / short+long coexistence oversampling** (R1 family E). Untested. High ID: if residual is starvation of coexistence-short gradient under uniform cache sampling, T (upsample coexistence-short batches) should beat N and X (upsample long-only).

Remaining after this: encoder unfreeze (cache-hostile) → near exhaustion under predictor-only recipe.
