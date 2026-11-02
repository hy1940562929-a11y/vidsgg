# Residual synthesis R5 (after H108) — 2026-10-21

Grace/supervisor. Builds on [R4](../cause_arch_residual_synthesis_20261021_r4/). Continuous loop policy: do not pause after REJECT.

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H108** | Train-only DN short queries (parallel DN forward; eval 9 content) | REJECT; eta(T,N)=0; T−N +2/0/−2 | Extra short-biased DN supervision does **not** teach content queries to form residual short intervals under native hull. |

Still closed: E091, E097–099, H100–H107 (loss/decode/assignment/depth-agg/x-adapter/f-adapter).

### Do-not-retry (additions)

19. Parallel or concatenated **train-only DN** from (class,span) under native content path.  
20. Another DN noise schedule / DN slot-count micro-variant as sole lever.

## 2. What H105–H108 jointly imply

- Soft decode, mid-feature temporal adapters (`x`,`f`), and DN query binding all fail to unlock residual334.  
- 089 empty bank remains; failure is still **interval formation under long coexistence**.  
- 080/085 did **not** establish semantic long-suppress, but **pair-level coexistence conditioning of queries** (not thin f-adapter) is still untested — matches the locked problem statement without reopening closed families.

## 3. Open families (re-ranked)

| Rank | Family | Status |
|---|---|---|
| **1** | **D. Coexistence-length FiLM on content query embeds** | **Open — select H109** |
| 2 | E. Duration-biased matching costs | Weak ID (G often 1) |
| 3 | B. Encoder/backbone unfreeze | Cache-hostile |
| 4 | Permanent eval-time extra short queries | Capacity confound; later |
| — | DN, adapters-on-x/f, decode, loss, depth, assignment | **Closed** |

## 4. Deliverable

Root cause still **not** validated.
