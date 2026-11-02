# Residual synthesis R4 (after H107) — 2026-10-21

Grace/supervisor. Builds on [R3](../cause_arch_residual_synthesis_20261021_r3/).

## 1. Newly closed

| ID | Claim | Verdict | Implication |
|---|---|---|---|
| **H107** | Residual temporal Conv adapter on fine mask features **`f`(B,256,512)** | REJECT; T≪N all seeds; eta≈−0.019; X→0 | Light re-encoding of **frozen mask features** is not the residual fix (and can hurt). Flip-`f` destroys masks → `f` time structure matters, but residual adapter≠remedy. |

Still closed: E091, E097–099, H100–H106 (loss / decode / assignment / depth-agg / **adapter-on-x**).

### Do-not-retry (additions)

16. Another **adapter-on-f** / deeper conv-on-f / FiLM-as-thin-adapter-on-cached-`f` as sole lever.  
17. Any **simple temporal Conv residual on cached tensors** (`x` or `f`) without a **new binding/query** mechanism.  
18. Claiming “representation irreparable on cache” without testing **query / DN** family.

## 2. What H106+H107 jointly imply

- 089: residual shorts mostly **lack a matching interval in the 9-query bank** (334/347 predicate-OK still fail; U≤~13%).  
- H105: re-reading the same masks softly does not create that interval.  
- H106/H107: adapting cached **`x` or `f`** with residual temporal conv does **not** stably create short support under native loss+hull (H107 even regresses).  
- Therefore the open lever with remaining ID power is **how queries are supervised / bound to short GTs during training**, not another mid-feature adapter.

Train still G≤9 always (often G=1); duration-biased **matching costs** alone have weak ID when G=1.

## 3. Open families (re-ranked)

| Rank | Family | Status |
|---|---|---|
| **1** | **C. DN / short-biased denoising queries (train-only)** | **Open — select H108** |
| 2 | D. Coexistence FiLM / pair-level gating (not thin f-adapter) | Open; prior 080/085 semantic long-suppress failed |
| 3 | E. Duration-biased matching costs | Weak ID (G often 1) |
| 4 | B. Full encoder unfreeze | Open but cache-hostile |
| — | B′/`f` adapters, decode, loss reweight, depth agg, assignment | **Closed** |

## 4. Deliverable

Root cause still **not** validated. Architecture still conditional.

## Post-R4 closure (E108)

- **H108 REJECT** (E108-DN-SHORT): residual334 T−N +2/0/−2; eta(T,N)=0; T−X not all>0. Do-not-retry: parallel train-only DN from (class,span) encoder under native content path/hull.
