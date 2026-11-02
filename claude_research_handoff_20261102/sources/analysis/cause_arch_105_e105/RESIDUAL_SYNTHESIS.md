# Residual synthesis after H100–H104 (2026-10-21)

Grace/supervisor takeover (Codex quota exhausted). **Purpose:** stop rapid hyp chaining; lock what is ruled out, what the 334 residual still means, and what must not be retried—before opening one new hypothesis.

**Hard deliverable status:** (1) experimentally validated root cause — **NOT MET**; (2) on-cause architecture — **NOT MET**. Failure location is supported; unique cause is not.

---

## 1. Supported failure localization (unchanged from 090)

| Claim | Status | Pointers |
|---|---|---|
| Long-coexistence short relations fail at **interval formation** after native score→hull | **SUPPORTED** | [090](../mechanism_synthesis_090/VERDICT_UPDATE.md); [089](../mech_query_score_089/) 334/347 predicate-OK still fail |
| Input geometry / grid expressibility is not the main excuse on the locked 400 | **SUPPORTED exclusion** | [086](../mechanism_verdict_086/) 400/400 ideal-grid feasible |
| Scores carry some localization vs length-matched null | **WEAK positive** | [088](../mech_interval_form_088/) beat null; not enough for +5pp hull-split |

**334 residual signature (operational):** among predicate-top1–qualified shorts, native one-query hull still misses tIoU≥0.5 match for the large majority. C0 time hits on residual334 stay ~29/334 across E100–E104. This is a **shape/interval** failure under long grids, not primarily a “wrong class query” failure (089: 0 effective swaps).

---

## 2. Ruled-out / stopped recipes (do not reopen)

| ID | Mechanism tested | Verdict | Evidence |
|---|---|---|---|
| E091 / H091 | Temporal geometry **aux** (CDF/W1) on binary mask | `NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE` | `analysis/cause_arch_095_e091/VERDICT.md` |
| E097 / H097 | Local state contrast under **old F/H capacity** | `REJECT / SCOPE_CAPACITY_CONFLICT` | oracle ≤149/334; `cause_arch_097_e097/` |
| E098 / H098 | Severe temporal **under-reading** majority | `REJECT / UNDERREADING_NOT_ESTABLISHED` | 47/334≈14% low-read; `cause_arch_098_e098/` |
| E099 / H099 | Class↔mask **gradient interference** | `REJECT / INTERFERENCE_NOT_ESTABLISHED` | C0 short gain CI crosses 0; `cause_arch_099_e099/` |
| E100 / H100 | Cross-layer Hungarian **assignment identity** | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE` | T−N +6/0/−2; `cause_arch_100_e100/` |
| E101 / H101 | Focal **positive-mass** equalization | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE` | T−N **−4/−4/−7** (T worse); `cause_arch_101_e101/` |
| E102 / H102 | Focal **exterior-negative** upweight | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE` | T−N +6/+1/+7 but eta≈0.016; unstable vs X; `cause_arch_102_e102/` |
| E103 / H103 | **Unit-mass W1 replaces** binary focal+Dice (hull kept) | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE` | T−N **−4/−13/−3**; X→0 hits; `cause_arch_103_e103/` |
| E104 / H104 | Decoder **depth mean-mask aggregation** | `REJECT / EFFECT_TOO_SMALL_OR_UNSTABLE` | T−N +6/+2/−4; eta≈0.0045; `cause_arch_104_e104/` |
| 087 | Fusion harms short evidence | not supported | `mech_minifalsify_087/` |
| 088 | Split-hull as **main** remedy | not supported at +5pp gate | `mech_interval_form_088/` |
| 089 | Wrong-query selection as main cause | not established | `mech_query_score_089/` |
| 080/085 | Long-relation **semantic suppression** | not established | `long_short_context_085/` |
| 078 | Start–end / joint span readout heads | no short improvement | `span_readout_078/REPORT.md` |

### Explicit “do not try again” list

1. Another **focal/Dice point-weight** variant (pos/neg/duration) as sole lever.  
2. Another **primary mask loss swap** to OT/W1/unit-mass **while keeping sigmoid>0.5 hull**.  
3. **Path-joint Hungarian** / Stable-DINO-style assignment identity (H100).  
4. **Equal-weight layer mask fusion** as sole lever (H104).  
5. **Geometry aux on top of binary** (E091).  
6. **Class-grad projection** C0 recipe (E099).  
7. **Attention under-reading volume** majority claim (E098).  
8. **Old F/H single-slot capacity** majority-recovery claim (E097).  
9. **078-style** independent/joint start–end heads as the next “obvious” fix.  
10. Silent membership/threshold loosening to force VERIFIED.

---

## 3. What remains open (families, not a queue)

After excluding the above, 090’s residual splits into still-open families:

| Family | Why still open | What would identify it |
|---|---|---|
| **A. Soft score → interval decode** | 088: scores > null; 088/H102/H103: hard hull / loss changes didn’t unlock; **soft mass / expectation / quantile decode of existing masks not cleanly isolated** | Same weights, change only decode; X=time-scrambled soft decode |
| **B. Representation before mask head** | Encoder/FPN frozen in all recent runs; visual short evidence may be weak | Unfreeze / swap temporal backbone (costly; last resort) |
| **C. Query–event binding structure** | 089 rules out naive swap among existing candidates; not all binding forms | New candidates / dynamic queries (overlaps RTD-Net lit; high novelty risk) |
| **D. Coexistence conditioning** | Long-suppress semantic claim failed; **feature-level gating** untested | Gate mask by coexist duration (easy to confound) |
| **E. Train sampling / batch composition** | Not tested as cause | Duration-balanced sampling (methodish; weak “cause”) |

**Strongest open with high identification power now:** Family **A**—because H101–H103 show that changing **how we train the mask** under the **same hull** fails or barely moves 334, while 088 shows **information exists** in scores relative to null. The untested clean lever is: **keep native training, change only how soft scores become an interval.**

---

## 4. Implications for the next single hypothesis

- Prefer **decode isolation** over another loss tweak.  
- Prefer matched weights across N/T/X so eta cannot be attributed to optimization path differences.  
- Keep E100-A1 mechanism-dev membership; do not expand silently.  
- VERIFIED would mean **scoped readout effect**, not full historical root cause of all VidOR shorts.

Next: [LITERATURE_MAP.md](LITERATURE_MAP.md) → one hyp in [../cause_arch_105/](../cause_arch_105/).
