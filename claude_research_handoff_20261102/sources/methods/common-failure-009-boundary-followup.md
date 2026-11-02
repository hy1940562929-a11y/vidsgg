# Conditional boundary follow-up after actual old32 outcomes

Source: Innovation Agent `/root/innovation_common_failure`, read-only analysis preserved by Commander. **UNVERIFIED method proposal. No training performed or authorized by this document.** The registered R4 expanded64 evaluation remains the only active model comparison; do not select a different winner from expanded outcomes.

## Supported local mechanism and limits

For three restored original C cases, the exact same input pair/query/predicate retains identical score and rank while its span changes. Min-vIoU changes are0.49652→0.61049,0.49882→0.51096,0.48937→0.68321. These local interventions show that decoder support was sufficient to change those matching outcomes; they do not establish historical-context interference or broad short-relation recovery.

The cases are from only two videos, all3–10seconds, and all original overlaps lie close to the official0.5 threshold. The first mask was contiguous; the third contained weaker/discontinuous tail support. Therefore discontinuity is not a common necessary cause. Continuous IoU changes must accompany binary hit counts.

The lost adult–watch–horse candidate shrinks from[106,311) to[106,299) against GT[11,316), lowering min-vIoU0.50555→0.47819 at the same rank and classification score. Its mask mean increases0.95303→0.97084. Thus higher mask mean need not mean more accurate temporal extent. The additional D recovery has unchanged span/IoU and zero-based rank101→97, a budget-competition effect.

## Target capability

Target input-feasible candidates where the correct predicate exists but predicted temporal support is too wide, too narrow or shifted. Duration and repeated GT records are reporting strata, not sufficient mechanistic conditions. The full-C66.32% input-infeasible group is outside a boundary head's reachable scope. The old32 feasible-C180 no-matching-span cases remain a mixed candidate pool, not180 guaranteed recoveries;120 rank/assignment and26 Top6 failures require different capabilities.

## Minimal trainable candidate

`BOUNDARY-RESIDUAL-M1 / UNVERIFIED / TRAINING_DATA_REQUIRED`

Freeze VrdONE; a shared small head receives query state and frozen local features around the current start/end, and outputs two signed residuals:

\[
(\delta_s,\delta_e)=f_\theta(h_q,x_s,x_e),\qquad s'=s+\delta_s,\quad e'=e+\delta_e.
\]

- Zero-initialize the last layer so initial behavior equals baseline.
- Preserve original temporal grid initially; a later frame-resolution change needs its own ablation.
- Preserve classification, scores, Top6, input trajectories and candidate count. No simultaneous NMS or quality reranking.
- Project to legal support and original minimum length; log projection/clipping frequency.
- Train using genuine train-only GT and frozen-model matches, including originally correct, too-long and too-short candidates. Never train only on validation C or only shrink targets.
- A D→32→2 MLP has32D+98 parameters. D, usable features, train proposal compatibility and runtime are not yet verified.

This allows retaining, shrinking or expanding each boundary separately. It does not solve missing semantics, absent trajectories, unobserved subsecond signals or multiple disjoint events represented by one interval.

## Necessary controls and stop outcomes

Use equal train data, video-disjoint train/development split and comparable fitting budgets:

1. Original0.5 and frozen R4=0.75.
2. One learned global threshold.
3. Query-conditioned single threshold.
4. Independent signed residual head.
5. Same residual head without local boundary features.

If a simple conditional threshold matches the residual head, record `SIMPLER_BASELINE_SUFFICIENT`. If metrics rise without improved target boundary errors and sample outcomes, record `MECHANISM_UNSUPPORTED`. Report temporal errors separately from geometric vIoU, because noisy trajectory boxes can cause these measures to disagree.

Unchanged or harmful effects and all originally-correct losses must remain visible. Do not keep changing the target definition to fit positive examples.

## Novelty and progression

Boundary regression, mask quality scoring and variable temporal extent are established ideas; this small head is a diagnostic mechanism candidate with **NOVELTY_UNCERTAIN / HIGH_RISK**, not a confirmed CVPR contribution. A method claim needs evidence that simpler calibration is insufficient and a fresh primary-source comparison with closest work. It has not been adopted as the main method.

Direction's subsequent primary-source check found direct overlap: TCANet(CVPR2021) and CBR(BMVC2017) already use separate signed boundary offsets and refinement of existing proposals; BIG(CVPR2022) performs triplet-conditioned temporal grounding. See `literature/common-failure-009-boundary-prior-art.md`. The basic residual operation itself is therefore established, not merely uncertain. Any distinct contribution beyond adapting this baseline remains unverified.

The registered64 results subsequently arrived: R4 gains11/loses6, including three short watch gains and two short watch losses; both metric delta confidence intervals cross zero. Full835 then completed with78 gains/108 losses,64 fixed-C recoveries and net−30. Uniform0.75 narrowing did not improve overall performance and must not be promoted. The full186-case audit records64 gains crossing the same-candidate geometry criterion (58C+6D),14 compatible-candidate ranking/assignment gains,103 retained but incompatible loss candidates and5 removed loss candidates. These outcomes support testing selective support correction but do not establish that this untrained residual head would help. Existing inspected AutoDL folders lack training features/labels; a user query about other training paths is pending. No claim that the entire machine was exhaustively searched.
