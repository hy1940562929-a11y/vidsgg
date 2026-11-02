# 009 minimal candidate rationale

Source: read-only Innovation Agent `/root/innovation_common_failure`, preserved by Commander. Status: four conventional mechanism probes ready for controlled experiments; full novel method NOT_READY_FOR_INNOVATION, NOVELTY_UNCERTAIN. No measured gain was supplied by the agent.

Exact frozen formulas and primary selection protocol are in `tasks/VIDOR-COMMON-FAILURE-009/frozen_candidates.md` and `reviews/common-failure-009-protocol.md`. The four candidates are independent single-factor changes on the complete raw candidate pool, not only savedTop200. No combinations, parameter search or GT-dependent inference.

## R1: geometric class-score aggregation

Replace mean(subject,predicate,object probability) by their product's cube root. Hypothesis: high entity confidence can compensate for a weak predicate under arithmetic averaging, displacing more reliable relations. No temporal interval or candidate is added. This is the simplest competing ranking explanation, not temporal-localization improvement. Failure: no net useful C recovery or harm to existing hits/main metrics.

## R2: temporal mask confidence in ranking

Multiply original score by the mean sigmoid mask probability over all sampled positions between first and last positives, including internal negative positions. Keep original intervals. Hypothesis: an over-wide or discontinuous span can have lower temporal confidence than a correctly localized lower-ranked candidate. Counterexample: an incorrect long mask can be confidently wrong. Compare this quality proxy with actual matching quality, and examine recovered/lost candidates' values; a total metric increase alone does not validate its semantics.

This is conventional: [Mask2Former official inference](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py) already combines class scores with average predicted mask probability. This temporal-span adaptation is not a claim of first quality scoring.

## R3: same-pair/predicate temporal NMS

Greedy temporal NMS at tIoU>0.7 under original score, before cap200, within the same ordered sid/oid/predicate only. Hypothesis: redundant queries for one temporal occurrence consume the ranking budget, hiding candidates for other events. Distinct nonoverlapping occurrences remain eligible.

Return suppression parent/child IDs, overlaps, duplicate counts, new entrants, and whether new hits cover a different temporal event. Highest-scoring broad spans may suppress more accurate intervals; preserve these losses. NMS changes candidate competition, so improvement cannot be called pure boundary refinement. It is an existing technique; [Soft-NMS](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf) illustrates established overlap-suppression tradeoffs.

## R4: fixed higher mask threshold

Threshold0.75, same first/last-positive decoding, min5, score and budget. No fallback or splitting. Hypothesis: moderate-confidence temporal tails enlarge spans around a stronger short event. Distinguish new geometric compatibility from rank gains caused by filtering other predictions. Counterexamples: uniformly confident wrong spans or weak true short events. No further threshold sweep if it fails.

## Conditional trained quality head

Not authorized for launch by this document; requires verified train annotations and proposal features first. Freeze VrdONE and add a12→32→1ReLU/sigmoid MLP (449parameters). Twelve inputs: subject/predicate/object probabilities, background probability, mask span mean/std/min/max, outside-span mean, outside-support indicator, span/sequence length, positive-position fraction. Empty outside support gets mean0 plus indicator.

Target is maximum joint subject/object vIoU against training GT with the same class triplet, or0 when absent. Inference score is original score times predicted quality, with no GT. This estimates joint matching quality, not exclusively temporal quality. Similarity to [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html) creates high novelty risk.

Suggested bounded training if inputs become available: train-only videos, internal video-disjoint development, fixed Adam lr1e-3, wd1e-4, batch512,10epochs,seed42. Compare original score/R1/R2 and a same-budget ordinary-head control. Training on GT-only proposal geometry does not automatically transfer to detected proposal geometry; domain mismatch must be disclosed. No validation-label training or validation-selected epoch.

## Interpretation and retention

FixedC recovery is newVrdONE_hit AND NOT baselineVrdONE_hit AND NOT baselineBIG_hit. The008 182/138/28 VrdONE-miss buckets cannot be relabeled as C-only. Keep all1,209GT, original hit losses, D recovery and feature strata. Every attempt must have exact config/hash, predictions, candidate identities and case-level changes. Negative evidence is retained. A conventional baseline that improves is an empirical diagnostic contribution, not proof of CVPR-level novelty.
