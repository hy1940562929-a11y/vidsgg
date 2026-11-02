# 085 frozen context sensitivity protocol

Status: code prepared; actual results require a complete run and review. No training.

Observation sought: does changing annotation-conditioned long-predicate context outside an unchanged short relation alter that short relation's frozen-model score? Annotation-negative windows are not certified semantic negatives. A score response does not establish semantic deletion or long-relation competition.

## Cohort and fixed selection

Use strict27 GT entity features and the 073 `setup`/`video_input` construction. Short GT duration is strictly below 3 seconds, contained in one pair support of at least 10 seconds. A different predicate lasting at least 3 seconds must cover at least 50% of the short GT duration. Pair features above 2048 tokens and events with no sampled target or unchanged boundary tokens are excluded with reasons. Selection uses GT and input features only.

All windows contain exactly 8 contiguous stride-4 tokens. Treat each token as its complete 4-frame cell for support and exclusion. The protected region is the target interval expanded by 0.5 seconds on either side; every token whose cell intersects it remains bitwise unchanged.

L is wholly within the covering long annotation, outside protection and outside every annotation of the short target predicate. N and donor are outside protection, every >=3-second annotation for this ordered pair (all predicates), and every annotation of the target predicate. L/N/donor are pairwise disjoint. N and L must be on the same side of the target; their nearest-edge distances from the target differ by at most 2 seconds. This bound was revised from 1 second before any outputs were inspected, after input-only geometry census found zero feasible pairs under 1 second (8 stride-4 tokens span about 1.07 seconds at 30 fps). No later relaxation is allowed.

Enumerate all L/N window pairs; sort by distance difference, then maximum distance, L start, N start. For each pair enumerate eligible donors by distance to N, then donor start; take the first with positive ROI replacement RMS in each arm and L/N RMS ratio in [0.5,2]. No output-dependent selection or removal after failed manipulation checks. Absolute RMS and actual distances are recorded. One covering long annotation is selected lexically among feasible annotations.

Deduplicate by (video, subject tid, object tid, predicate, begin, end); where multiple pair segments are feasible choose lowest pair index. Rank stable SHA256 case IDs and take up to 64 events, at most 4 per video. Unselected feasible events and exclusion reasons are retained.

## Four arms and invariants

Baseline; exact cloned-input noop; editL; editN. Both edits use the identical same-pair donor. Replace only channels 0:2048 (ROI), preserving last 21 geometry channels. Length, position, timestamps, checkpoint, query count and all protected inputs remain unchanged. Native preprocessing and `_mask_vrd` run at microbatch 1. No crop, temporal shift, class reranking or parameter update.

Baseline picks the query with highest target-predicate softmax probability, then locks it across all arms. Independently lock the baseline highest long-predicate query. Native decoder is sigmoid >0.5, first positive frame to last positive frame +1, minimum physical length 5; disconnected positives remain one envelope.

## Measurements and decision rule

Per locked target query: mean target mask logit/probability, target-minus-outside logit margin, target class probability, decoded temporal IoU, and hit@0.5 gain/loss. Outside for the margin is only unchanged cells within the protection band and fully outside target; no edited L/N cells enter that reference. GT-time any-query best tIoU is separately labeled an oracle, never confused with the locked-query result or deployed recall/mAP.

Manipulation check: locked long-query mean mask probability/logit on original L window and long-class probability changes. Failure stays in the cohort; it cannot establish that long semantic context was removed.

Primary endpoint: video-equal mean of target mask probability(editL)-probability(editN). Support requires >=10 videos, effect >=0.01 (probability score units, not recall percentage points), fixed-seed 10,000 video bootstrap 95% lower bound >0, and positive video-equal editL-baseline effect. This is sensitivity evidence conditional on the intervention, not proof of semantics or the unique mechanism. Report per-video effects and all event records to permit auditing; any case illustrations are only candidates.

## Invocation and audit

Run `prepare` first, inspect feasibility, then `infer` using that sealed plan. Use absolute paths. Example (private authorized environment):

```sh
python context_probe.py prepare --helper /path/to/073/code_v2 --out /private/085/prepare
python context_probe.py infer --helper /path/to/073/code_v2 --plan /private/085/prepare/selection.json --out /private/085/infer
```

Inference wall budget is 900 seconds including setup/hash loading; no training, checkpoint/environment/data edits, or shared outputs. The runner must use its own approved GPU allocation. Source/config/checkpoint SHA and original input fingerprints are checked before inference. Every arm stores raw masks/class logits/binary masks, input/output hashes, invariance audit, per-event metrics and complete summary with raw file hashes. Noop must be exactly equal for every output tensor. Model state hash must remain unchanged. Any failure creates failure.json; partial outputs never imply completion. A complete zero-event run remains uninformative.
