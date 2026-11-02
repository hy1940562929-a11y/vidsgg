# 078 — Frozen-feature relation interval readout pilot

Status: PREPARED. No outcome is assumed. User authorized bounded AutoDL execution on 2026-10-15; no school server is involved.

## Question
With entity proposals, pair/query identities, predicate scores, encoder features, output count and ranking fixed, does a joint start/end interaction recover short relations more effectively than separable endpoint scores? Does training on same-pair/same-predicate nearby intervals add value beyond equally sized random interval supervision?

## Data and scope
- Reuse frozen 043 training cache: official training videos, 32 videos / 350 pairs / 563 retained GT, 88 batches.
- Adapter development: 8 disjoint training-source videos / 89 pairs / 169 GT, 23 batches.
- Evaluation: already exposed 047 and 048 validation subsets, 64 videos each, 128 videos / 4,739 GT together. They are NOT independent untouched test data.
- Long-coexisting short slice is fixed by evaluation_groups.json: relation <3 seconds and continuous joint visibility >=10 seconds, 331 GT. Baseline expected 6 hits; overall 588 hits at Top100.
- Input features and Hungarian query assignments remain frozen. This tests readout on existing representations, not full end-to-end capacity or missing proposal recovery.

## Arms
1. baseline: original VrdONE output, unchanged.
2. mask: original mask projection continued for 500 steps with original loss, learning rate 1e-5.
3. independent: query-conditioned separable endpoint score.
4. joint: same parameter budget, with cross-endpoint low-rank interaction.
5. joint_hard: joint plus nearby-interval quality-ranking loss.
6. joint_random: joint plus same-length random-position interval ranking, paired counts with hard arm.

All span heads start from cloned parameters and zero residual output. Their shared prior is -(abs(i-s0)+abs(j-e0))/8, where (s0,e0) is the original mask hull. Every valid original selected candidate must have an exactly identical zero-update interval. The independent control uses separable diagonal terms of the same low-rank projections; the joint arm uses cross-endpoint interactions. This is an inspired prototype, NOT a reproduction of TriDet, GlobalPointer, or DiFiNet.

All span arms use the same joint categorical interval loss, fixed original final-layer matching, legal grid i<j, stride 4, and inherited minimum 5-frame support. No GT-injected inference candidates. Singleton sampled targets (known 2 train / 1 dev) are logged and excluded from exact-span CE in every span arm. No dataset exclusions in evaluation. Mask arm still uses its original training objective and therefore is a practical control, not the exact architecture-only comparator.

## Training and interval contrast
Seed 78, 500 steps, identical batch order across arms; span AdamW lr 3e-4, weight decay .05, gradient clip 1.0. Final checkpoint only. No hyperparameter or checkpoint selection on exposed validation. All span arms use identical optimizer settings; mask lr follows its pretrained continuation control.

Training-only hard intervals are expanded/truncated/shifted around assigned target endpoints. Interval quality is maximum sampled IoU over ALL retained same-pair/same-predicate instances. Censored pairs are excluded from the auxiliary ranking loss; retained targets are checked against original annotations. Ranking uses target-versus-candidate margin (1-quality), weight .1. Random control preserves each hard candidate length and number, changes position, and uses its own measured quality. This targets annotation alignment, not a claim that all nonannotated time is semantically negative.

## Evaluation
No learned score fusion in this round. Preserve exact original candidate identities, categories, confidence scores and ordering; change durations and crop the same existing proposal boxes only. Do not claim new all-query recall from this selected-candidate test. Seal all predictions before reading evaluation GT.

Report official per-video AP mean, global R@50/100/200, gained/lost per GT, C recovery, <1 / 1–3 / >=3 seconds, long-coexisting short and persistent groups, both cohorts separately, train/dev fit, singleton and fallback counts. Keep all failures and raw per-GT ranks. Confirm base model/checkpoint unchanged by hashes.

## Decision / kill criteria
- Any baseline, source hash, split, or zero-update mismatch blocks performance interpretation.
- Joint modeling is not supported if it does not beat independent scoring in the fixed-budget comparison.
- Hard interval training is not supported if it fails to beat both plain joint training and length/count-matched random supervision.
- Do not expand a version whose target gain comes with worse combined overall mAP or Top100 hits, or net loss of persistent relations. Cohort disagreement requires another check, not a success claim.
- Negative results limit this 500-step, frozen-feature, fixed-matching prototype; they do not refute every joint interval architecture.

All new code/output goes in commander_span_readout_078 on the user's AutoDL data disk. No checkpoint, shared dataset or environment is modified. 30-minute process alarm; no unattended large training.
