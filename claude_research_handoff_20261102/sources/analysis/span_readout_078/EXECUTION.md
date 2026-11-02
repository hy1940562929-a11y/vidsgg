# 078 execution receipt — completed

User authorized bounded experiments on their AutoDL instance, connect.bjb1.seetacloud.com:16241, RTX4090D48GB. No school server used. Original model/checkpoint/data/environment unchanged.

- Remote root: `/root/autodl-tmp/VrdONE-data/commander_span_readout_078`.
- Frozen code: `code_v1`; results: `results_v1`; logs: `run_v1.log`, `tests_v1.log`.
- Original PID7657 finished. The job continued through the SSH interruption and was NOT restarted after reconnection.
- Job runtime: 366.9554037321359 seconds; all five trainable arms reached step500.
- GPU synthetic tests: 7 passed. Real-input zero-update checks passed.
- All128 evaluation predictions and summaries imported to `handoff/incoming/VIDOR-SPAN-READOUT-078/remote_v1`.
- Import manifest covers180 original files. Independent recomputation verifies prediction seals, source freeze, AP, TopK counts and new/lost records.
- Independent review: PASS_WITH_CONCERNS; continuation: DO_NOT_EXPAND_THIS_VERSION.
- Additional SAME-cache shared-objective audit: completed in2.94seconds without training; source `audit_training_objective.py`, output `verification_v1/objective_audit.json`.
- No further training launched. Original checkpoints and cache files unchanged.

Baseline588/4739 Top100; joint577/4739, +4/-15; short135/1905→122/1905, +0/-13; target6/331→5/331. Joint/hard/random have identical GT matches, not identical interval predictions. Shared training CE declines7.126685→5.142454 on561 eligible events while tIoU50 matches fall458→447; exact endpoint hits increase215→227, so not all localization measures decline.

The stricter continuation addendum was written before any trained outcome was read. Original frozen protocol/code preserved. See REPORT.md for scope and interpretation.

Formal evidence ledger is an iCloud dataless file and its bounded read timed out. Actual evidence entry saved to `state/pending/EV-VIDOR-SPAN-READOUT-078.jsonl`, pending safe merge. No existing state ledger overwritten.
