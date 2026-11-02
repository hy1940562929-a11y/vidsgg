# 075 execution record

Authorized AutoDL: `root@connect.bjb1.seetacloud.com:16241`. Execution uses existing private files and the existing `rebinding` Python environment.

The initial `code_v1/results_v1` launch passed six loss/domain tests but stopped at cache projection validation before training. All results from that attempt are preserved.

The imported numerical probes show that all 88 training batches are bit-exact. Only adapter_dev batch 22, containing one pair, has maximum absolute drift 3.814697265625e-6. All 111 batches preserve finite values, actual sigmoid>0.5 masks, decoded intervals and four-layer Hungarian assignments. The independent reviewer approved a narrowly bounded numerical compatibility revision.

The actual execution is `commander_temporal_order_075/code_v2/run.py`, SHA256 `32763abad8e6dd54777147a641d1b904cbfc1741535a3f9d3dc1510d7c88a7b1`, writing `results_v2`. The remote `code_v2/software_freeze.json` is authoritative. Local cloud-file reads stalled during packaging, so the unchanged server-side code_v1 files were reused and only reviewed changed files were deployed. The remote protocol includes the same numerical revision in an appended English addendum; its exact text differs from the local Chinese working copy. The numerical audit script and all actual deployed files will be imported with results.

No loss, seed, optimization budget, evaluation group, or continuation gate was changed. After import, the local protocol, technical revision, verifier and freeze were synchronized to the actual deployed code_v2. The completed result is negative and independently verified; see REPORT.md and reviews/temporal_order_075_result_review.md. No additional training was launched.
