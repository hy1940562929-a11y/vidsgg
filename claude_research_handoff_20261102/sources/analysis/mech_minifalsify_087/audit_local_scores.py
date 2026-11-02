"""Independent CPU verification of imported 087 scores; never fits a model."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
load = lambda p: json.loads(p.read_text())
summary = load(OUT / "summary.json")
assert summary["complete"] is True
locked = {(r["video_name"], int(r["gt_index"])): r
          for r in csv.DictReader((ROOT / "eval_locked.csv").open())}
members = load(ROOT / "members.json")
primary = {tuple([v, int(i)]) for v, i in members["long_coexist_short_covered"]}
sub = {tuple([v, int(i)]) for v, i in members["long_coexist_subsecond_covered"]}
assert len(primary) == 400 and len(sub) == 163 and sub <= primary
rows = load(OUT / "per_gt.json")
assert len(rows) == 2573 and len({(r["video_name"], r["gt_index"]) for r in rows}) == 2573
receipts = load(OUT / "score_receipts.json")
arrays = {}
for receipt in receipts:
    p = OUT / "scores" / Path(receipt["path"]).name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == receipt["sha256"]
    arrays[(receipt["video_name"], receipt["pair_idx"])] = np.load(p)
max_error = 0.0
verified = []
for r in rows:
    key = (r["video_name"], r["gt_index"])
    old = locked[key]
    if old["pair_idx"]:
        assert r["pair_idx"] == int(old["pair_idx"]) and r["query"] == int(old["query"])
    if not r["eligible"]:
        continue
    z = arrays[(r["video_name"], r["pair_idx"])]
    frames = z["frames"]
    assert np.array_equal(frames, r["origin"] + 4 * np.arange(r["N_valid"]))
    start, end = r["begin_fid"], r["end_fid"]
    pos = (frames >= start) & (frames < end)
    neg = ((frames >= 2 * start - end) & (frames < start)) | (
        (frames >= end) & (frames < 2 * end - start))
    for x in locked.values():
        if (x["video_name"], x["subject_tid"], x["object_tid"], x["predicate"]) == (
                r["video_name"], str(r["subject_tid"]), str(r["object_tid"]), r["predicate"]):
            neg &= ~((frames >= int(x["begin_fid"])) & (frames < int(x["end_fid"])))
    assert pos.sum() == r["N_positive"] and neg.sum() == r["N_negative"]
    assert np.flatnonzero(pos).tolist() == r["positive_indices"]
    assert np.flatnonzero(neg).tolist() == r["negative_indices"]
    result = {}
    for arm in ("fine", "fused", "position"):
        scores = z[arm][r["predicate_id"] - 1]
        # Explicit pairwise comparison, independent of the runner's rank method.
        p, n = scores[pos, None], scores[neg][None, :]
        value = float(((p > n).astype(float) + 0.5 * (p == n)).mean())
        max_error = max(max_error, abs(value - r["auroc"][arm]))
        assert abs(value - r["auroc"][arm]) < 1e-12
        result[arm] = value
    verified.append((key, result))
group_audits = {}
for name, keys in (("long_coexist_short_covered", primary),
                   ("long_coexist_subsecond_covered", sub)):
    by_video = defaultdict(list)
    for key, values in verified:
        if key in keys:
            by_video[key[0]].append(values)
    means = {v: {a: np.mean([x[a] for x in xx]) for a in ("fine", "fused", "position")}
             for v, xx in sorted(by_video.items())}
    rng = np.random.default_rng(87)
    bootstrap = rng.integers(0, len(means), (10000, len(means)))
    comparisons = {}
    for a, b in (("fine", "fused"), ("fine", "position"), ("fused", "position")):
        values = np.array([m[a] - m[b] for m in means.values()])
        mean = float(values.mean())
        ci = np.quantile(values[bootstrap].mean(1), [0.025, 0.975])
        source = summary["groups"][name]["comparisons"][a + "_minus_" + b]
        assert abs(mean - source["mean"]) < 1e-12
        assert np.allclose(ci, source["ci95"], atol=1e-12, rtol=0)
        comparisons[a + "_minus_" + b] = {"mean": mean, "ci95": ci.tolist()}
    group_audits[name] = {"N_total": len(keys), "N_evaluable": sum(map(len, by_video.values())),
                          "N_videos": len(means), "comparisons": comparisons}
history = load(OUT / "training_history.json")
assert len(history) == 1500
for i in range(500):
    items = history[i * 3:(i + 1) * 3]
    assert [r["arm"] for r in items] == ["fine", "fused", "position"]
    assert all(r["step"] == i + 1 for r in items)
    assert len({r["batch"] for r in items}) == 1
audit = {"status": "PASS", "score_files_verified": len(receipts),
         "evaluable_GT_independently_recomputed": len(verified), "max_AUROC_error": max_error,
         "groups": group_audits, "training_history_rows": len(history),
         "scope": "CPU raw-score/coordinate/coverage/aggregation verification, not a new experiment"}
(ROOT / "independent_cpu_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
