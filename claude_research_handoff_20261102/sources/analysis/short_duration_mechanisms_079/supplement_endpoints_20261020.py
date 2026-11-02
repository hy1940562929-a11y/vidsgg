"""Supplement A from saved raw outputs. Explicit GT-time oracle, not inference."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Use a fresh output directory"
    sys.path.insert(0, str(args.code.resolve()))
    from sampling_math import envelope_intervals, interval_iou

    with (args.source / "per_gt.csv").open(encoding="utf-8-sig") as f:
        gt = list(csv.DictReader(f))
    audit = json.loads((args.source / "video_input_audit.json").read_text())
    rows = []
    for video in audit:
        targets = [r for r in gt if r["video"] == video["video"]]
        for arm, info in video["arms"].items():
            candidates = {}
            for rec in info["records"]:
                key = tuple(rec["key"][:2])
                with np.load(args.source / rec["raw_file"]) as raw:
                    spans = envelope_intervals(raw["native_binary_mask"],
                                               raw["frames"], min_frames=5)
                for q, span in enumerate(spans):
                    if span is not None:
                        candidates.setdefault(key, []).append(
                            (span, rec["raw_file"], q))
            for target in targets:
                a, b = int(target["start"]), int(target["end"])
                fps = float(target["fps"])
                key = (int(target["subject_tid"]), int(target["object_tid"]))
                cs = candidates.get(key, [])
                # First in recorded pair order and query order breaks IoU ties.
                winner = max(cs, key=lambda c: interval_iou(c[0], (a, b))) if cs else None
                iou = interval_iou(winner[0], (a, b)) if winner else 0.0
                expected = float(target[arm + "__any_query_iou"])
                assert abs(iou - expected) < 1e-6, (video["video"], arm, target["gt_index"])
                r = dict(video=target["video"], gt_index=target["gt_index"], arm=arm,
                         duration_seconds=(b-a)/fps,
                         long_copresence_short=target["long_copresence_short"],
                         gt_start=a, gt_end=b, tiou=iou, has_interval=bool(winner))
                if winner:
                    (start, end), source, query = winner
                    r.update(pred_start=int(start), pred_end=int(end),
                             source_raw=source, query=query,
                             start_abs_seconds=abs(start-a)/fps,
                             end_abs_seconds=abs(end-b)/fps,
                             mean_abs_endpoint_seconds=(abs(start-a)+abs(end-b))/(2*fps),
                             mean_relative_endpoint_error=(abs(start-a)+abs(end-b))/(2*(b-a)))
                rows.append(r)
    groups = {
        "all": lambda r: True,
        "short_lt3": lambda r: r["duration_seconds"] < 3,
        "long_ge3": lambda r: r["duration_seconds"] >= 3,
        "sub1": lambda r: r["duration_seconds"] < 1,
        "long_copresence_short": lambda r: str(r["long_copresence_short"]).lower() == "true",
    }
    summaries = []
    for arm in sorted({r["arm"] for r in rows}):
        for group, choose in groups.items():
            rr = [r for r in rows if r["arm"] == arm and choose(r)]
            valid = [r for r in rr if r["has_interval"]]
            s = dict(arm=arm, group=group, N_GT=len(rr), N_endpoint=len(valid),
                     N_missing_endpoint=len(rr)-len(valid),
                     mean_tiou=sum(r["tiou"] for r in rr)/len(rr))
            for field in ("start_abs_seconds", "end_abs_seconds",
                          "mean_abs_endpoint_seconds", "mean_relative_endpoint_error"):
                s[field] = sum(r[field] for r in valid)/len(valid) if valid else None
            summaries.append(s)
    args.out.mkdir(parents=True)
    for name, data in [("per_gt_endpoints.csv", rows), ("endpoint_summary.csv", summaries)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (args.out/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    receipt = dict(
        scope="Post-run reporting supplement; no inference/training/new threshold",
        selection="Any-predicate GT-time best-IoU interval; first recorded pair/query breaks ties",
        endpoint_denominator="Only rows with valid predicted interval; absence retained separately",
        warning="Oracle boundary diagnostic, not deployable or official recall",
        rows=len(rows), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((args.source/n).read_bytes()).hexdigest()
                 for n in ["per_gt.csv", "video_input_audit.json"]},
        outputs={n:hashlib.sha256((args.out/n).read_bytes()).hexdigest()
                 for n in ["per_gt_endpoints.csv", "endpoint_summary.csv"]})
    (args.out/"manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(rows=len(rows), all_ious_reproduced=True, out=str(args.out))))


if __name__ == "__main__":
    main()
