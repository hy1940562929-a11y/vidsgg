"""E098 material, nine-slot capacity and original underreading gates. CPU only."""
import csv
import hashlib
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/root/autodl-tmp/VrdONE-data")
CACHE = ROOT / "commander_cache_import_094/cache_v4"
PREFLIGHT = ROOT / "commander_cause_arch_098/preflight_v1"
OUT = ROOT / "commander_cause_arch_098/admission_v1"
EVENTS = ROOT / "commander_cause_arch_095/results_v2/events.json"
MANIFEST_SHA = "2613d524a528528212cb28c1d2a543431d7fc9b08cfc9d61f324a6aebd2f2269"
EVENT_SHA = "0c412de71d7ba10dbd1380798741c0069fb65557ee6659ff21e7992c2dc781fa"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csvout(name, rows):
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tiou(intervals, events):
    if not intervals or not events:
        return np.zeros((len(intervals), len(events))), np.zeros((len(intervals), len(events)), bool)
    i = np.asarray(intervals, dtype=np.int64)
    g = np.asarray([[e["begin_fid"], e["end_fid"]] for e in events], dtype=np.int64)
    inter = np.maximum(0, np.minimum(i[:, None, 1], g[None, :, 1]) - np.maximum(i[:, None, 0], g[None, :, 0]))
    total = (i[:, 1] - i[:, 0])[:, None] + (g[:, 1] - g[:, 0])[None, :]
    return inter / (total - inter), 3 * inter >= total


def matched_count(intervals, rows):
    score, eligible = tiou(intervals, rows)
    if not score.size:
        return 0
    # Count-only gate: lexicographic tie resolution cannot alter this count.
    a, b = linear_sum_assignment(-(eligible * (10 + score)))
    return int(eligible[a, b].sum())


def legal(e, n, origin):
    # Enumerate every legal endpoint pair. This oracle never generates predictions.
    starts, ends = np.triu_indices(n, k=1)
    left, right = origin + 4 * starts, origin + 4 * ends + 1
    inter = np.maximum(0, np.minimum(right, e["end_fid"]) - np.maximum(left, e["begin_fid"]))
    return bool(np.any(3 * inter >= right - left + e["end_fid"] - e["begin_fid"]))


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    phase = "MATERIAL"
    try:
        torch.set_num_threads(2)
        require(sha(CACHE / "manifest.json") == MANIFEST_SHA, "MANIFEST_SHA")
        require(sha(EVENTS) == EVENT_SHA, "EVENT_MEMBERS_SHA")
        m = json.loads((CACHE / "manifest.json").read_text())
        for rec in m["files"]:
            require(sha(CACHE / rec["path"]) == rec["sha256"], "CACHE_SHA " + rec["path"])
        prior = json.loads((PREFLIGHT / "status.json").read_text())
        require(prior["status"] == "NATIVE_REBUILD_PASS" and prior["checked"] == 1725 and prior["max_abs"] <= 4e-6, "NATIVE_GATE")
        models = json.loads((PREFLIGHT / "material_receipt.json").read_text())["models"]
        for p, h in models.items():
            require(sha(p) == h, "ACTUAL_SOURCE_CHANGED " + p)
        splits = {}
        for split, batch_count, pair_count, event_count in (("train", 88, 350, 563), ("adapter_dev", 23, 89, 169)):
            paths = sorted((CACHE / split).glob("*.pt"))
            require(len(paths) == batch_count, "BATCH_COUNT")
            pair_ids, videos, event_total = [], set(), 0
            for p in paths:
                c = torch.load(p, map_location="cpu", weights_only=False)
                require(len(c["metadata"]) == len(c["preds"]) == len(c["targets"]) == c["f"].shape[0], "TARGET_BATCH")
                for j, (meta, preds, targets) in enumerate(zip(c["metadata"], c["preds"], c["targets"])):
                    n = len(preds)
                    require(targets.shape == (n, c["f"].shape[-1]) and n == len(meta["events"]), "EVENT_COMPLETE")
                    require(n <= 9 and bool((preds > 0).all()) and bool((preds <= 50).all()), "LABELS_OR_COUNT")
                    require([x["target_index"] for x in meta["events"]] == list(range(n)), "TARGET_INDICES")
                    require([x["predicate_id"] for x in meta["events"]] == preds.tolist(), "TARGET_LABEL_IDENTITY")
                    require(len(set(x["annotation_gt_index"] for x in meta["events"])) == n, "GT_DUPLICATE")
                    require(bool(((targets == 0) | (targets == 1)).all()), "TARGET_BINARY")
                    require(bool((targets.sum(-1) > 0).all()), "TARGET_EMPTY")
                    require(bool((targets[:, ~c["valid"][j, 0]] == 0).all()), "TARGET_PADDING")
                    videos.add(meta["video_id"])
                    pair_ids.append((meta["video_id"], tuple(meta["pair_key"])))
                    event_total += n
            require(len(pair_ids) == pair_count and event_total == event_count, "SOURCE_COUNTS")
            require(videos == set(m["split_videos"][split]), "SPLIT_MEMBERS")
            splits[split] = {"batches": batch_count, "pairs": pair_count, "events": event_total, "videos": len(videos)}
        events = json.loads(EVENTS.read_text())
        counts = {g: sum(bool(e[g]) for e in events) for g in ("main400", "sub163", "residual334", "long_protection")}
        require(len(events) == 2573 and counts == {"main400": 400, "sub163": 163, "residual334": 334, "long_protection": 1101}, "COHORT_COUNTS")
        require(set(m["split_videos"]["train"]).isdisjoint(m["eval_videos"]) and set(m["split_videos"]["adapter_dev"]).isdisjoint(m["eval_videos"]), "SPLIT_OVERLAP")
        dump("material_identity.json", {
            "status": "PASS", "manifest_sha256": MANIFEST_SHA, "events_sha256": EVENT_SHA,
            "verified_manifest_files": len(m["files"]), "splits": splits, "eval_counts": counts,
            "source_note": "All actual model sources frozen before native rebuild and rechecked unchanged; local_transformer has no prior094 source digest, accepted only with full same-input native reconstruction, not claimed historically hashed.",
            "targets_note": "094 hashed complete independent targets and metadata, all732 entries/indices/classes retained; no frozen matching subset used.",
            "training_updates": 0})
        phase = "CAPACITY_AND_READING"
        bypair = defaultdict(list)
        for e in events:
            bypair[(e["video_name"], e["pair_idx"])].append(e)
        per_gt, interval_sets, legal_flags = [], {}, {}
        key_audit = []
        for (video, pi), rows in sorted(bypair.items()):
            name = f"gt27_{video}_{pi:04d}.pt"
            c = torch.load(CACHE / "eval" / name, map_location="cpu", weights_only=False)
            d = torch.load(PREFLIGHT / "decoder_inputs" / name, map_location="cpu", weights_only=False)
            meta = c["metadata"]
            origin, n = int(meta["origin"]), int(meta["N_valid"])
            require(all(e["origin"] == origin and e["N_valid"] == n and e["stride"] == 4 for e in rows), "FIXED_PAIR_COORDINATE")
            require(torch.equal(d["output_mask"], c["valid"]), "VALID_MASK")
            key_mask = d["key_mask"][0, 0].numpy()
            require(np.array_equal(np.flatnonzero(key_mask), np.arange(key_mask.sum())), "KEY_PREFIX")
            nk = int(key_mask.sum())
            attention = d["attention"].numpy().astype(np.float64)
            require(attention.shape[:3] == (1, 8, 9) and attention.shape[-1] == len(key_mask), "ATTENTION_SHAPE")
            require(np.isfinite(attention).all() and np.all(attention >= 0), "ATTENTION_FINITE")
            require(np.max(np.abs(attention.sum(-1) - 1)) < 1e-6 and not attention[..., ~key_mask].any(), "ATTENTION_VALIDITY")
            # Three centered stride2 backbone layers; decoder key_conv stride1,
            # kernel3/pad1. All four decoder layers share the same memory grid.
            centers = origin + 32 * np.arange(nk)
            edges = np.r_[meta["pair_start"], (centers[:-1] + centers[1:]) / 2, meta["pair_end"]].astype(float)
            require(nk >= 1 and (np.diff(edges) > 0).all(), "KEY_CELL_COORDINATES")
            require(centers[0] >= edges[0] and centers[-1] < edges[-1], "KEY_CENTERS_OUTSIDE_SEGMENT")
            A = attention[0, :, :, :nk].mean(0)
            key_audit.append({"video_name": video, "pair_idx": pi, "key_count": nk,
                              "origin": origin, "key_stride_frames": 32,
                              "pair_start": meta["pair_start"], "pair_end": meta["pair_end"],
                              "layers_same_memory_grid": 4, "last_layer_heads": 8})
            intervals = []
            for q in range(9):
                pos = torch.where(c["masks"][-1, 0, q, :n].sigmoid() > .5)[0]
                if len(pos):
                    a, b = origin + 4 * int(pos[0]), origin + 4 * int(pos[-1]) + 1
                    if b - a >= 5:
                        intervals.append((a, b))
            interval_sets[(video, pi)] = intervals
            for e in rows:
                key = (video, e["gt_index"])
                legal_flags[key] = legal(e, n, origin)
                overlap = np.maximum(0, np.minimum(edges[1:], e["end_fid"]) - np.maximum(edges[:-1], e["begin_fid"]))
                duration = e["end_fid"] - e["begin_fid"]
                reading = float((A * (overlap / np.diff(edges))[None, :]).sum())
                relative = reading / (9 * duration / (edges[-1] - edges[0]))
                per_gt.append({"video_name": video, "gt_index": e["gt_index"], "pair_idx": pi,
                               **{g: e[g] for g in counts}, "R": relative, "low_R": relative < .5,
                               "GT_duration_frames": duration, "pair_length_frames": edges[-1] - edges[0],
                               "target_fully_in_pair": e["begin_fid"] >= edges[0] and e["end_fid"] <= edges[-1],
                               "legal_interval_exists": legal_flags[key]})
        csvout("per_gt_reading.csv", per_gt)
        csvout("key_coordinate_audit.csv", key_audit)
        groups = {}
        detail = []
        for group in [*counts, "all"]:
            selected = [e for e in events if group == "all" or e[group]]
            pairs = defaultdict(list)
            for e in selected:
                pairs[(e["video_name"], e["pair_idx"])].append(e)
            video_counts = defaultdict(lambda: [0, 0, 0])
            for pair, rs in sorted(pairs.items()):
                feasible = sum(legal_flags[(e["video_name"], e["gt_index"])] for e in rs)
                u = min(9, feasible)
                h = matched_count(interval_sets[pair], rs)
                require(h <= u, "CAPACITY_UPPER_VIOLATED")
                detail.append({"group": group, "video_name": pair[0], "pair_idx": pair[1],
                               "N": len(rs), "legal_N": feasible, "U9": u, "C0_hits": h,
                               "nonempty_H": len(interval_sets[pair])})
                for i, v in enumerate((len(rs), h, u)):
                    video_counts[pair[0]][i] += v
            groups[group] = {"N": len(selected), "videos": len(video_counts),
                             "C0_hits": sum(x[1] for x in video_counts.values()),
                             "U9": sum(x[2] for x in video_counts.values()),
                             "video_macro_max_improvement": float(np.mean([(x[2] - x[1]) / x[0] for x in video_counts.values()])),
                             "video_counts_N_C0_U9": dict(video_counts)}
        csvout("capacity_per_pair.csv", detail)
        capacity_pass = (groups["residual334"]["U9"] > groups["residual334"]["C0_hits"]
                         and groups["main400"]["video_macro_max_improvement"] >= .05)
        dump("capacity_audit.json", {"pass": capacity_pass, "groups": groups,
                                    "count_matching_ties": "Count invariant to tied optimal assignments; no per-GT C0 hit identities claimed at this gate."})
        if not capacity_pass:
            dump("status.json", {"status": "REJECT", "subreason": "SCOPE_NOT_TESTABLE", "training_runs": 0})
            return 0
        summary = {}
        for group in [*counts, "all"]:
            rs = [r for r in per_gt if group == "all" or r[group]]
            bc = defaultdict(lambda: [0, 0])
            for r in rs:
                bc[r["video_name"]][0] += int(r["low_R"])
                bc[r["video_name"]][1] += 1
            arr = np.array(list(bc.values()))
            # Video cluster sampling, then re-sum event numerator/denominator.
            ix = np.random.default_rng(98).integers(0, len(arr), (10000, len(arr)))
            boot = arr[ix].sum(1)
            proportions = boot[:, 0] / boot[:, 1]
            lo, hi = np.quantile(proportions, [.025, .975])
            summary[group] = {"N": len(rs), "videos": len(arr), "low_R_numerator": int(arr[:, 0].sum()),
                              "low_R_denominator": len(rs), "low_R_proportion": float(arr[:, 0].sum()/len(rs)),
                              "CI95": [float(lo), float(hi)], "mean_R": float(np.mean([r["R"] for r in rs])),
                              "median_R": float(np.median([r["R"] for r in rs])),
                              "target_not_fully_in_selected_pair": sum(not r["target_fully_in_pair"] for r in rs)}
        low_pass = summary["residual334"]["CI95"][0] > .5
        dump("underreading_audit.json", {"pass": low_pass, "threshold_R": .5,
                                        "required_lower_CI": .5, "bootstrap_seed": 98,
                                        "bootstrap_reps": 10000, "groups": summary})
        dump("status.json", {"status": "ADMISSION_PASS" if low_pass else "REJECT",
                            "subreason": None if low_pass else "UNDERREADING_NOT_ESTABLISHED",
                            "training_runs": 0, "elapsed_seconds": time.monotonic() - start})
        print((OUT / "status.json").read_text(), flush=True)
        print(json.dumps(summary["residual334"]), flush=True)
    except Exception as exc:
        dump("status.json", {"status": "TECHNICAL_FAILURE", "phase": phase, "error": str(exc),
                            "traceback": traceback.format_exc(), "training_runs": 0,
                            "elapsed_seconds": time.monotonic() - start})
        print((OUT / "status.json").read_text(), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
