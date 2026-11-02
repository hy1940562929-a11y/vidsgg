#!/usr/bin/env python3
"""073 sealed-output history analysis. NumPy + Python standard library only.

No model inference, candidate selection, threshold fitting, or formal detection
evaluation. The only permissible positive decision is annotation-conditioned
feature-history sensitivity; visual semantics remain unverified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = ("B", "N", "P", "R")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 73073
EFFECT_THRESHOLD = 0.03
SCORE_THRESHOLD = 0.25


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def edge_score(class_probabilities, mask_probabilities, predicate_id):
    return np.max(class_probabilities[:, predicate_id, None] * mask_probabilities, axis=0)


def tiou(interval, target):
    if interval is None:
        return 0.0
    start, end = interval
    a, b = target
    inter = max(0, min(end, b) - max(start, a))
    return inter / ((end - start) + (b - a) - inter)


def decoded_hull(bits, origin, stride=4, min_frames=5):
    indices = np.flatnonzero(bits)
    if not len(indices):
        return None
    start = int(origin + stride * indices[0])
    end = int(origin + stride * indices[-1] + 1)
    return [start, end] if end - start >= min_frames else None


def binary_support_iou(bits, times, pair_end, target):
    """GT probe convention: positive samples cover [t,min(t+4,pair_end))."""
    starts = np.asarray(times, dtype=np.int64)[np.asarray(bits, dtype=bool)]
    ends = np.minimum(starts + 4, int(pair_end))
    lengths = np.maximum(0, ends - starts)
    a, b = target
    intersection = np.maximum(0, np.minimum(ends, b) - np.maximum(starts, a)).sum()
    predicted_length = lengths.sum()
    union = predicted_length + (b - a) - intersection
    return float(intersection / union) if union else None


def response_delay(score, times, boundary, stop, fps, want_active):
    """First two adjacent sampled positions satisfying the auxiliary threshold.

    Censor at the annotated gap/next-event end. Immediate activation is reported
    as such; it is not evidence that a distinct new event was established.
    """
    eligible = np.flatnonzero((times >= boundary) & (times < stop))
    condition = score >= SCORE_THRESHOLD if want_active else score < SCORE_THRESHOLD
    first = None
    for left, right in zip(eligible, eligible[1:]):
        if right == left + 1 and condition[left] and condition[right]:
            first = int(left)
            break
    prior = np.flatnonzero(times < boundary)
    return {"observed": first is not None, "censored": first is None,
            "delay_seconds": float((times[first] - boundary) / fps) if first is not None else None,
            "first_sample_index": first,
            "censor_at_seconds": float((stop - boundary) / fps),
            "eligible_sample_count": len(eligible),
            "pre_boundary_score": float(score[prior[-1]]) if len(prior) else None,
            "active_at_first_eligible_sample": bool(score[eligible[0]] >= SCORE_THRESHOLD) if len(eligible) else None,
            "status": "observed" if first is not None else "RIGHT_CENSORED_NOT_SET_TO_ZERO"}


def summarize_score(score, gap_indices, next_indices):
    gap = float(np.mean(score[gap_indices]))
    nxt = float(np.mean(score[next_indices]))
    return {"gap_mean": gap, "next_mean": nxt, "next_minus_gap": nxt - gap}


def bootstrap_video_mean(rows, field):
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_name"]].append(float(row[field]))
    values = np.array([np.mean(by_video[v]) for v in sorted(by_video)], dtype=np.float64)
    if not len(values):
        return {"N_cases": 0, "N_videos": 0, "point": None, "CI_97_5_percent": [None, None],
                "video_means": {}, "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    estimates = values[draws].mean(axis=1)
    ci = np.quantile(estimates, [0.0125, 0.9875]).tolist()
    return {"N_cases": len(rows), "N_videos": len(values), "point": float(values.mean()),
            "CI_97_5_percent": [float(x) for x in ci],
            "video_means": {v: float(np.mean(by_video[v])) for v in sorted(by_video)},
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}


def decide(effect, n_vs_b, descriptive=False):
    if descriptive:
        return "DESCRIPTIVE_SUBGROUP_NOT_A_PRIMARY_GATE"
    if effect["N_cases"] < 30 or effect["N_videos"] < 10:
        return "INCONCLUSIVE_INSUFFICIENT_CASES_OR_VIDEOS"
    lower, upper = effect["CI_97_5_percent"]
    if upper < EFFECT_THRESHOLD:
        return "BELOW_PRESPECIFIED_EFFECT"
    if effect["point"] >= EFFECT_THRESHOLD and lower > 0 and n_vs_b["point"] > 0:
        return "SUPPORT_ANNOTATION_CONDITIONED_SENSITIVITY_ONLY"
    return "INCONCLUSIVE"


def candidate_metrics(raw, record):
    bits = raw["bits"]
    predicate = int(record["predicate_id"])
    top6 = np.any(raw["top6"] == predicate, axis=1)
    times = int(record["origin"]) + 4 * np.arange(int(record["N"]))
    target = record["next"]
    hulls = [decoded_hull(query_bits, record["origin"]) for query_bits in bits]
    hull_iou = np.array([tiou(hull, target) for hull in hulls])
    binary_iou = np.array([binary_support_iou(query_bits, times, record["pair_support"][1], target)
                           for query_bits in bits], dtype=float)
    bridge = np.array([hull is not None and hull[0] < record["previous"][1] and hull[1] > target[0]
                       for hull in hulls], dtype=bool)
    return {"hulls": hulls, "hull_tiou_per_query": hull_iou.tolist(),
            "binary_support_iou_per_query": binary_iou.tolist(), "target_in_top6_per_query": top6.tolist(),
            "hull_candidate_all9": bool(np.any(hull_iou >= .5)),
            "hull_candidate_top6": bool(np.any((hull_iou >= .5) & top6)),
            "binary_candidate_all9": bool(np.any(binary_iou >= .5)),
            "binary_candidate_top6": bool(np.any((binary_iou >= .5) & top6)),
            "gap_bridge_all9": bool(np.any(bridge)), "gap_bridge_top6": bool(np.any(bridge & top6)),
            "gap_bridge_query_count_all9": int(bridge.sum()),
            "gap_bridge_query_count_top6": int((bridge & top6).sum())}


def load_outputs(raw_dir, record):
    outputs = {}
    for arm in ARMS:
        with np.load(raw_dir / f"{record['case_id']}_{arm}.npz", allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in ["pred_logits", "pred_masks", "bits", "top6"]}
        expected = (9, int(record["N"]))
        if arrays["pred_masks"].shape != expected or arrays["bits"].shape != expected:
            raise ValueError(f"Mask shape mismatch {record['case_id']}/{arm}")
        if arrays["pred_logits"].shape[0] != 9 or arrays["pred_logits"].ndim != 2 or arrays["top6"].shape != (9, 6):
            raise ValueError(f"Class/Top6 shape mismatch {record['case_id']}/{arm}")
        if arrays["bits"].dtype != np.bool_:
            raise ValueError("Saved CUDA bits must be boolean, never rethresholded")
        if not np.isfinite(arrays["pred_logits"]).all() or not np.isfinite(arrays["pred_masks"]).all():
            raise ValueError("Nonfinite saved outputs")
        if np.any(arrays["top6"] <= 0) or np.any(arrays["top6"] >= arrays["pred_logits"].shape[-1]):
            raise ValueError("Invalid saved foreground Top6 IDs")
        arrays["class_prob"] = softmax(arrays["pred_logits"])
        arrays["mask_prob"] = sigmoid(arrays["pred_masks"])
        outputs[arm] = arrays
    return outputs


def analyze_case(record, outputs):
    times = int(record["origin"]) + 4 * np.arange(int(record["N"]))
    gap_indices = np.array(record["gap_indices"], dtype=int)
    next_indices = np.array(record["next_indices"], dtype=int)
    if min(len(gap_indices), len(next_indices)) < 2:
        raise ValueError("Selected complete case lacks required gap/next support")
    if np.any(times >= record["pair_support"][1]):
        raise ValueError("Sample start beyond pair support")
    if not np.array_equal(gap_indices, np.flatnonzero((times >= record["previous"][1]) & (times + 4 <= record["next"][0]))):
        raise ValueError("Gap indices differ from frozen fully-contained definition")
    if not np.array_equal(next_indices, np.flatnonzero((times >= record["next"][0]) & (times + 4 <= record["next"][1]))):
        raise ValueError("Next indices differ from frozen fully-contained definition")
    result = {key: record[key] for key in ["case_id", "video_name", "predicate", "predicate_id", "next_short", "next_sec",
                                           "subject_tid", "object_tid", "previous_gt_index", "next_gt_index", "previous", "next", "donors"]}
    result["visual_audit"] = "PENDING"
    result["arms"] = {}
    for arm in ARMS:
        raw = outputs[arm]
        score = edge_score(raw["class_prob"], raw["mask_prob"], record["predicate_id"])
        fixed_class_score = edge_score(outputs["B"]["class_prob"], raw["mask_prob"], record["predicate_id"])
        metrics = summarize_score(score, gap_indices, next_indices)
        metrics["B_class_auxiliary"] = summarize_score(fixed_class_score, gap_indices, next_indices)
        metrics.update(candidate_metrics(raw, record))
        metrics["termination_delay"] = response_delay(score, times, record["previous"][1], record["next"][0], record["fps"], False)
        metrics["establishment_delay"] = response_delay(score, times, record["next"][0], record["next"][1], record["fps"], True)
        result["arms"][arm] = metrics
    arms = result["arms"]
    result["off"] = arms["P"]["gap_mean"] - arms["N"]["gap_mean"]
    result["restart"] = arms["N"]["next_mean"] - arms["P"]["next_mean"]
    result["off_N_vs_B"] = arms["B"]["gap_mean"] - arms["N"]["gap_mean"]
    result["restart_N_vs_B"] = arms["N"]["next_mean"] - arms["B"]["next_mean"]
    result["off_Bclass_auxiliary"] = arms["P"]["B_class_auxiliary"]["gap_mean"] - arms["N"]["B_class_auxiliary"]["gap_mean"]
    result["restart_Bclass_auxiliary"] = arms["N"]["B_class_auxiliary"]["next_mean"] - arms["P"]["B_class_auxiliary"]["next_mean"]
    for arm in ("P", "R"):
        result[f"off_{arm}_vs_B"] = arms["B"]["gap_mean"] - arms[arm]["gap_mean"]
        result[f"restart_{arm}_vs_B"] = arms[arm]["next_mean"] - arms["B"]["next_mean"]
    result["continuing_predicates"] = []
    support = np.union1d(gap_indices, next_indices)
    for item in record["continuing_predicates"]:
        values = {arm: float(np.mean(edge_score(outputs[arm]["class_prob"], outputs[arm]["mask_prob"], item["predicate_id"])[support]))
                  for arm in ARMS}
        result["continuing_predicates"].append(dict(item, gap_and_next_mean=values,
            delta_vs_B={arm: values[arm] - values["B"] for arm in ARMS if arm != "B"},
            role="ANNOTATED_CONTINUING_PREDICATE_NOT_OCCLUSION_CONTROL"))
    return result


def fraction(n, d):
    return {"numerator": int(n), "denominator": int(d), "fraction": float(n / d) if d else None}


def summarize_cohort(rows, descriptive=False):
    primary = {}
    for field in ("off", "restart"):
        effect = bootstrap_video_mean(rows, field)
        vs_baseline = bootstrap_video_mean(rows, field + "_N_vs_B")
        primary[field] = {"effect_P_vs_N": effect, "directional_N_vs_B": vs_baseline,
                          "decision": decide(effect, vs_baseline, descriptive)}
    candidates = {}
    for metric in ("hull_candidate_all9", "hull_candidate_top6", "binary_candidate_all9", "binary_candidate_top6", "gap_bridge_all9", "gap_bridge_top6"):
        base = np.array([r["arms"]["B"][metric] for r in rows], dtype=bool)
        candidates[metric] = {}
        for arm in ARMS:
            values = np.array([r["arms"][arm][metric] for r in rows], dtype=bool)
            candidates[metric][arm] = {"count": fraction(values.sum(), len(rows)),
                "new_vs_B": int(np.sum(values & ~base)), "lost_vs_B": int(np.sum(base & ~values)),
                "net_vs_B": int(np.sum(values) - np.sum(base))}
    delay = {}
    for metric in ("termination_delay", "establishment_delay"):
        delay[metric] = {}
        for arm in ARMS:
            values = [r["arms"][arm][metric] for r in rows]
            observed = [v["delay_seconds"] for v in values if v["observed"]]
            delay[metric][arm] = {"observed": fraction(len(observed), len(values)),
                "right_censored": fraction(sum(v["censored"] for v in values), len(values)),
                "observed_only_mean_seconds": float(np.mean(observed)) if observed else None,
                "observed_only_median_seconds": float(np.median(observed)) if observed else None,
                "warning": "Observed-only delay does not compare methods independently of censoring; all censored cases retained in per_case"}
    secondary = {}
    for metric in ("gap_mean", "next_mean", "next_minus_gap"):
        secondary[metric] = {}
        for arm in ARMS:
            synthetic = [{"video_name": r["video_name"], "value": r["arms"][arm][metric]} for r in rows]
            secondary[metric][arm] = bootstrap_video_mean(synthetic, "value")
    continued = []
    for r in rows:
        if r["continuing_predicates"]:
            continued.append({"video_name": r["video_name"], **{
                f"{arm}_minus_B": float(np.mean([x["delta_vs_B"][arm] for x in r["continuing_predicates"]]))
                for arm in ("N", "P", "R")}})
    return {"N_cases": len(rows), "N_videos": len({r["video_name"] for r in rows}), "primary": primary,
            "candidate_existence_not_recall": candidates, "auxiliary_delay": delay,
            "arm_score_descriptive": secondary,
            "B_class_auxiliary": {field: bootstrap_video_mean(rows, field) for field in ("off_Bclass_auxiliary", "restart_Bclass_auxiliary")},
            "donor_and_shuffle_vs_B_auxiliary": {field: bootstrap_video_mean(rows, field)
                for field in ("off_P_vs_B", "restart_P_vs_B", "off_R_vs_B", "restart_R_vs_B")},
            "continuing_predicate_score_control": {"role": "NOT_AN_OCCLUSION_CONTROL",
                "N_cases": len(continued), "effects": {f"{arm}_minus_B": bootstrap_video_mean(continued, f"{arm}_minus_B") for arm in ("N", "P", "R")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="Inference output root containing raw_seal.json and raw/*.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Refusing nonempty output directory")
    root = args.raw if (args.raw / "raw_seal.json").is_file() else args.raw.parent
    raw_dir = root / "raw"
    seal_path = root / "raw_seal.json"
    seal = json.loads(seal_path.read_text())
    plan = json.loads(args.plan.read_text())
    if seal.get("complete") is not True or plan.get("complete") is not True:
        raise ValueError("Plan/inference must be sealed complete")
    if seal["selection_sha256"] != sha(args.plan):
        raise ValueError("Selection hash differs from raw seal")
    if not seal.get("model_state_unchanged") or seal.get("training_steps") != 0:
        raise ValueError("Frozen-weight/no-training precondition failed")
    cases = plan["cases"]
    if plan["N_cases"] != len(cases) or len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Case population mismatch")
    if cases and not seal.get("noop_exact"):
        raise ValueError("No-op exactness precondition failed")
    expected = {f"{r['case_id']}_{a}.npz" for r in cases for a in ARMS}
    actual = {p.name for p in raw_dir.glob("*.npz")}
    if expected != actual or expected != set(seal["raw_sha256"]):
        raise ValueError("Case/arm/raw-seal file sets differ")
    for name, fingerprint in seal["raw_sha256"].items():
        if sha(raw_dir / name) != fingerprint:
            raise ValueError(f"Raw fingerprint mismatch: {name}")
    audits_path = root / "input_audits.json"
    audits = json.loads(audits_path.read_text())
    by_audit = {(r["case_id"], r["arm"]): r for r in audits}
    if len(by_audit) != len(audits) or set(by_audit) != {(r["case_id"], a) for r in cases for a in ARMS}:
        raise ValueError("Input-audit case/arm mismatch")
    for audit in audits:
        if not all(audit.get(k) is True for k in ("future_exact", "gap_exact", "next_exact", "length_unchanged")):
            raise ValueError("Input invariance audit failed")
    rows = [analyze_case(record, load_outputs(raw_dir, record)) for record in cases]
    for row in rows:
        row["input_perturbation_audits"] = {arm: by_audit[row["case_id"], arm] for arm in ARMS}
    short = [r for r in rows if r["next_short"]]
    by_predicate = defaultdict(list)
    for row in rows:
        by_predicate[row["predicate"]].append(row)
    summary = {"status": "COMPLETE_SEALED_OUTPUT_ANALYSIS", "formal_mAP": "NOT_APPLICABLE_GT_ASSISTED_FEATURE_INTERVENTION",
        "formal_R_at_100": "NOT_APPLICABLE_GT_ASSISTED_FEATURE_INTERVENTION", "visual_audit": "PENDING",
        "method_admission": "NOT_ALLOWED_BY_THIS_PROBE", "semantic_causal_claim_allowed": False,
        "primary_all_cases": summarize_cohort(rows), "short_next_descriptive_only": summarize_cohort(short, True),
        "predicate_strata_descriptive_only": {p: {"N_cases": len(rr), "N_videos": len({r["video_name"] for r in rr}),
            "off": bootstrap_video_mean(rr, "off"), "restart": bootstrap_video_mean(rr, "restart")} for p, rr in sorted(by_predicate.items())},
        "definitions": {"score": "max_q softmax(class_logits_q)[predicate] * sigmoid(mask_logits_q(t)); float64 offline evaluation",
            "score_regions": "Frozen plan fully-contained 4-frame gap/next indices",
            "off": "P gap_mean - N gap_mean", "restart": "N next_mean - P next_mean",
            "B_class_auxiliary": "Baseline class probabilities times each arm mask probabilities; descriptive only",
            "hull": "Actual saved CUDA sigmoid>0.5 bits; first=origin+4*k_first,last=origin+4*k_last+1; min 5 frames",
            "binary_IoU": "Union of [origin+4k,min(origin+4k+4,pair_end)) for true saved CUDA bits versus full GT next interval",
            "candidate_existence": "Any of nine queries; separate saved CUDA target-in-Top6 condition; no one-to-one assignment or prediction budget",
            "gap_bridge": "Valid decoded query hull starts before previous GT end and ends after next GT start; not verified old-edge persistence",
            "delay": "First two consecutive score samples <0.25 after previous end before next start, or >=0.25 within next; no success retains right censoring",
            "bootstrap": "Equal-video mean of within-video event means; 10000 video resamples; two-sided 97.5 percent CI quantiles .0125/.9875",
            "gate": "At least 30 complete cases and 10 videos; point>=.03 and lower>0 and N-vs-B directional point>0; upper<.03 below-prespecified-effect; otherwise inconclusive",
            "selection": "GT-assisted, exposed selected 27-video input cache; not a generalization evaluation"},
        "technical_checks": {"sealed_hashes_verified": True, "selection_matches_seal": True, "frozen_model": True,
            "noop_exact": seal["noop_exact"], "input_invariance_audits_passed": True, "training_steps": 0},
        "input_perturbation_strength_descriptive": {arm: bootstrap_video_mean([
            {"video_name": record["video_name"], "rms": by_audit[record["case_id"], arm]["edited_block_rms"]}
            for record in cases], "rms") for arm in ARMS},
        "limitations": ["No visual verification of annotation gap or N donor semantic absence",
            "Donor replacement may alter correlated visual/semantic features despite coarse matching",
            "Neither candidate-existence gains nor GT-selected donor effects are method improvements",
            "R and B-class auxiliary outputs cannot rescue failed primary endpoints",
            "Continuing other predicates are not occlusion-persistence controls",
            "Decoded bridge can occur despite negative raw-mask gap; inspect per-case raw and decoded behavior separately"]}
    sources = {"script_sha256": sha(__file__), "plan": {"path": str(args.plan.resolve()), "sha256": sha(args.plan)},
        "seal": {"path": str(seal_path.resolve()), "sha256": sha(seal_path)},
        "input_audits": {"path": str(audits_path.resolve()), "sha256": sha(audits_path)},
        "raw": {"directory": str(raw_dir.resolve()), "sha256": seal["raw_sha256"]},
        "selection_recorded_upstream_sources": plan.get("sources", {})}
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "summary.json", summary)
    dump(args.out / "per_case.json", rows)
    dump(args.out / "source_hashes.json", sources)
    flat = []
    for row in rows:
        item = {k: row[k] for k in ("case_id", "video_name", "predicate", "next_short", "next_sec", "off", "restart", "off_N_vs_B", "restart_N_vs_B")}
        for arm in ARMS:
            for metric in ("gap_mean", "next_mean", "next_minus_gap", "hull_candidate_all9", "hull_candidate_top6", "binary_candidate_all9", "binary_candidate_top6", "gap_bridge_all9", "gap_bridge_top6"):
                item[f"{arm}_{metric}"] = row["arms"][arm][metric]
            for metric in ("termination_delay", "establishment_delay"):
                item[f"{arm}_{metric}_seconds"] = row["arms"][arm][metric]["delay_seconds"]
                item[f"{arm}_{metric}_censored"] = row["arms"][arm][metric]["censored"]
        flat.append(item)
    with (args.out / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["case_id", "video_name"])
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": summary["status"], "N_cases": len(rows), "N_videos": len({r["video_name"] for r in rows}),
                      "primary": summary["primary_all_cases"]["primary"], "output": str(args.out.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
