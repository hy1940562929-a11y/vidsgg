#!/usr/bin/env python3
"""Read-only census of existing VidOR records; no model, GT oracle, or training.

Sources may be either a flat bundle or a workspace root. Only the mandatory
duration CSV defines the population; absent optional joins remain unknown.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_PATHS = {
    "durations": "analysis/entity_relation_duration_20260929/per_gt_durations.csv",
    "features013": "analysis/vrdone_common_failure_013/per_gt_features.csv",
    "cascade056": "analysis/temporal_evidence_audit_056/per_gt_cascade.csv",
    "correction057": "analysis/numerical_closure_057/extended_cascade.json",
}


def key(row):
    return row["video_name"], int(row["gt_index"])


def optional_bool(value):
    if value is None or value == "":
        return None
    if value in (True, 1, "True", "true", "1"):
        return True
    if value in (False, 0, "False", "false", "0"):
        return False
    raise ValueError(f"Unrecognized boolean {value!r}")


def optional_float(value):
    return None if value is None or value == "" else float(value)


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate video/GT key in {path}")
    return result


def locate(root, relative):
    candidates = [root / relative, root / Path(relative).name]
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(existing) > 1:
        if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in existing}) != 1:
            raise ValueError(f"Ambiguous conflicting copies of {relative}: {existing}")
    return existing[0] if existing else None


def patch_stages(cascade, correction):
    """Preserve every 056 row; overlay only explicitly verified 057 changes."""
    patched = {k: dict(v) for k, v in cascade.items()}
    seen = set()
    for change in correction.get("changes", []):
        k = key(change)
        if k in seen or k not in patched:
            raise ValueError(f"Duplicate or absent 057 correction key {k}")
        seen.add(k)
        row = patched[k]
        if row["stage"] != change["old_stage"] or row["cohort"] != change["cohort"]:
            raise ValueError(f"057 source mismatch at {k}")
        row["stage_056"] = row["stage"]
        row["stage"] = change["new_stage"]
    for row in patched.values():
        row.setdefault("stage_056", row["stage"])
    return patched


def event_components(rows):
    """Merge overlap/touching same-triplet annotations; retain original IDs.

    Positive gaps between returned components lack this predicate's annotation,
    but are NOT evidence of true absence or continuous entity support.
    """
    result = []
    for row in sorted(rows, key=lambda r: (int(r["begin_fid"]), int(r["end_fid"]), int(r["gt_index"]))):
        start, end = int(row["begin_fid"]), int(row["end_fid"])
        if end <= start:
            raise ValueError(f"Non-positive GT interval at {key(row)}")
        if result and start <= result[-1]["end_fid"]:
            result[-1]["end_fid"] = max(end, result[-1]["end_fid"])
            result[-1]["gt_indices"].append(int(row["gt_index"]))
        else:
            result.append({"begin_fid": start, "end_fid": end,
                           "gt_indices": [int(row["gt_index"])]})
    return result


def structural_history(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["video_name"], int(row["subject_tid"]), row["predicate"],
                 int(row["object_tid"]))].append(row)
    transitions = []
    for (video, sid, predicate, oid), members in sorted(grouped.items()):
        rates = {float(r["fps"]) for r in members}
        if len(rates) != 1:
            raise ValueError(f"Inconsistent fps within {video}")
        fps = rates.pop()
        components = event_components(members)
        by_id = {int(r["gt_index"]): r for r in members}
        gaps = [{"begin_fid": a["end_fid"], "end_fid": b["begin_fid"]}
                for a, b in zip(components, components[1:])]
        for index, (previous, current) in enumerate(zip(components, components[1:])):
            # A donor for replacing the previous event must be earlier than it.
            # Never use the current target event or future to propose a donor.
            donor_gaps = [g for g in gaps[:index] if g["end_fid"] <= previous["begin_fid"]]
            old_length = previous["end_fid"] - previous["begin_fid"]
            matched = [dict(g, replacement_frames=old_length,
                            latest_length_matched_begin_fid=g["end_fid"] - old_length)
                       for g in donor_gaps if g["end_fid"] - g["begin_fid"] >= old_length]
            transitions.append({
                "video_name": video, "subject_tid": sid, "predicate": predicate,
                "object_tid": oid, "fps": fps, "previous": previous, "current": current,
                "gap": dict(gaps[index], seconds=(current["begin_fid"] - previous["end_fid"]) / fps),
                "current_has_primary_cohort_gt": any(by_id[g]["primary_cohort"] for g in current["gt_indices"]),
                "previous_has_primary_cohort_gt": any(by_id[g]["primary_cohort"] for g in previous["gt_indices"]),
                "earlier_length_matched_unlabeled_donor_candidates": matched,
                "joint_support_in_gap_and_donors": None,
                "joint_support_status": "UNKNOWN_REQUIRES_ANNOTATION_TRAJECTORIES",
                "visual_status": "PENDING_NOT_TRUE_OFF_LABELS",
            })
    return {"selection": "All same-video/subject/predicate/object adjacent merged annotation components; no model failure filter",
            "interval_convention": "[begin_fid,end_fid)",
            "overlap_and_touching": "Merged, never treated as independent end/restart",
            "donor_policy": "Only earlier annotation gaps ending before the previous event; length sufficient to replace that previous event; geometry and semantic neutrality UNVERIFIED",
            "N_triplet_keys": len(grouped), "N_transition_candidates": len(transitions),
            "N_videos": len({r["video_name"] for r in transitions}),
            "N_with_length_matched_earlier_gap": sum(bool(r["earlier_length_matched_unlabeled_donor_candidates"]) for r in transitions),
            "N_current_has_primary_cohort_gt": sum(r["current_has_primary_cohort_gt"] for r in transitions),
            "transitions": transitions}


def tally(rows):
    def known_count(field, desired=True):
        known = [r[field] for r in rows if r.get(field) is not None]
        return {"count": sum(v is desired for v in known), "known": len(known),
                "unknown": len(rows) - len(known)}
    covered = [r for r in rows if r["cascade_stage"] is not None]
    return {"N_GT": len(rows), "N_videos": len({r["video_name"] for r in rows}),
            "C": known_count("is_C"), "F": known_count("is_F"),
            "C_and_F": known_count("is_CF"),
            "VrdONE_hits": known_count("vrdone_hit"), "BIG_hits": known_count("big_hit"),
            "groups": dict(Counter(r["group"] for r in rows)),
            "cascade_coverage": fraction(len(covered), len(rows)),
            "cascade_cohorts": dict(Counter(r["cascade_cohort"] for r in covered)),
            "cascade_stages_current_version": dict(Counter(r["cascade_stage"] for r in covered)),
            "cascade_stages_056_original": dict(Counter(r["cascade_stage_056"] for r in covered)),
            "cascade_missing_N": len(rows) - len(covered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Refusing nonempty output directory; choose a fresh output path")
    paths = {name: locate(args.sources, path) for name, path in SOURCE_PATHS.items()}
    if paths["durations"] is None:
        raise SystemExit("Mandatory per_gt_durations.csv missing; population cannot be defined")
    duration = load_csv(paths["durations"])
    features = load_csv(paths["features013"]) if paths["features013"] else {}
    cascade_original = load_csv(paths["cascade056"]) if paths["cascade056"] else {}
    correction = json.loads(paths["correction057"].read_text()) if paths["correction057"] else None
    cascade = patch_stages(cascade_original, correction or {})
    if correction and not cascade_original:
        raise SystemExit("057 corrections supplied without 056 baseline cascade")
    for label, source in [("013", features), ("056", cascade)]:
        if set(source) - set(duration):
            raise ValueError(f"{label} has GT keys absent from population")
    conflicts = Counter()
    rows = []
    for k, raw in sorted(duration.items()):
        feature, stage = features.get(k), cascade.get(k)
        if feature:
            fields = [("subject_tid", "sid", int), ("object_tid", "oid", int),
                      ("begin_fid", "s", int), ("end_fid", "e", int), ("predicate", "predicate", str),
                      ("group", "group", str), ("F", "F", optional_bool), ("C", "C", optional_bool)]
            for a, b, cast in fields:
                if cast(raw[a]) != cast(feature[b]):
                    raise ValueError(f"013 identity/source disagreement {k}: {a}/{b}")
        if stage and stage["original_group"] != raw["group"]:
            raise ValueError(f"056 group disagreement at {k}")
        short = float(raw["relation_sec"]) < 3
        joint = optional_float(raw["joint_local_continuous_sec"])
        is_c, is_f = optional_bool(raw.get("C")), optional_bool(raw.get("F"))
        if is_c is not None and is_c != (raw["group"] == "C"):
            raise ValueError(f"Internal C/group disagreement at {k}")
        row = dict(raw, primary_cohort=short and joint is not None and joint >= 10,
                   short_relation=short, joint_support_unknown=joint is None,
                   is_C=is_c, is_F=is_f, is_CF=None if is_c is None or is_f is None else is_c and is_f,
                   vrdone_hit=optional_bool(raw.get("VrdONE_hit")), big_hit=optional_bool(raw.get("BIG_hit")),
                   features013_present=feature is not None,
                   repeat_count013=feature.get("repeat_count") if feature else None,
                   later_episode013=optional_bool(feature.get("later_episode")) if feature else None,
                   cascade_cohort=stage["cohort"] if stage else None,
                   cascade_stage=stage["stage"] if stage else None,
                   cascade_stage_056=stage["stage_056"] if stage else None,
                   cascade_grid_feasible_ANY=optional_bool(stage["grid_feasible_ANY"]) if stage else None,
                   cascade_baseline_hit100=optional_bool(stage["baseline_hit100"]) if stage else None)
        if stage:
            conflicts["F_vs_cascade_input_feasible"] += is_f != optional_bool(stage["input_feasible"])
            conflicts["original_hit_vs_cascade_baseline"] += row["vrdone_hit"] != row["cascade_baseline_hit100"]
        rows.append(row)
    target = [r for r in rows if r["primary_cohort"]]
    cohorts = {"all_GT": rows, "short_relation": [r for r in rows if r["short_relation"]],
               "primary_cohort": target, "primary_C": [r for r in target if r["is_C"] is True],
               "primary_F": [r for r in target if r["is_F"] is True],
               "primary_CF": [r for r in target if r["is_CF"] is True]}
    summary = {"status": "COMPLETE_DESCRIPTIVE_JOIN_NO_MODEL_RUN", "evidence_type": "Observation and prior Diagnosis",
               "definition": {"relation_seconds": "<3", "joint_local_continuous_seconds": ">=10",
                              "denominator": "Relation instances in supplied VidOR validation duration table",
                              "C": "Original BIG and VrdONE both miss at Top100",
                              "F": "Previously audited VrdONE input-pair geometry feasible; not actual prediction",
                              "joint_support": "GT-annotated continuous joint support including generated boxes; not verified visibility"},
               "cohorts": {name: tally(rr) for name, rr in cohorts.items()},
               "coverage_by_batch": {b: {name: tally([r for r in rr if r["cascade_cohort"] == b])
                                          for name, rr in cohorts.items()} for b in ["047", "048"]},
               "primary_fraction_all": fraction(len(target), len(rows)),
               "joint_support_unknown_N": sum(r["joint_support_unknown"] for r in rows),
               "features013_missing_N": sum(not r["features013_present"] for r in rows),
               "source_disagreements": dict(conflicts),
               "cascade_version": correction.get("version") if correction else "056_original_if_available",
               "correction_count": len(correction["changes"]) if correction else None,
               "limitations": ["Prior 047/048 exposed validation subset; no full-validation stage extrapolation",
                               "Mutually exclusive sequential diagnosis buckets are not causal module contributions",
                               "Oracle feasibility is opportunity, not simultaneous recovery or metric gain",
                               "No new model predictions, no method effect, no causal history evidence",
                               "Annotation absence in gaps is not visually verified relation termination"]}
    if correction:
        observed = summary["coverage_by_batch"]
        for batch in ["047", "048"]:
            expected = correction["cohorts"][batch]["all"]
            actual = observed[batch]["all_GT"]
            if expected["N_GT"] != actual["N_GT"] or expected["counts"] != actual["cascade_stages_current_version"]:
                raise ValueError(f"057 aggregate mismatch in {batch}")
    history = structural_history(rows)
    summary["history_candidates"] = {k: v for k, v in history.items() if k != "transitions"}
    manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sources": {name: {"status": "present" if path else "missing",
                                   "path": str(path) if path else None,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                                   "bytes": path.stat().st_size if path else None}
                            for name, path in paths.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in [("summary.json", summary), ("history_events.json", history), ("source_manifest.json", manifest)]:
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "per_gt_cohort.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "all_GT": len(rows), "primary": tally(target),
                      "history_candidates": summary["history_candidates"], "out": str(args.out.resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
