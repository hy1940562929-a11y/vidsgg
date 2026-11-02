#!/usr/bin/env python3
"""Frozen descriptive controls for relation duration and entity co-presence.

No model inference, no fitting to outcomes, no pixel/occlusion labels are invented.
Only NumPy plus the Python standard library are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEED = 20261015
OUTCOMES = ("VrdONE_miss", "C_miss", "VrdONE_hit", "BIG_hit")
SIZE_EDGES = (.01, .05)
SPEED_EDGES = (.02, .10)
DURATION_FINE_EDGES = tuple(i / 4 for i in range(1, 13))
COEXIST_EDGES = (3., 10., 20.)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def bin_value(value, edges):
    if value is None:
        return "MISSING"
    return str(sum(value >= e for e in edges))


def key(row):
    return str(row["video_name"]), str(int(row["gt_index"]))


def first(row, names, required=True):
    for n in names:
        if n in row and row[n] != "":
            return row[n]
    if required:
        raise KeyError(f"None of {names} in row; fields={list(row)}")
    return None


def to_bool(value):
    return str(value).lower() in ("1", "true", "yes")


def build_rows(features_path, durations_path, coexist_field=None):
    features, durations = read_csv(features_path), read_csv(durations_path)
    fi, di = {key(r): r for r in features}, {key(r): r for r in durations}
    assert len(fi) == len(features), "duplicate feature keys"
    assert len(di) == len(durations), "duplicate duration keys"
    assert set(fi) == set(di), "feature and duration GT universes differ"
    out = []
    used_coexist_fields = Counter()
    for k in sorted(fi, key=lambda k: (k[0], int(k[1]))):
        f, d = fi[k], di[k]
        # These are two independent exports; never silently override a conflict.
        for left, right in (("predicate", "predicate"), ("subject_tid", "sid"),
                            ("object_tid", "oid"), ("begin_fid", "s"), ("end_fid", "e"),
                            ("subject_class", "sc"), ("object_class", "oc"), ("group", "group")):
            assert str(d[left]) == str(f[right]), (k, left, d[left], f[right])
        assert to_bool(d["F"]) == to_bool(f["F"]), (k, "F disagreement")
        assert to_bool(d["C"]) == to_bool(f["C"]), (k, "C disagreement")
        merged = {**d, **f}
        duration = float(d["relation_sec"])
        assert abs(float(f["d"]) - duration) < 1e-6
        names = (coexist_field,) if coexist_field else (
            "joint_local_continuous_sec",
            "joint_contiguous_support_sec", "joint_contiguous_duration_sec",
            "joint_continuous_support_sec", "pair_contiguous_support_sec",
            "pair_joint_contiguous_sec", "joint_component_duration_sec",
        )
        present = [n for n in names if n in d]
        if not present:
            raise KeyError("Need GT joint CONTINUOUS support duration containing the relation. "
                           "Do not use entity marginal lifetimes or relation-annotation span. "
                           f"Duration fields: {list(d)}; set --coexist-field after inspecting provenance.")
        field = present[0]
        coexist = finite(d[field])
        used_coexist_fields[field] += 1
        if coexist is not None:
            assert coexist >= duration - 1e-6, ("joint support shorter than GT", k, coexist, duration)
        v = int(d["VrdONE_hit"])
        b = int(d["BIG_hit"])
        c = int(not v and not b)
        expected_group = {(1, 1): "A", (1, 0): "B", (0, 0): "C", (0, 1): "D"}[(v, b)]
        assert merged["group"] == expected_group, ("group and hits disagree", k)
        feasible_raw = first(merged, ("input_pair_state", "input_state", "input_feasible", "feasible_input_pair", "F"))
        feasible = feasible_raw == "feasible_input_pair" or to_bool(feasible_raw)
        area = finite(first(merged, ("min_entity_area_fraction",), required=False))
        speed = finite(first(merged, ("relative_speed_per_second",), required=False))
        # 013 has already-frozen bins, not underlying continuous measurements.
        # Preserve these observed bins; never impute bin midpoints as measurements.
        size_bin = {"<1%": "0", "1-5%": "1", ">=5%": "2", "NA": "MISSING"}[f["size"]]
        speed_bin = {"<.02": "0", ".02-.1": "1", ">=.1": "2", "NA": "MISSING"}[f["motion"]]
        r = {"video_id": f["video_id"], "gt_index": int(k[1]), "video_name": k[0],
             "subject_tid": int(merged["subject_tid"]), "object_tid": int(merged["object_tid"]),
             "subject_class": merged["subject_class"], "predicate": merged["predicate"],
             "object_class": merged["object_class"], "duration_sec": duration,
             "joint_contiguous_support_sec": coexist, "input_feasible": int(feasible),
             "group_at100": expected_group, "VrdONE_hit": v, "VrdONE_miss": 1-v, "BIG_hit": b, "C_miss": c,
             "min_entity_area_fraction": area, "relative_speed_per_second": speed,
             "size_bin": size_bin, "speed_bin": speed_bin,
             "size_source_bin": f["size"], "speed_source_bin": f["motion"],
             "relation_frames_without_both_GT_boxes": int(d["relation_frames_without_both_GT_boxes"]),
             "duration_fine_bin": bin_value(duration, DURATION_FINE_EDGES),
             "coexist_bin": bin_value(coexist, COEXIST_EDGES),
             "duration_bin": bin_value(duration, (1., 3., 10.)),
             "short": int(duration < 3.),
             "long_coexist": int(coexist >= 10.) if coexist is not None else None,
             "occlusion_label": "UNOBSERVED", "visual_ambiguity_label": "UNOBSERVED"}
        out.append(r)
    return out, dict(used_coexist_fields)


def designs():
    return [
        {"name": "duration_predicate", "question": "duration", "fields": ["predicate"], "role": "supporting"},
        {"name": "duration_triplet", "question": "duration", "fields": ["subject_class", "predicate", "object_class"], "role": "supporting"},
        {"name": "duration_predicate_size_motion", "question": "duration", "fields": ["predicate", "size_bin", "speed_bin"], "role": "supporting"},
        {"name": "duration_triplet_size_motion_coexist", "question": "duration", "fields": ["subject_class", "predicate", "object_class", "size_bin", "speed_bin", "coexist_bin"], "role": "primary"},
        {"name": "duration_same_video_pair_predicate", "question": "duration", "fields": ["video_id", "subject_tid", "object_tid", "predicate"], "role": "strict_sensitivity"},
        {"name": "duration_long_coexist_predicate_size_motion", "question": "duration_long_coexist", "fields": ["predicate", "size_bin", "speed_bin"], "role": "exploratory"},
        {"name": "coexist_short_predicate", "question": "coexist", "fields": ["predicate"], "role": "supporting"},
        {"name": "coexist_short_predicate_duration", "question": "coexist", "fields": ["predicate", "duration_fine_bin"], "role": "supporting"},
        {"name": "coexist_short_predicate_duration_size_motion", "question": "coexist", "fields": ["predicate", "duration_fine_bin", "size_bin", "speed_bin"], "role": "primary"},
        {"name": "coexist_short_triplet_duration_size_motion", "question": "coexist", "fields": ["subject_class", "predicate", "object_class", "duration_fine_bin", "size_bin", "speed_bin"], "role": "exploratory"},
        {"name": "coexist_short_same_video_pair_predicate", "question": "coexist", "fields": ["video_id", "subject_tid", "object_tid", "predicate", "duration_fine_bin"], "role": "strict_sensitivity"},
    ]


def population(rows, cohort, question):
    result = [r for r in rows if cohort == "all" or r["input_feasible"]]
    if question == "coexist":
        result = [r for r in result if r["short"] and r["long_coexist"] is not None]
    elif question == "duration_long_coexist":
        result = [r for r in result if r["long_coexist"] == 1]
    return result


def rates(rows, exposure):
    res = {}
    for side in (0, 1):
        rr = [r for r in rows if r[exposure] == side]
        for outcome in OUTCOMES:
            num = sum(r[outcome] for r in rr)
            res[f"{outcome}_{side}_numerator"] = num
            res[f"{outcome}_{side}_denominator"] = len(rr)
            res[f"{outcome}_{side}_pct"] = 100 * num / len(rr) if rr else None
    for outcome in OUTCOMES:
        x, y = (res[f"{outcome}_{s}_pct"] for s in (0, 1))
        res[f"{outcome}_gap_pp"] = y - x if x is not None and y is not None else None
    return res


def bootstrap_strata(rows, strata_ids, exposure, n_strata, replicates, seed):
    """Positive video-weight (Exp(1)) cluster bootstrap on fixed support.

    Each video's GTs share one positive multiplier. All original eligible
    strata and both sides remain; rates and pooled weights are re-estimated.
    This avoids resampling-away sparse strata and changing the comparison
    population. Intervals are conditional descriptive uncertainty, not causal
    intervals or a claim that every sparse stratum is measured precisely.
    """
    video_names = sorted({r["video_id"] for r in rows})
    vi = {v: i for i, v in enumerate(video_names)}
    video_idx = np.asarray([vi[r["video_id"]] for r in rows])
    side = np.asarray([r[exposure] for r in rows], dtype=np.int64)
    cell = 2 * np.asarray(strata_ids, dtype=np.int64) + side
    yy = np.asarray([[r[o] for o in OUTCOMES] for r in rows], dtype=float)
    point_n = np.bincount(cell, minlength=2 * n_strata).reshape(-1, 2)
    point_mass = point_n.sum(axis=1)
    rng = np.random.default_rng(seed)
    estimates, retained, draw_retained = [], [], []
    for _ in range(replicates):
        multiplicity = rng.exponential(1., len(video_names))
        # The chance of exact zero is negligible, but strict positivity is a
        # mathematical requirement of the fixed-support bootstrap definition.
        multiplicity = np.maximum(multiplicity, np.finfo(float).tiny)
        weights = multiplicity[video_idx]
        nn = np.bincount(cell, weights=weights, minlength=2 * n_strata).reshape(-1, 2)
        keep = np.all(nn > 0, axis=1)
        assert np.all(keep), "Positive video weights must preserve common support"
        pooled = nn[keep].sum(axis=1)
        pooled = pooled / pooled.sum()
        rr = []
        for j in range(len(OUTCOMES)):
            sums = np.bincount(cell, weights=weights * yy[:, j], minlength=2 * n_strata).reshape(-1, 2)
            standardized = (pooled[:, None] * sums[keep] / nn[keep]).sum(axis=0)
            rr.append(100 * (standardized[1] - standardized[0]))
        estimates.append(rr)
        retained.append(point_mass[keep].sum() / point_mass.sum())
        draw_retained.append(nn[keep].sum() / nn.sum())
    result = {"method": "positive_video_multiplier_Exp(1)_cluster_bootstrap",
              "replicates_requested": replicates, "replicates_valid": len(estimates),
              "seed": seed, "N_videos": len(video_names),
              "retained_original_support_mass_mean": float(np.mean(retained)) if retained else None,
              "retained_original_support_mass_p05": float(np.quantile(retained, .05)) if retained else None,
              "retained_resampled_support_mass_mean": float(np.mean(draw_retained)) if draw_retained else None,
              "retained_resampled_support_mass_p05": float(np.quantile(draw_retained, .05)) if draw_retained else None}
    for j, outcome in enumerate(OUTCOMES):
        vals = np.asarray(estimates)[:, j] if estimates else np.asarray([])
        result[outcome] = {"lower_pp": float(np.quantile(vals, .025)) if len(vals) else None,
                           "upper_pp": float(np.quantile(vals, .975)) if len(vals) else None}
    return result


def effect_judgment(gap, lo, hi, videos, support, total, retention, threshold):
    if lo is None or videos < 20 or support < 60 or retention is None or retention < .80:
        return "INCONCLUSIVE_SUPPORT_OR_BOOTSTRAP_STABILITY"
    if gap >= threshold and lo > 0:
        return "SUPPORTS_PRESET_DESCRIPTIVE_ASSOCIATION"
    if hi < threshold:
        return "DOES_NOT_SUPPORT_PRESET_AT_LEAST_3PP_EFFECT"
    return "INCONCLUSIVE_PRECISION"


def compare(rows, cohort, design, replicates, seed, minimum_each_side, threshold):
    pop = population(rows, cohort, design["question"])
    exposure = "long_coexist" if design["question"] == "coexist" else "short"
    fields = design["fields"]
    eligible = [r for r in pop if all(r[f] not in (None, "MISSING", "") for f in fields)]
    strata = defaultdict(lambda: [[], []])
    for r in eligible:
        strata[tuple(r[f] for f in fields)][r[exposure]].append(r)
    valid = {k: v for k, v in strata.items() if min(map(len, v)) >= minimum_each_side}
    supported, ids, details = [], [], []
    for i, (k, sides) in enumerate(sorted(valid.items())):
        for side in (0, 1):
            supported.extend(sides[side]); ids.extend([i] * len(sides[side]))
        rec = {"cohort": cohort, "design": design["name"], "stratum_key": json.dumps(k, ensure_ascii=False),
               "N_0": len(sides[0]), "N_1": len(sides[1])}
        for o in OUTCOMES:
            for side in (0, 1):
                rec[f"{o}_{side}_numerator"] = sum(r[o] for r in sides[side])
        details.append(rec)
    n = len(supported)
    result = {"cohort": cohort, **design, "exposure": exposure,
              "side_1": "GT relation <3s" if exposure == "short" else "joint continuous support >=10s",
              "side_0": "GT relation >=3s" if exposure == "short" else "joint continuous support <10s",
              "N_population": len(pop), "N_complete_covariates": len(eligible),
              "N_missing_covariates": len(pop) - len(eligible), "N_common_support": n,
              "N_strata_common_support": len(valid), "N_videos_support": len({r["video_id"] for r in supported}),
              "support_fraction_numerator": n, "support_fraction_denominator": len(pop),
              "support_pct": 100 * n / len(pop) if pop else None,
              "raw_population": rates(pop, exposure), "raw_complete_covariates": rates(eligible, exposure),
              "raw_same_support": rates(supported, exposure), "standardized": {}}
    if not n:
        result["judgment"] = "INCONCLUSIVE_NO_COMMON_SUPPORT"
        return result, details, []
    for d in details:
        d["pooled_weight_numerator"] = d["N_0"] + d["N_1"]
        d["pooled_weight_denominator"] = n
    for o in OUTCOMES:
        adj = []
        for side in (0, 1):
            numerator = sum(d["pooled_weight_numerator"] * d[f"{o}_{side}_numerator"] / d[f"N_{side}"] for d in details)
            adj.append(100 * numerator / n)
            result["standardized"][f"{o}_{side}_weighted_numerator"] = numerator
            result["standardized"][f"{o}_{side}_weight_denominator"] = n
            result["standardized"][f"{o}_{side}_pct"] = adj[-1]
        result["standardized"][f"{o}_gap_pp"] = adj[1] - adj[0]
    boot = bootstrap_strata(supported, ids, exposure, len(valid), replicates, seed)
    result["bootstrap"] = boot
    result["judgment"] = effect_judgment(
        result["standardized"]["VrdONE_miss_gap_pp"], boot["VrdONE_miss"]["lower_pp"], boot["VrdONE_miss"]["upper_pp"],
        result["N_videos_support"], n, len(pop), boot["retained_resampled_support_mass_p05"], threshold)
    membership = [{"video_id": r["video_id"], "gt_index": r["gt_index"], "cohort": cohort,
                   "design": design["name"], "side": r[exposure], "stratum_id": s}
                  for r, s in zip(supported, ids)]
    return result, details, membership


def flatten_summary(result):
    flat = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
    flat["fields"] = "|".join(result["fields"])
    for name in ("raw_population", "raw_complete_covariates", "raw_same_support", "standardized"):
        for k, v in result.get(name, {}).items():
            flat[f"{name}__{k}"] = v
    for o in OUTCOMES:
        for k, v in result.get("bootstrap", {}).get(o, {}).items():
            flat[f"bootstrap__{o}__{k}"] = v
    return flat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--features", type=Path)
    p.add_argument("--durations", type=Path)
    p.add_argument("--coexist-field", help="Verified GT joint continuous component duration field")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--minimum-each-side", type=int, default=1)
    p.add_argument("--expected-gt", type=int, default=30142)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    assert args.bootstrap >= 200, "Use >=200 only for smoke; default 2000 for reporting"
    assert not args.out.exists(), "Choose a fresh output directory; do not overwrite evidence"
    fp = args.features or args.root / "analysis/event_support_073/inputs/per_gt_features.csv"
    dp = args.durations or args.root / "analysis/event_support_073/inputs/per_gt_durations.csv"
    rows, mapped = build_rows(fp, dp, args.coexist_field)
    assert len(rows) == args.expected_gt
    if args.expected_gt == 30142:
        assert Counter(r["group_at100"] for r in rows) == {"A": 2487, "B": 1809, "C": 25373, "D": 473}
        assert sum(r["input_feasible"] for r in rows) == 13294
    args.out.mkdir(parents=True)
    results, details, members = [], [], []
    for ci, cohort in enumerate(("all", "F")):
        for j, design in enumerate(designs()):
            result, dd, mm = compare(rows, cohort, design, args.bootstrap, args.seed + ci * 100 + j, args.minimum_each_side, 3.)
            results.append(result); details.extend(dd); members.extend(mm)
            print(json.dumps({"cohort": cohort, "design": design["name"], "N_support": result["N_common_support"],
                              "VrdONE_miss_gap_pp": result["standardized"].get("VrdONE_miss_gap_pp"), "judgment": result["judgment"]}), flush=True)
    write_csv(args.out / "per_gt_controls.csv", rows)
    write_csv(args.out / "comparisons.csv", [flatten_summary(r) for r in results])
    write_csv(args.out / "strata.csv", details)
    write_csv(args.out / "per_gt_support_membership.csv", members)
    predicate_rows = []
    for cohort in ("all", "F"):
        pp = population(rows, cohort, "duration")
        for pred in sorted({r["predicate"] for r in pp}):
            for dur in range(4):
                rr = [r for r in pp if r["predicate"] == pred and r["duration_bin"] == str(dur)]
                if rr:
                    predicate_rows.append({"cohort": cohort, "predicate": pred, "duration_bin": dur,
                                           "N_GT": len(rr), **{o + "_count": sum(r[o] for r in rr) for o in OUTCOMES}})
    write_csv(args.out / "predicate_duration_counts.csv", predicate_rows)
    metadata = {"complete": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "argv": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_files": [{"path": str(x), "sha256": digest(x)} for x in (fp, dp, Path(__file__))],
                "coexist_field_mapping": mapped, "N_GT": len(rows), "N_F": sum(r["input_feasible"] for r in rows),
                "N_videos": len({r["video_id"] for r in rows}), "group_counts": dict(Counter(r["group_at100"] for r in rows)),
                "definitions": {"F": "Existing VrdONE proposal pair geometry oracle feasibility, not perfect input and not BIG feasibility",
                                "outcomes": "Published/saved Top100 GT hit flags; each model separately and joint miss C",
                                "weights": "Same pooled GT stratum weights for both sides, conditional on common support",
                                "difference": "side 1 minus side 0; positive C is worse, negative hit difference is worse",
                                "weighted_numerator": "Fractional standardized count, not actual recovered GT instances",
                                "bootstrap": "Positive Exp(1) video-weight cluster bootstrap; same positive weight for all GT in a video, fixed common-support strata, re-estimate rates and pooled weights",
                                "primary_population": "F; all-GT is a prespecified companion analysis, not a second chance for the primary decision",
                                "primary_decision": "F VrdONE_miss gap >=3pp and 95% bootstrap lower >0; otherwise distinguish effect upper<3 from imprecision",
                                "covariate_source": "013 frozen categorical size/motion bins only; underlying continuous values unavailable in these inputs",
                                "size_edges": SIZE_EDGES, "speed_edges": SPEED_EDGES,
                                "duration_fine_edges": DURATION_FINE_EDGES, "coexist_edges": COEXIST_EDGES},
                "limitations": ["Descriptive conditional association, not intervention or causal effect",
                                "No pixel-based occlusion, visibility or boundary-ambiguity control",
                                "F is selected using VrdONE input feasibility and is not a common-input comparison of models",
                                "Prior validation predictions reused; no new independent generalization result",
                                "Bootstrap intervals are marginal; supporting/exploratory designs are not multiplicity-adjusted confirmatory claims",
                                "Coarse bins leave within-bin confounding; comparing different design gaps is not a causal explained fraction",
                                "Per-GT model hits do not establish event boundary mechanism, and duration-sensitive vIoU remains a possible explanation"],
                "comparisons": results}
    (args.out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.out / "output_manifest.json").write_text(json.dumps({x.name: digest(x) for x in sorted(args.out.iterdir()) if x.is_file()}, indent=2) + "\n")


def self_test():
    # Simpson example: raw difference is positive while common-stratum difference is zero.
    rows = []
    for video in range(30):
        for pred, n0, n1, fail in (("hard", 1, 9, 1), ("easy", 9, 1, 0)):
            for side, nn in ((0, n0), (1, n1)):
                for j in range(nn):
                    rows.append({"video_id": str(video), "gt_index": len(rows), "predicate": pred,
                                 "short": side, "input_feasible": 1, "C_miss": fail,
                                 "VrdONE_hit": 1-fail, "BIG_hit": 1-fail})
                    rows[-1]["VrdONE_miss"] = fail
    design = {"name": "synthetic", "question": "duration", "fields": ["predicate"], "role": "test"}
    r, detail, members = compare(rows, "all", design, 200, SEED, 1, 3.)
    assert r["raw_population"]["C_miss_gap_pp"] == 80
    assert abs(r["standardized"]["C_miss_gap_pp"]) < 1e-12
    assert r["bootstrap"]["C_miss"]["lower_pp"] == 0
    assert r["bootstrap"]["C_miss"]["upper_pp"] == 0
    assert len(members) == 600 and sum(d["pooled_weight_numerator"] for d in detail) == 600
    assert effect_judgment(4, 1, 6, 30, 600, 600, 1, 3) == "SUPPORTS_PRESET_DESCRIPTIVE_ASSOCIATION"
    assert effect_judgment(0, -5, 5, 30, 600, 600, 1, 3) == "INCONCLUSIVE_PRECISION"
    assert effect_judgment(0, -1, 1, 30, 600, 600, 1, 3) == "DOES_NOT_SUPPORT_PRESET_AT_LEAST_3PP_EFFECT"
    print("SELF_TEST_PASS: common-support weights, Simpson control, video bootstrap, uncertainty judgment")


if __name__ == "__main__":
    main()
