#!/usr/bin/env python3
"""Independently verify frozen 077 controls and apply the pre-frozen support rule.

Reads saved counts/membership and never imports controls.py, recomputes a model,
or changes the original summary. Bootstrap limits are verified as finite and
ordered but are not re-estimated here. Standardized point estimates are rebuilt
with exact rational arithmetic from strata.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


OUTCOMES = ("VrdONE_miss", "C_miss", "VrdONE_hit", "BIG_hit")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError((label, actual, expected))


def support_judgment(gap, lo, hi, n, videos_by_side, retention):
    if n < 60 or min(videos_by_side) < 20 or retention is None or retention < .80:
        return "INCONCLUSIVE_SUPPORT_OR_BOOTSTRAP_STABILITY"
    if gap >= 3. and lo > 0:
        return "SUPPORTS_PRESET_DESCRIPTIVE_ASSOCIATION"
    if hi < 3.:
        return "DOES_NOT_SUPPORT_PRESET_AT_LEAST_3PP_EFFECT"
    return "INCONCLUSIVE_PRECISION"


def verify(results, out, protocol):
    assert not out.exists(), "Use a fresh review directory; do not overwrite prior review"
    before = {p.name: sha(p) for p in results.iterdir() if p.is_file()}
    manifest = json.loads((results / "output_manifest.json").read_text())
    for filename, expected in manifest.items():
        assert before[filename] == expected, ("manifest mismatch", filename)
    summary = json.loads((results / "summary.json").read_text())
    assert summary["complete"] is True
    strata = defaultdict(list)
    for row in read_csv(results / "strata.csv"):
        strata[row["cohort"], row["design"]].append(row)
    members = defaultdict(list)
    for row in read_csv(results / "per_gt_support_membership.csv"):
        members[row["cohort"], row["design"]].append(row)
    per_gt = read_csv(results / "per_gt_controls.csv")
    gt = {(r["video_id"], int(r["gt_index"])): r for r in per_gt}
    assert len(gt) == len(per_gt) == summary["N_GT"]
    records, errors, max_abs_diff = [], [], 0.
    for result in summary["comparisons"]:
        group = result["cohort"], result["name"]
        dd, mm = strata[group], members[group]
        n = sum(int(d["N_0"]) + int(d["N_1"]) for d in dd)
        assert n == result["N_common_support"] == len(mm), group
        assert len(dd) == result["N_strata_common_support"], group
        unique = {(m["video_id"], int(m["gt_index"])) for m in mm}
        assert len(unique) == len(mm), (group, "duplicate membership")
        video_sets = [{m["video_id"] for m in mm if m["side"] == str(s)} for s in (0, 1)]
        ns = [sum(m["side"] == str(s) for m in mm) for s in (0, 1)]
        assert len(video_sets[0] | video_sets[1]) == result["N_videos_support"], group
        cells, successes = Counter(), Counter()
        for m in mm:
            r = gt[m["video_id"], int(m["gt_index"])]
            side, sid = int(m["side"]), int(m["stratum_id"])
            assert str(r[result["exposure"]]) == str(side), (group, "wrong exposure", m)
            if result["cohort"] == "F":
                assert int(r["input_feasible"]) == 1
            actual_key = [str(r[k]) for k in result["fields"]]
            expected_key = [str(v) for v in json.loads(dd[sid]["stratum_key"])]
            assert actual_key == expected_key, (group, actual_key, expected_key)
            cells[sid, side] += 1
            for outcome in OUTCOMES:
                successes[sid, side, outcome] += int(r[outcome])
        for sid, d in enumerate(dd):
            assert int(d["pooled_weight_denominator"]) == n
            assert int(d["pooled_weight_numerator"]) == int(d["N_0"]) + int(d["N_1"])
            for side in (0, 1):
                assert cells[sid, side] == int(d[f"N_{side}"]), (group, sid, side)
                assert cells[sid, side] > 0
                for outcome in OUTCOMES:
                    assert successes[sid, side, outcome] == int(d[f"{outcome}_{side}_numerator"])
        independent, raw = {}, {}
        if n:
            for outcome in OUTCOMES:
                pct = []
                for side in (0, 1):
                    raw_numerator = sum(int(d[f"{outcome}_{side}_numerator"]) for d in dd)
                    weighted = sum((Fraction(int(d["N_0"]) + int(d["N_1"])) *
                                    Fraction(int(d[f"{outcome}_{side}_numerator"]), int(d[f"N_{side}"]))
                                    for d in dd), Fraction())
                    rate = float(weighted / n * 100)
                    pct.append(rate)
                    raw[f"{outcome}_{side}"] = {"numerator": raw_numerator, "denominator": ns[side],
                                                  "pct": 100 * raw_numerator / ns[side]}
                    independent[f"{outcome}_{side}"] = {"weighted_numerator": float(weighted),
                                                          "weight_denominator": n, "pct": rate,
                                                          "weighted_numerator_exact_fraction": str(weighted)}
                    original = result["standardized"][f"{outcome}_{side}_pct"]
                    max_abs_diff = max(max_abs_diff, abs(rate - original))
                    close(rate, original, (group, outcome, side))
                    assert raw_numerator == result["raw_same_support"][f"{outcome}_{side}_numerator"]
                    assert ns[side] == result["raw_same_support"][f"{outcome}_{side}_denominator"]
                gap = pct[1] - pct[0]
                independent[outcome + "_gap_pp"] = gap
                close(gap, result["standardized"][outcome + "_gap_pp"], (group, outcome, "gap"))
                limits = result["bootstrap"][outcome]
                assert all(math.isfinite(limits[k]) for k in ("lower_pp", "upper_pp"))
                assert limits["lower_pp"] <= limits["upper_pp"]
            ci = result["bootstrap"]["VrdONE_miss"]
            reviewed = support_judgment(independent["VrdONE_miss_gap_pp"], ci["lower_pp"], ci["upper_pp"],
                                         n, [len(v) for v in video_sets],
                                         result["bootstrap"]["retained_resampled_support_mass_p05"])
        else:
            ci = {"lower_pp": None, "upper_pp": None}
            reviewed = "INCONCLUSIVE_NO_COMMON_SUPPORT"
        rec = {"cohort": group[0], "name": group[1], "role": result["role"],
               "side_0": result["side_0"], "side_1": result["side_1"],
               "N_population": result["N_population"], "N_common_support": n,
               "N_side_0": ns[0], "N_side_1": ns[1],
               "N_videos_side_0": len(video_sets[0]), "N_videos_side_1": len(video_sets[1]),
               "N_videos_union": len(video_sets[0] | video_sets[1]),
               "N_videos_overlap": len(video_sets[0] & video_sets[1]),
               "raw_same_support": raw, "independently_recomputed_standardized": independent,
               "reported_video_weight_bootstrap_VrdONE_miss": ci,
               "original_judgment": result["judgment"], "reviewed_judgment": reviewed,
               "judgment_changed": reviewed != result["judgment"],
               "support_rule_pass": n >= 60 and min(map(len, video_sets)) >= 20}
        records.append(rec)
    assert {p.name: sha(p) for p in results.iterdir() if p.is_file()} == before, "Original result changed during verification"
    out.mkdir(parents=True)
    decisions = {"complete": True, "evidence_type": "Independent recomputation of existing results, no new experiment",
                 "support_rule": {"minimum_total_GT": 60, "minimum_independent_videos_each_side": 20,
                                  "timing": "Commander PROTOCOL.md froze this rule before first reading real effects; frozen controls.py incorrectly checked only union videos"},
                 "effect_rule": "VrdONE miss gap >=3pp and lower 95% video-weight bootstrap limit >0; F is primary",
                 "source_summary_sha256": before["summary.json"], "comparisons": records}
    (out / "reviewed_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
    validation = {"status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "N_comparisons": len(records), "N_primary_F": sum(r["role"] == "primary" and r["cohort"] == "F" for r in records),
                  "N_changed_judgments": sum(r["judgment_changed"] for r in records),
                  "max_absolute_standardized_rate_disagreement_pp": max_abs_diff,
                  "checks": ["Original output manifest SHA256 matches", "Unique GT membership per design",
                             "Membership exposure and exact stratum keys match per_gt_controls.csv",
                             "Membership cell counts and all 4 outcomes exactly match strata.csv",
                             "All standardized rates/differences recomputed from exact rational stratum weights",
                             "Raw same-support outcome numerators/denominators exactly match summary",
                             "Both-side independent video counts independently computed",
                             "Frozen original outputs unchanged after verification"],
                  "not_recomputed": ["Bootstrap draws and percentile limits (read saved limits; this is not an independent bootstrap re-run)",
                                     "Original model inference, input geometry feasibility or manual visibility labels"],
                  "source_directory": str(results.resolve()), "source_file_hashes": before,
                  "verification_script_sha256": sha(Path(__file__)),
                  "commander_protocol_path": str(protocol.resolve()), "commander_protocol_sha256": sha(protocol)}
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "N_changed_judgments": validation["N_changed_judgments"],
                      "primary_F": [r for r in records if r["role"] == "primary" and r["cohort"] == "F"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path(__file__).with_name("PROTOCOL.md"))
    args = p.parse_args()
    verify(args.results, args.out, args.protocol)
