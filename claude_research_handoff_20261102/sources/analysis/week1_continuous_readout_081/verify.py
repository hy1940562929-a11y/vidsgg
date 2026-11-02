#!/usr/bin/env python3
"""Independent saved-artifact checks; no model execution or new inference."""
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify():
    output = HERE/"results_v1"
    manifest = json.loads((output/"manifest.json").read_text())
    source = read(manifest["input_path"])
    assert sha(manifest["input_path"]) == manifest["input_sha256"]
    assert sha(manifest["controlled_path"]) == manifest["controlled_sha256"]
    for filename, expected in manifest["outputs"].items():
        assert sha(output/filename) == expected, filename
    original = {(r["video_name"], r["gt_index"]): r for r in source}
    checked = []
    for filename in sorted(output.glob("H5_*.json.gz")):
        with gzip.open(filename, "rt") as handle:
            result = json.load(handle)
        groups = collections.defaultdict(lambda: [[], []])
        for r in source:
            if result["population"] != "all" and r["chosen_pair_covers_gt"] != "True":
                continue
            duration, coexist = float(r["relation_sec"]), float(r["coexist_sec"])
            if not (0 < duration < 3 and coexist >= 10 and coexist + 1e-6 >= duration):
                continue
            side = int(json.loads(r["competition_flags"])["overlap_other_pred_ge3"])
            key = (r["predicate"], int(duration/.25), sum(coexist >= e for e in (3, 10, 20)))
            if result["load_control"]:
                key += (sum(int(r["video_GT_load"]) >= e for e in (50, 100, 200)),)
            groups[key][side].append(r)
        common = [sides for sides in groups.values() if sides[0] and sides[1]]
        n = sum(len(a)+len(b) for a, b in common)
        means = [sum((len(a)+len(b))*sum(float(r["tiou"]) for r in (a, b)[side])/len((a, b)[side])
                     for a, b in common)/n for side in (0, 1)]
        assert n == result["N_common_support"]
        for side in (0, 1):
            assert abs(means[side] - result[f"standardized_side{side}"]) < 1e-12
        expected_members = {(r["video_name"], r["gt_index"], side) for sides in common for side in (0, 1) for r in sides[side]}
        actual_members = {(str(r["video_name"]), str(r["gt_index"]), r["side"]) for r in result["membership"]}
        assert expected_members == actual_members
        checked.append(dict(filename=filename.name, N=n, gap=means[1]-means[0], point_recomputed=True, members_exact=True))
    partition = read(output/"per_gt_partition.csv")
    assert len(partition) == len(original) == 2573
    counts = collections.Counter()
    for r in partition:
        raw = original[(r["video_name"], r["gt_index"])]
        selected, oracle = float(raw["tiou"]) >= .5, float(raw["oracle_tiou"]) >= .5
        expected = "selected_hit" if selected else "oracle_only_hit" if oracle else "neither"
        assert r["readout_state"] == expected
        counts[expected] += 1
    tables = read(output/"decomposition.csv")
    for population in ("all", "chosen_pair_covers_gt"):
        rows = partition if population == "all" else [r for r in partition if r["chosen_pair_covers_gt"] == "True"]
        for view in {r["view"] for r in tables if r["population"] == population}:
            group_rows = [r for r in tables if r["population"] == population and r["view"] == view]
            assert sum(int(r["N"]) for r in group_rows) == len(rows)
            for state in ("selected_hit", "oracle_only_hit", "neither"):
                assert sum(int(r[state+"_count"]) for r in group_rows) == sum(r["readout_state"] == state for r in rows)
            for r in group_rows:
                for state in ("selected_hit", "oracle_only_hit", "neither"):
                    assert int(r[state+"_denominator"]) == int(r["N"])
                    assert math.isclose(float(r[state+"_rate"]), int(r[state+"_count"])/int(r["N"]), abs_tol=1e-12)
    receipt = dict(passed=True, statistic_checks=checked, partition_counts=dict(counts),
                   all_output_hashes_verified=True, model_or_training_execution=False,
                   validation_scope="independent point estimates, exact membership, outcome partition, aggregate balance; saved bootstrap CIs were not independently redrawn")
    (HERE/"verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
