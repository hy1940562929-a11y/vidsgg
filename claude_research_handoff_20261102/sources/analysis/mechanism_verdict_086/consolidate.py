"""086: deterministic intersection of existing 083/084 records; no new experiment."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
paths = [ROOT / f"analysis/{name}/results_v1/per_gt.csv" for name in
         ("week1_query_feasibility_083", "week1_component_readout_084")]
def read(p):
    with p.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    d = {(r["video_name"], r["gt_index"]): r for r in rows}
    assert len(d) == len(rows) == 2573
    return d
a, b = map(read, paths)
assert a.keys() == b.keys()
for k, r in a.items():
    for col in ("subject_tid", "object_tid", "predicate", "begin_fid", "end_fid",
                "chosen_pair_covers_gt"):
        assert r[col] == b[k][col]
    assert float(r["default_min5"]) == float(b[k]["default_hull"])
    assert float(r["raw_logit"]) == float(b[k]["raw_hull"])

groups = {
    "long_coexist_short_all": lambda r: float(r["relation_sec"]) < 3 and float(r["coexist_sec"]) >= 10,
    "long_coexist_short_covered": lambda r: float(r["relation_sec"]) < 3 and float(r["coexist_sec"]) >= 10 and r["chosen_pair_covers_gt"] == "True",
    "long_coexist_subsecond_covered": lambda r: float(r["relation_sec"]) < 1 and float(r["coexist_sec"]) >= 10 and r["chosen_pair_covers_gt"] == "True",
    "short_coexist_short_covered": lambda r: float(r["relation_sec"]) < 3 and float(r["coexist_sec"]) < 10 and r["chosen_pair_covers_gt"] == "True",
}
out = []
members = {}
for name, pred in groups.items():
    rr = [r for r in a.values() if pred(r)]
    bb = [b[r["video_name"], r["gt_index"]] for r in rr]
    c = Counter(r["stage_tiou05"] for r in rr)
    z = dict(group=name, N=len(rr), videos=len({r["video_name"] for r in rr}),
             selected_hits=sum(float(r["selected_tiou"]) >= .5 for r in rr),
             grid_feasible=sum(float(r["grid_min5"]) >= .5 for r in rr),
             default_hull_hits=c["default_hit"],
             threshold_hull_additional=c["sigmoid_threshold_reachable"]+c["raw_logit_only_reachable"],
             grid_feasible_hull_family_missing=c["grid_feasible_threshold_family_infeasible"])
    for prefix in ("default", "raw"):
        h = [float(r[f"{prefix}_hull"]) >= .5 for r in bb]
        v = [float(r[f"{prefix}_component"]) >= .5 for r in bb]
        z[f"{prefix}_component_hits"] = sum(v)
        z[f"{prefix}_component_gain"] = sum(y and not x for x,y in zip(h,v))
        z[f"{prefix}_component_loss"] = sum(x and not y for x,y in zip(h,v))
    out.append(z)
    members[name] = [(r["video_name"], r["gt_index"]) for r in rr]
with (OUT / "existing_evidence_intersections.csv").open("w", newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(OUT / "intersection_members.json").write_text(json.dumps(members,indent=2)+"\n")
manifest=dict(kind="POSTHOC_EXISTING_EVIDENCE_INTERSECTION_NOT_NEW_EXPERIMENT",
              threshold=.5, relation_short_seconds=3, long_coexist_seconds=10,
              oracle_scope="all same-identity segments and queries; GT temporal selection; predicate unconstrained",
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              rows=out)
(OUT / "intersection_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(out,indent=2))
