"""CPU-only summary of completed 085; does not change the registered endpoint."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "handoff/incoming/VIDOR-LONGSHORT-CONTEXT-085/remote_20261020"
OUT = Path(__file__).resolve().parent
rows = json.loads((RAW / "infer_v3/per_event.json").read_text())
summary = json.loads((RAW / "infer_v3/summary.json").read_text())
plan = json.loads((RAW / "plan_v3/selection.json").read_text())
assert summary["complete"] and len(rows) == plan["selected_count"]
records = []
for r in rows:
    z = dict(case_id=r["case_id"], video=r["video"], predicate=r["predicate"],
             long_predicate=r["long_predicate"], start=r["target"][0],
             end=r["target"][1], fps=r["fps"])
    for arm in ("baseline", "editL", "editN"):
        for k in ("target_mask_probability", "target_class_probability", "tiou",
                  "hit50", "long_L_mask_probability", "long_class_probability"):
            z[f"{arm}_{k}"] = r["arms"][arm][k]
    z["paired_mask_effect"] = r["paired_editL_minus_editN"]["target_mask_probability"]
    records.append(z)
with (OUT / "event_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
counts = {
    "N": len(rows),
    "videos": dict(Counter(r["video"] for r in rows)),
    "eligible": plan["eligible_count"],
    "selected": plan["selected_count"],
    "target_probability_up_editL": sum(r["delta"]["editL"]["target_mask_probability"] > 0 for r in rows),
    "target_probability_editL_above_editN": sum(r["paired_editL_minus_editN"]["target_mask_probability"] > 0 for r in rows),
    "long_L_mask_probability_down_editL": sum(r["delta"]["editL"]["long_L_mask_probability"] < 0 for r in rows),
    "primary": summary["primary"],
    "arms": {},
}
for arm in ("baseline", "editL", "editN"):
    counts["arms"][arm] = {
        "locked_query_hits50": sum(r["arms"][arm]["hit50"] for r in rows),
        "gt_time_any_query_oracle_hits50": sum(r["arms"][arm]["gt_time_any_query_oracle_tiou"] >= .5 for r in rows),
    }
    if arm != "baseline":
        for k in ("gain50", "loss50"):
            counts["arms"][arm][k] = sum(r["delta"][arm][k] for r in rows)
(OUT / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(counts, ensure_ascii=False, indent=2))
