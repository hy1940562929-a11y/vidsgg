"""Exact single-slot capacity audit. GT oracle only, never model predictions."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def hit(interval, e):
    if interval is None:
        return False
    a, b = interval
    g, h = e["begin_fid"], e["end_fid"]
    inter = max(0, min(b, h) - max(a, g))
    return 3 * inter >= b - a + h - g


def capacity(rows):
    """All legal [origin+4a,origin+4b+1), b>a, and the empty interval."""
    n, o = rows[0]["N_valid"], rows[0]["origin"]
    assert all((r["N_valid"], r["origin"], r["admitted"]) ==
               (n, o, rows[0]["admitted"]) for r in rows)
    best, best_interval, best_mask, union = 0, None, 0, 0
    if rows[0]["admitted"]:
        for a in range(n - 1):
            s = o + 4 * a
            for b in range(a + 1, n):
                end = o + 4 * b + 1
                bits = 0
                for k, r in enumerate(rows):
                    if hit((s, end), r):
                        bits |= 1 << k
                union |= bits
                count = bits.bit_count()
                if count > best:
                    best, best_interval, best_mask = count, [s, end], bits
    return best, best_interval, best_mask, union


def main():
    start = time.time()
    freeze = read(OUT / "input_freeze.json")["files"]
    for name, digest in freeze.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    receipt = read(ROOT / "analysis/cause_arch_095_e091/import_receipt_v2.json")["files"]
    source = ROOT / "analysis/cause_arch_095_e091/results_v2"
    for name in ["events.json", "decoded_Z.json"]:
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == receipt["results_v2/" + name]
    events, zero = read(source / "events.json"), read(source / "decoded_Z.json")
    assert len(events) == 2573
    assert sum(e["main400"] and e["admitted"] for e in events) == 381
    results = {}
    for group, expected in [("main400", 400), ("sub163", 163), ("residual334", 334)]:
        selected = [e for e in events if e[group]]
        assert len(selected) == expected
        slots = defaultdict(list)
        for e in selected:
            slots[e["slot_id"]].append(e)
        details = []
        for sid, rows in sorted(slots.items()):
            c = zero[sid]["C"]
            baseline = [hit(None if c is None else (c["start"], c["end"]), e) for e in rows]
            assert rows[0]["admitted"] or c is None
            best, interval, bits, union = capacity(rows)
            details.append(dict(
                slot_id=sid, admitted=rows[0]["admitted"], N=len(rows),
                keys=[[r["video_name"], r["gt_index"]] for r in rows],
                C0_hits=sum(baseline), max_simultaneous_hits=best,
                max_net_recovery=best-sum(baseline),
                individual_feasible=union.bit_count(),
                conflicting=best < union.bit_count(),
                oracle_interval=interval,
                oracle_hit_keys=[[r["video_name"], r["gt_index"]] for k, r in enumerate(rows) if bits & (1 << k)],
            ))
        results[group] = dict(
            N=expected, unique_slots=len(slots),
            admitted_GT=sum(e["admitted"] for e in selected),
            C0_hits=sum(d["C0_hits"] for d in details),
            max_hits=sum(d["max_simultaneous_hits"] for d in details),
            max_net_recovery=sum(d["max_net_recovery"] for d in details),
            individually_feasible=sum(d["individual_feasible"] for d in details),
            conflicting_slots=sum(d["conflicting"] for d in details),
            details=details,
        )
        print(group, {k: v for k, v in results[group].items() if k != "details"}, flush=True)
    upper = results["residual334"]["max_net_recovery"]
    result = dict(
        capacity_gate="PASS" if upper > 167 else "REJECT_SCOPE_CAPACITY_CONFLICT",
        threshold_numerator=167, denominator=334,
        elapsed_seconds=time.time()-start, groups=results,
        oracle_only=True, model_training_runs=0,
    )
    (OUT / "capacity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print("CAPACITY_GATE", result["capacity_gate"], flush=True)


if __name__ == "__main__":
    main()
