"""Frozen E091 evaluation; no training, candidate selection, or GT-guided decoding."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (90, 91, 92)
ARMS = ["Z"] + [f"{arm}{seed}" for arm in "ABS" for seed in SEEDS]
N_BOOT = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(x):
    if x is None:
        return None
    if isinstance(x, dict):
        x = (x["start"], x["end"])
    require(len(x) == 2, "interval must have two endpoints")
    a, b = x
    require(np.isfinite([a, b]).all() and int(a) == a and int(b) == b and b > a,
            "invalid half-open integer frame interval")
    return int(a), int(b)


def tiou(x, gt):
    if x is None:
        return 0.0
    overlap = max(0, min(x[1], gt[1]) - max(x[0], gt[0]))
    return overlap / (x[1] - x[0] + gt[1] - gt[0] - overlap)


def position_null(x, video, pair, query, origin, n_valid, repetition, stride=4):
    """088 integer hash placement, namespace/key 91; accepts no GT annotation."""
    if x is None:
        return None
    require(stride == 4 and 0 <= repetition < 100 and 0 <= query < 9, "null contract")
    require((x[0] - origin) % stride == 0 and (x[1] - origin - 1) % stride == 0,
            "null source is not on native grid")
    length = (x[1] - x[0] - 1) // stride + 1
    available = n_valid - length + 1
    require(length >= 2 and available > 0, "null length/support mismatch")
    key = f"91/{video}/{int(pair)}/{int(query)}/{int(repetition)}"
    numerator = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    first = numerator * available // (2**64)
    return (origin + stride * first, origin + stride * (first + length - 1) + 1)


def integer_change(b, k):
    b, k = np.asarray(b, bool), np.asarray(k, bool)
    require(b.shape == k.shape, "comparison shape mismatch")
    gain, loss = int(np.sum(b & ~k)), int(np.sum(~b & k))
    return dict(N=int(b.size), B_hits=int(b.sum()), reference_hits=int(k.sum()),
                gain=gain, loss=loss, net=gain-loss,
                both_hit=int(np.sum(b & k)), neither_hit=int(np.sum(~b & ~k)))


class VideoStats:
    def __init__(self, videos):
        self.videos = np.asarray(videos, str)
        self.order = sorted(set(videos))
        require(bool(self.order), "empty video support")
        self.groups = [np.flatnonzero(self.videos == v) for v in self.order]
        self.draws = np.random.default_rng(91).integers(
            0, len(self.order), size=(N_BOOT, len(self.order)))

    def means(self, values):
        return np.array([np.mean(np.asarray(values)[ids]) for ids in self.groups])

    def difference(self, b, k):
        by_video = self.means(np.asarray(b) - np.asarray(k))
        bootstrap = by_video[self.draws].mean(1)
        return dict(point=float(by_video.mean()),
                    ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                    N_events=len(self.videos), N_videos=len(self.order),
                    per_video=dict(zip(self.order, by_video.tolist())))

    def ratio(self, b, k):
        bm, km = self.means(b), self.means(k)
        if float(km.mean()) <= 0:
            return dict(point=None, ci95=None, passes=False, reason="zero reference error")
        numerator = bm[self.draws].mean(1)
        denominator = km[self.draws].mean(1)
        if np.any(denominator <= 0):
            return dict(point=float(bm.mean()/km.mean()), ci95=None,
                        passes=False, reason="zero bootstrap reference error")
        point = float(bm.mean()/km.mean())
        ci = np.quantile(numerator/denominator, [.025, .975]).tolist()
        return dict(point=point, ci95=ci, passes=bool(point <= .90 and ci[1] < 1),
                    B_error=float(bm.mean()), reference_error=float(km.mean()),
                    N_events=len(self.videos), N_videos=len(self.order))

    def residual_net(self, difference):
        """Event-weighted NR; each resample uses the resampled event denominator."""
        d = np.asarray(difference, float)
        sums = np.array([d[ids].sum() for ids in self.groups])
        counts = np.array([len(ids) for ids in self.groups])
        denominator = counts[self.draws].sum(1)
        value = sums[self.draws].sum(1)/denominator
        return dict(point=float(d.sum()/len(d)), numerator=float(d.sum()),
                    denominator=len(d), ci95=np.quantile(value, [.025, .975]).tolist(),
                    resampled_denominator_min=int(denominator.min()),
                    resampled_denominator_max=int(denominator.max()),
                    denominator_rule="number of residual events in sampled whole videos")

    def receipt(self):
        return dict(seed=91, repetitions=N_BOOT, video_order=self.order,
                    indices_sha256=hashlib.sha256(self.draws.astype("<i8").tobytes()).hexdigest(),
                    uncertainty="conditional on the three fixed training seeds",
                    seed_reduction="paired seed average before video aggregation")


def decide(gates):
    """First matching row of CAUSE_STATEMENT 5.5; no combined verdicts."""
    if not gates["technical"]:
        return "TECHNICAL_FAILURE"
    if gates["primary_ci_upper_below_5pp"] or gates["long_clear_harm"]:
        return "NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE"
    base = all(gates[k] for k in ("primary", "seed_direction", "protection", "null"))
    if base and gates["endpoint_E"] and gates["score_C"]:
        return "VERIFIED_INTERVENTION_EFFECT"
    if base and not (gates["endpoint_E"] and gates["score_C"]):
        return "METHOD_ONLY_NOT_VERIFIED_CAUSE"
    return "UNRESOLVED_STOP_WITHOUT_EXPANSION"


def descriptive_scores(data, real, masks, out, score_base):
    """Annotation-dependent descriptive receipts, never used to select an exit."""
    files = data.get("score_files")
    if files is None:
        return dict(status="NOT_PROVIDED", reason="native logits absent; no fabricated score metrics")
    require(set(files) == set(ARMS), "score files must cover all frozen arms")
    require(score_base is not None, "score base required")
    receipt, rows, aggregates = {}, [], {}
    metrics = ("cdf_w1_over_gt_duration", "outside_probability_sum",
               "outside_normalized_mass", "outside_distance_seconds",
               "outside_distance_over_gt_duration", "positive_components",
               "legal_min5_components", "P_duration_ratio", "C_duration_ratio")
    for key in ARMS:
        path = Path(score_base) / files[key]
        receipt[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        aggregates[key] = {}
        values = {name: [] for name in metrics}
        with np.load(path, allow_pickle=False) as scores:
            for i, event in enumerate(data["events"]):
                sid = event["slot_id"]
                row = dict(arm=key, video_name=event["video_name"], gt_index=event["gt_index"],
                           slot_id=sid, zero_grid_positive=False)
                row.update({name: None for name in metrics})
                if sid in scores:
                    z = np.asarray(scores[sid])
                    require(z.dtype == np.float32 and z.ndim == 1 and
                            len(z) == int(event["N_valid"]) and np.isfinite(z).all(),
                            "invalid sealed native score")
                    z = z.astype(np.float64)
                    logp = -np.logaddexp(0., -z)
                    p = np.exp(logp)
                    rho = np.exp(logp-logp.max())
                    rho /= rho.sum()
                    fps = float(event["fps"])
                    require(fps > 0, "fps required for score diagnostics")
                    a, b = int(event["begin_fid"]), int(event["end_fid"])
                    duration = (b-a)/fps
                    frames = int(event["origin"])+4*np.arange(len(z))
                    target = (frames >= a) & (frames < b)
                    row["zero_grid_positive"] = not bool(target.any())
                    if target.any():
                        target_mass = target/target.sum()
                        row["cdf_w1_over_gt_duration"] = float(
                            4/fps*np.abs(np.cumsum(rho-target_mass)[:-1]).sum()/max(duration, 4/fps))
                    outside = ~target
                    mass = float(rho[outside].sum())
                    distances = np.minimum(np.abs(frames-a), np.abs(frames-b))/fps
                    row["outside_probability_sum"] = float(p[outside].sum())
                    row["outside_normalized_mass"] = mass
                    row["outside_distance_seconds"] = (float(np.sum(rho[outside]*distances[outside])/mass)
                                                       if mass > 0 else 0.)
                    row["outside_distance_over_gt_duration"] = row["outside_distance_seconds"]/duration
                    bits = z > 0
                    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
                    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                    row["positive_components"] = int(len(starts))
                    row["legal_min5_components"] = int(np.sum(4*(stops-starts-1)+1 >= 5))
                    for rd in ("P", "C"):
                        x = real[key][rd][i]
                        row[f"{rd}_duration_ratio"] = (x[1]-x[0])/(b-a) if x is not None else None
                else:
                    require(int(event["pair_idx"]) < 0 or event["slot_id"] is None,
                            f"missing native score for fixed slot {sid}")
                rows.append(row)
                for metric in metrics:
                    values[metric].append(row[metric])
        for group in ("main400", "sub163", "residual334", "long_protection", "all"):
            aggregates[key][group] = {}
            for metric in metrics:
                observed = [x for i, x in enumerate(values[metric]) if masks[group][i] and x is not None]
                aggregates[key][group][metric] = dict(
                    mean=float(np.mean(observed)) if observed else None,
                    N_defined=len(observed), N_group=int(masks[group].sum()))
    with (out/"score_descriptives_per_gt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(status="DESCRIPTIVE_ONLY", sources=receipt, aggregates=aggregates,
                outside_distance_definition="rho-weighted mean distance of GT-exterior grid points to nearest GT endpoint",
                W1_zero_positive_rule="undefined when GT has zero sampled positives; support explicitly reported",
                admission="score diagnostics include all existing fixed slots, not only P-admitted slots")


def evaluate(data, out, expected_counts=True, score_base=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    events, predictions = data["events"], data["predictions"]
    require(set(predictions) == set(ARMS), "must contain precisely Z and A/B/S x 90/91/92")
    require(len({(e["video_name"], int(e["gt_index"])) for e in events}) == len(events),
            "duplicate event identity")
    n = len(events)
    masks = {k: np.array([bool(e[k]) for e in events])
             for k in ("main400", "sub163", "residual334", "long_protection", "admitted")}
    masks["E381"] = masks["main400"] & masks["admitted"]
    masks["all"] = np.ones(n, bool)
    require(np.all(~masks["sub163"] | masks["main400"]), "163 outside 400")
    require(np.all(~masks["residual334"] | masks["main400"]), "334 outside 400")
    if expected_counts:
        for k, count in (("all", 2573), ("main400", 400), ("sub163", 163),
                         ("residual334", 334), ("E381", 381), ("long_protection", 1101)):
            require(int(masks[k].sum()) == count, f"{k} count differs from sealed support")
        require(len({e["video_name"] for e in events}) == 27, "GT27 video count")
        require(len({e["video_name"] for e in events if e["main400"]}) == 23,
                "main400 video count")
    real = {}
    for key in ARMS:
        require(len(predictions[key]) == n, f"{key} row count mismatch")
        real[key] = {readout: [interval(r[readout]) for r in predictions[key]] for readout in ("P", "C")}
        slots = {}
        for i, e in enumerate(events):
            p, c = real[key]["P"][i], real[key]["C"][i]
            require((p is not None) == bool(e["admitted"]), f"{key}/{i} P admission changed")
            require(c is None or e["admitted"], f"{key}/{i} C outside frozen admission")
            for x in (p, c):
                if x is not None:
                    origin, nv = int(e["origin"]), int(e["N_valid"])
                    require(x[1]-x[0] >= 5 and x[0] >= origin and
                            x[1] <= origin+4*(nv-1)+1 and
                            (x[0]-origin) % 4 == 0 and (x[1]-origin-1) % 4 == 0,
                            f"{key}/{i} illegal interval")
            if e["pair_idx"] is not None and int(e["pair_idx"]) >= 0:
                sid = (e["video_name"], int(e["pair_idx"]), int(e["query"]))
                require(sid not in slots or slots[sid] == (p, c), "same slot reused different predictions")
                slots[sid] = (p, c)
    # Seal all no-GT position libraries before using annotations to score them.
    null = {}
    for seed in SEEDS:
        key = f"B{seed}"
        null[key] = []
        for i, e in enumerate(events):
            x = real[key]["P"][i]
            null[key].append([
                position_null(x, e["video_name"], e["pair_idx"], int(e["query"]),
                              int(e["origin"]), int(e["N_valid"]), rep)
                if x is not None else None for rep in range(100)])
    null_path = out / "null_intervals.json"
    null_path.write_text(json.dumps(dict(key=91, repetitions=100, libraries=null), separators=(",", ":")))
    null_sha = hashlib.sha256(null_path.read_bytes()).hexdigest()
    gt = [(int(e["begin_fid"]), int(e["end_fid"])) for e in events]
    require(all(b > a for a, b in gt), "invalid GT time")
    hit, ious, errors = {}, {}, {}
    for key in ARMS:
        hit[key], ious[key] = {}, {}
        for readout in ("P", "C"):
            v = np.array([tiou(x, g) for x, g in zip(real[key][readout], gt)])
            ious[key][readout], hit[key][readout] = v, v >= .5
        errors[key] = np.array([
            (abs(x[0]-g[0])+abs(x[1]-g[1]))/(2*(g[1]-g[0])) if x is not None else np.nan
            for x, g in zip(real[key]["P"], gt)])
    null_hit = {key: np.array([[tiou(x, gt[i]) >= .5 for x in rows]
                               for i, rows in enumerate(libraries)], float)
                for key, libraries in null.items()}
    average = {arm: {readout: np.mean([hit[f"{arm}{s}"][readout] for s in SEEDS], axis=0)
                     for readout in ("P", "C")} for arm in "ABS"}
    average["Z"] = {r: hit["Z"][r].astype(float) for r in ("P", "C")}
    avg_errors = {a: np.mean([errors[f"{a}{s}"] for s in SEEDS], axis=0) for a in "ABS"}
    avg_errors["Z"] = errors["Z"]
    stats = {k: VideoStats([e["video_name"] for i, e in enumerate(events) if mask[i]])
             for k, mask in masks.items() if k != "admitted"}
    m, e_mask, l, r = (masks[k] for k in ("main400", "E381", "long_protection", "residual334"))
    comp_p = {k: stats["main400"].difference(average["B"]["P"][m], average[k]["P"][m])
              for k in ("A", "S", "Z")}
    comp_c = {k: stats["main400"].difference(average["B"]["C"][m], average[k]["C"][m])
              for k in ("A", "S", "Z")}
    comp_e = {k: stats["E381"].ratio(avg_errors["B"][e_mask], avg_errors[k][e_mask])
              for k in ("A", "S", "Z")}
    protection, overall = {}, {}
    for k in ("A", "S", "Z", "C0"):
        reference = average[k]["P"] if k != "C0" else average["Z"]["C"]
        protection[k] = stats["long_protection"].difference(average["B"]["P"][l], reference[l])
        overall[k] = dict(N=n, net=float((average["B"]["P"]-reference).sum()))
    null_mean = np.mean([null_hit[f"B{s}"] for s in SEEDS], axis=0)
    null_comp = stats["main400"].difference(average["B"]["P"][m], null_mean.mean(1)[m])
    per_seed = {}
    for seed in SEEDS:
        seed_summary = {}
        for group in ("main400", "sub163", "residual334", "long_protection", "all"):
            sel = masks[group]
            d = {}
            for readout in ("P", "C"):
                comparisons = {}
                for k in ("A", "S", "Z"):
                    ref = hit[k if k == "Z" else f"{k}{seed}"][readout]
                    comparisons[k] = integer_change(hit[f"B{seed}"][readout][sel], ref[sel])
                    comparisons[k]["video_equal_difference"] = float(
                        stats[group].means(hit[f"B{seed}"][readout][sel].astype(float)-ref[sel]).mean())
                if readout == "P":
                    comparisons["C0"] = integer_change(hit[f"B{seed}"]["P"][sel], hit["Z"]["C"][sel])
                d[readout] = comparisons
            seed_summary[group] = d
        per_seed[str(seed)] = seed_summary
    nr = {k: stats["residual334"].residual_net(average["B"]["P"][r]-average[k]["P"][r])
          for k in ("A", "S", "Z")}
    gates = dict(
        technical=True,
        primary_ci_upper_below_5pp=any(v["ci95"][1] < .05 for v in comp_p.values()),
        long_clear_harm=any(v["ci95"][1] < -.01 for v in protection.values()),
        primary=all(v["point"] >= .05 and v["ci95"][0] > 0 for v in comp_p.values()),
        seed_direction=all(per_seed[str(s)]["main400"]["P"][k]["video_equal_difference"] > 0
                           for s in SEEDS for k in ("A", "S")),
        endpoint_E=all(v["passes"] for v in comp_e.values()),
        score_C=all(v["ci95"][0] > 0 for v in comp_c.values()),
        protection=all(v["ci95"][0] >= -.01 for v in protection.values())
                   and all(v["net"] >= 0 for v in overall.values()),
        null=null_comp["ci95"][0] > 0)
    groups = {}
    for group in ("main400", "sub163", "residual334", "long_protection", "all"):
        sel = masks[group]
        groups[group] = dict(
            N=int(sel.sum()), N_videos=len(stats[group].order),
            hits={key: {readout: int(hit[key][readout][sel].sum()) for readout in ("P", "C")} for key in ARMS},
            nonempty={key: {readout: sum(x is not None for i, x in enumerate(real[key][readout]) if sel[i])
                           for readout in ("P", "C")} for key in ARMS},
            video_equal_rates={a: {rd: float(stats[group].means(average[a][rd][sel]).mean())
                                   for rd in ("P", "C")} for a in ("A", "B", "S", "Z")})
    summary = dict(
        experiment="E091-GEOMETRY-CONTRACT", verdict=decide(gates), gates=gates,
        support={k: int(v.sum()) for k, v in masks.items()}, groups=groups,
        primary_P=comp_p, endpoint_E=comp_e, score_C=comp_c,
        long_protection=protection, full_population_net=overall,
        null=dict(comparison=null_comp, sha256=null_sha, key=91, libraries=100,
                  pooling="evaluate each library separately, then average",
                  per_library_video_equal_rates=[float(stats["main400"].means(null_mean[m, j]).mean())
                                                 for j in range(100)],
                  per_seed_library_integer_hits={str(s): null_hit[f"B{s}"][m].sum(0).astype(int).tolist()
                                                 for s in SEEDS}),
        NR334=nr, per_seed=per_seed,
        bootstrap={k: v.receipt() for k, v in stats.items()},
        scope="controlled frozen-branch intervention only; never VERIFIED_CAUSE or full benchmark mAP")
    summary["descriptive_scores"] = descriptive_scores(data, real, masks, out, score_base)
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    fields = ["video_name", "gt_index", "pair_idx", "query", "begin_fid", "end_fid",
              "main400", "sub163", "residual334", "long_protection", "admitted"]
    fields += [f"{key}_{rd}_{metric}" for key in ARMS for rd in ("P", "C")
               for metric in ("start", "end", "tiou", "hit")]
    fields += [f"{key}_P_endpoint_error" for key in ARMS]
    fields += [f"B{s}_minus_{ref}_{rd}" for s in SEEDS for ref in ("A", "S", "Z") for rd in ("P", "C")]
    with (out/"per_gt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, event in enumerate(events):
            row = {k: event[k] for k in fields[:11]}
            for key in ARMS:
                for rd in ("P", "C"):
                    x = real[key][rd][i]
                    row.update({f"{key}_{rd}_start": x[0] if x else "",
                                f"{key}_{rd}_end": x[1] if x else "",
                                f"{key}_{rd}_tiou": float(ious[key][rd][i]),
                                f"{key}_{rd}_hit": int(hit[key][rd][i])})
                row[f"{key}_P_endpoint_error"] = float(errors[key][i]) if np.isfinite(errors[key][i]) else ""
            for s in SEEDS:
                for ref in ("A", "S", "Z"):
                    rk = ref if ref == "Z" else f"{ref}{s}"
                    for rd in ("P", "C"):
                        row[f"B{s}_minus_{ref}_{rd}"] = int(hit[f"B{s}"][rd][i])-int(hit[rk][rd][i])
            writer.writerow(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = Path(args.predictions).read_bytes()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(input_path=str(Path(args.predictions).resolve()),
                   input_sha256=hashlib.sha256(raw).hexdigest())
    (out/"evaluation_input_seal.json").write_text(json.dumps(receipt, indent=2))
    try:
        summary = evaluate(json.loads(raw), out, score_base=Path(args.predictions).parent)
    except Exception as exc:
        (out/"technical_failure.json").write_text(json.dumps(dict(verdict="TECHNICAL_FAILURE", error=repr(exc)), indent=2))
        raise
    print(json.dumps(dict(verdict=summary["verdict"], support=summary["support"]), indent=2))


if __name__ == "__main__":
    main()
