"""Batch sampling weights for short / long-only oversampling."""
import torch


def batch_stats(cpu_c):
    """Return n_short, n_long, n_coexist_pairs (pair has ≥1 short and ≥1 long)."""
    n_short = n_long = n_co = 0
    for meta in cpu_c["metadata"]:
        durs = [float(e["duration_seconds"]) for e in meta["events"]]
        s = sum(1 for d in durs if d < 3)
        l = sum(1 for d in durs if d >= 3)
        n_short += s
        n_long += l
        if s > 0 and l > 0:
            n_co += 1
    return n_short, n_long, n_co


def build_weights(cpu_batches, arm):
    names = sorted(cpu_batches.keys())
    stats = {n: batch_stats(cpu_batches[n]["c"]) for n in names}
    w = []
    for n in names:
        ns, nl, nc = stats[n]
        if arm == "N":
            w.append(1.0)
        elif arm == "T":
            w.append(1.0 + float(ns) + 2.0 * float(nc))
        elif arm == "X":
            # prefer long-only / long-heavy, downweight shorts
            w.append(1.0 + float(nl) + (0.0 if ns > 0 else 2.0))
        else:
            raise RuntimeError(arm)
    total = sum(w)
    probs = [x / total for x in w]
    return names, probs, stats


def sample_order(names, probs, steps, rng):
    # rng: random.Random
    return [rng.choices(names, weights=probs, k=1)[0] for _ in range(steps)]


def exposure_rates(order, stats):
    ns = nl = nc = 0
    for n in order:
        a, b, c = stats[n]
        ns += a; nl += b; nc += c
    m = max(len(order), 1)
    return dict(mean_short_per_batch=ns / m, mean_long_per_batch=nl / m, mean_coexist_pairs_per_batch=nc / m)
