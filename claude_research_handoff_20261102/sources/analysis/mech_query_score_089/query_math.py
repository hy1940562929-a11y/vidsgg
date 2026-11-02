"""Frozen 089 query selection and evaluation; no I/O or model inference."""
import numpy as np

ARMS = ('F_H', 'T_H', 'U_H', 'F_C', 'T_C', 'U_C', 'U_union')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def select_queries(class_logits, target_pid, legacy_query):
    """No time, duration, candidates, or evaluation outcomes enter selection."""
    z = np.asarray(class_logits, dtype=np.float64)
    require(z.shape == (9, 51) and np.isfinite(z).all(), 'class logits must be finite 9x51')
    require(1 <= target_pid <= 50 and 0 <= legacy_query < 9, 'invalid class/query')
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    p = exp / exp.sum(axis=1, keepdims=True)
    top = np.argmax(z[:, 1:], axis=1) + 1
    eligible = np.flatnonzero(top == target_pid)
    new_q = int(np.argmax(p[:, target_pid]))
    t = int(eligible[np.argmax(p[eligible, target_pid])]) if len(eligible) else None
    order = sorted(range(9), key=lambda q: (-p[q, target_pid], q))
    per_query = []
    for q in range(9):
        fg_order = sorted(range(1, 51), key=lambda c: (-z[q, c], c))
        per_query.append(dict(query=q, target_probability=float(p[q, target_pid]),
            background_probability=float(p[q, 0]), foreground_top1=int(top[q]),
            foreground_target_rank=fg_order.index(target_pid)+1,
            target_query_rank=order.index(q)+1,
            background_dominates_foreground=bool(z[q, 0] > z[q, 1:].max()),
            background_ties_foreground=bool(z[q, 0] == z[q, 1:].max())))
    return dict(F=legacy_query, T=t, target_pid=target_pid, current_target_argmax=new_q,
                score_drift=legacy_query != new_q, eligible_queries=eligible.tolist(),
                T_exists=t is not None, per_query=per_query)

def tiou(interval, start, end):
    if interval is None:
        return 0.0
    a, b = interval['start'], interval['end']
    intersection = max(0, min(b, end) - max(a, start))
    return float(intersection / (b-a+end-start-intersection))

def evaluate_event(selection, pair_candidates, start, end):
    require(start < end, 'invalid GT interval')
    out = dict(T_exists=selection['T_exists'], score_drift=selection['score_drift'])
    for arm in ('H', 'C'):
        slots = pair_candidates[arm]
        require(len(slots) == 9, 'nine query slots required')
        for policy in ('F', 'T'):
            name = policy+'_'+arm
            q = selection[policy]
            interval = slots[q] if q is not None else None
            out[name+'_query'] = q
            out[name+'_candidate_count'] = int(interval is not None)
            out[name+'_tiou'] = tiou(interval, start, end)
            out[name+'_hit'] = int(out[name+'_tiou'] >= .5)
            out[name+'_interval'] = interval
        out['U_'+arm+'_candidate_count'] = sum(x is not None for x in slots)
        out['U_'+arm+'_tiou'] = max(tiou(x, start, end) for x in slots)
        out['U_'+arm+'_hit'] = int(out['U_'+arm+'_tiou'] >= .5)
        out[arm+'_gain'] = int(out['T_'+arm+'_hit'] and not out['F_'+arm+'_hit'])
        out[arm+'_loss'] = int(out['F_'+arm+'_hit'] and not out['T_'+arm+'_hit'])
    out['U_union_tiou'] = max(out['U_H_tiou'], out['U_C_tiou'])
    out['U_union_hit'] = int(out['U_union_tiou'] >= .5)
    out['U_union_candidate_count'] = out['U_H_candidate_count'] + out['U_C_candidate_count']
    return out

def summarize(rows, bootstrap=True):
    videos = sorted({r['video_name'] for r in rows})
    per_video = []
    for video in videos:
        rr = [r for r in rows if r['video_name'] == video]
        per_video.append(dict(video_name=video, N_GT=len(rr),
            **{a: float(np.mean([r[a+'_hit'] for r in rr])) for a in ARMS}))
    counts = {a: sum(r[a+'_hit'] for r in rows) for a in ARMS}
    for a in ('H', 'C'):
        counts[a+'_gain'] = sum(r[a+'_gain'] for r in rows)
        counts[a+'_loss'] = sum(r[a+'_loss'] for r in rows)
    result = dict(N_GT=len(rows), N_videos=len(videos), videos=videos, integer_counts=counts,
        event_rates={a: counts[a]/len(rows) if rows else None for a in ARMS},
        video_equal_rates={a: float(np.mean([v[a] for v in per_video])) if videos else None for a in ARMS},
        candidate_count_unit='per-GT opportunities, not independent predictions; U_union counts H+C slots without deduplication',
        candidate_counts={a: sum(r[a+'_candidate_count'] for r in rows) for a in ARMS},
        comparisons={}, per_video=per_video)
    indices = np.random.default_rng(89).integers(0, len(videos), size=(10000, len(videos))) if videos and bootstrap else np.empty((0, len(videos)), dtype=np.int64)
    def estimate(values):
        if not len(values):
            return dict(mean=None, ci95=None)
        arr = np.asarray(values)
        ci = np.percentile(arr[indices].mean(axis=1), [2.5, 97.5]).tolist() if len(indices) else None
        return dict(mean=float(arr.mean()), ci95=ci)
    for arm in ('H', 'C'):
        result['comparisons'][arm] = estimate([v['T_'+arm]-v['F_'+arm] for v in per_video])
    result['T_C_miss'] = estimate([1-v['T_C'] for v in per_video])
    fail = [r for r in rows if not r['F_C_hit']]
    n = sum(not r['U_union_hit'] for r in fail)
    result['descriptive_only_U_union_miss_among_F_C_failure'] = dict(numerator=n, denominator=len(fail), fraction=n/len(fail) if fail else None, independent_attribution_evidence=False)
    return result, indices

def decide(full, common, score_drift):
    if score_drift:
        return 'INCONCLUSIVE_SCORE_DRIFT'
    f = full['comparisons']['C']; c = common['comparisons']['C']
    if f['mean'] >= .05 and f['ci95'][0] > 0 and c['mean'] is not None and c['mean'] > 0 and common['integer_counts']['C_gain'] > common['integer_counts']['C_loss']:
        return 'SUPPORTED_QUERY_RESELECTION'
    if common['N_GT'] >= 40 and common['N_videos'] >= 10 and common['T_C_miss']['ci95'][0] > .5 and f['ci95'][1] < .05:
        return 'SUPPORTED_RESIDUAL_INTERVAL_DEFICIT'
    return 'INCONCLUSIVE'
