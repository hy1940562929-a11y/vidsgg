"""Pure NumPy mathematics for annotation-readout diagnostics, not detection."""
import numpy as np


def position_features(length, width=256):
    if not 1 <= length <= 512 or width < 9:
        raise ValueError("Expected 1..512 valid samples and at least nine channels")
    z = np.zeros((width, length), np.float32)
    u = np.linspace(-1, 1, length, dtype=np.float32)
    z[0], z[1], z[2] = 1, u, u * u
    for k in range(1, 4):
        z[2*k+1], z[2*k+2] = np.sin(k*np.pi*u), np.cos(k*np.pi*u)
    return z


def annotation_union(episodes, origin, stride, length, classes=50):
    """Half-open original-frame labels sampled on the exact loader grid."""
    if stride <= 0 or length < 1:
        raise ValueError("Invalid grid")
    frames = origin + stride * np.arange(length)
    y = np.zeros((classes, length), dtype=bool)
    for e in episodes:
        p, a, b = int(e['predicate_id']), int(e['begin']), int(e['end'])
        if not (1 <= p <= classes and a < b):
            raise ValueError("Invalid annotation")
        y[p-1] |= (frames >= a) & (frames < b)
    return y


def components(bits, origin, stride, minimum_frames=5):
    """Connected sampled positives -> [first frame,last frame+1)."""
    bits = np.asarray(bits, dtype=bool)
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(origin+stride*a), int(origin+stride*(b-1)+1))
            for a, b in zip(starts, stops) if stride*(b-a-1)+1 >= minimum_frames]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def average_precision(labels, scores):
    """Threshold-grouped AP: tied scores cannot exploit target ordering."""
    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("Invalid AP vectors")
    if not y.any():
        return None
    order = np.argsort(-s, kind='stable')
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends+1)
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * precision))


def video_bootstrap(rows, arm, reference, reps=2000, seed=79):
    if not rows:
        return dict(N_GT=0, N_videos=0, net=0, net_pp=None, ci95_pp=None)
    videos = sorted({r['video_name'] for r in rows})
    counts = np.array([sum(r['video_name'] == v for r in rows) for v in videos])
    delta = np.array([sum(int(r['arms'][arm]['hit'])-int(r['arms'][reference]['hit'])
                          for r in rows if r['video_name'] == v) for v in videos])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(videos), size=(reps, len(videos)))
    values = 100*delta[sampled].sum(1)/counts[sampled].sum(1)
    gained = sum(r['arms'][arm]['hit'] and not r['arms'][reference]['hit'] for r in rows)
    lost = sum(r['arms'][reference]['hit'] and not r['arms'][arm]['hit'] for r in rows)
    return dict(N_GT=len(rows), N_videos=len(videos), new=int(gained), lost=int(lost),
                net=int(delta.sum()), net_pp=float(100*delta.sum()/len(rows)),
                ci95_pp=np.quantile(values, [.025, .975]).tolist(),
                bootstrap_unit='video; repeated crop/pair observations clustered together')
