"""NumPy-only, frame-coordinate helpers for 079 (no model or dataset needed)."""
from __future__ import annotations
import numpy as np


def sampled_frames(start: int, end: int, stride: int, offset: int = 0):
    if stride < 1 or offset < 0 or offset >= stride or end < start:
        raise ValueError('invalid physical sampling coordinates')
    return np.arange(start + offset, end, stride, dtype=np.int64)


def interpolate_features(source, source_frames, target_frames):
    """Linear interpolation in actual frame coordinates; boundary values clamp.

    No temporal extrapolation of observations is claimed: the returned edge
    extension repeats the nearest observed feature and is saved as such.
    """
    x = np.asarray(source)
    sf, tf = np.asarray(source_frames), np.asarray(target_frames)
    if x.ndim != 2 or x.shape[1] != len(sf) or len(sf) < 1:
        raise ValueError('source must be C x sampled-time with nonempty frames')
    if np.any(np.diff(sf) <= 0):
        raise ValueError('source frames must increase strictly')
    if len(sf) == 1:
        return np.repeat(x, len(tf), axis=1)
    r = np.searchsorted(sf, tf, side='right').clip(1, len(sf)-1)
    l = r-1
    w = ((tf-sf[l])/(sf[r]-sf[l])).clip(0, 1)
    return (x[:, l] * (1-w) + x[:, r] * w).astype(x.dtype)


def visual_interpolation(source, dense, source_frames, target_frames, full=False):
    """2069 channels: 1024 subject ROI + 1024 object ROI + 21 geometry.

    Primary visual-only control retains exactly the dense geometry (whose
    finite differences depend on stride), modifying only 2048 ROI channels.
    """
    source, dense = np.asarray(source), np.asarray(dense)
    if source.ndim != 2 or dense.ndim != 2 or source.shape[0] != 2069 or dense.shape[0] != 2069:
        raise ValueError('visual prefix contract requires exactly 2069 channels')
    if dense.shape[1] != len(target_frames):
        raise ValueError('dense coordinate length mismatch')
    result = dense.copy()
    channels = 2069 if full else 2048
    result[:channels] = interpolate_features(source[:channels], source_frames, target_frames)
    if not full:
        assert np.array_equal(result[2048:], dense[2048:])
    return result


def envelope_intervals(binary_masks, frames, min_frames=5):
    """Native min/max envelope at sigmoid > .5, [first_frame,last_frame+1).

    This intentionally does NOT split disconnected islands or append a stride
    cell. The same >=5 physical-frame rule applies to every sampling arm.
    """
    z, fs = np.asarray(binary_masks), np.asarray(frames)
    if z.dtype != np.bool_:
        raise TypeError('pass native Torch sigmoid > 0.5 booleans, not logits')
    if z.ndim != 2 or z.shape[1] != len(fs):
        raise ValueError('mask / frame coordinate shape mismatch')
    result = []
    for row in z:
        ids = np.flatnonzero(row)
        span = None if not len(ids) else (int(fs[ids[0]]), int(fs[ids[-1]])+1)
        result.append(span if span and span[1]-span[0] >= min_frames else None)
    return result


def forward_hold_binary_iou(binary_masks, frames, pair_end, target):
    """Annotation IoU of raw occupancy, with each token held to the next frame.

    This is an explicit additional diagnostic, not the native hull decoder or
    historical GT27 binary metric. The last cell ends at continuous co-presence
    end; a phase offset leaves the unsampled prefix unpredicted. A GT interval
    extending beyond this pair remains in the union, never cropped to the pair.
    """
    z, fs = np.asarray(binary_masks), np.asarray(frames, dtype=np.int64)
    if z.dtype != np.bool_ or z.ndim != 2 or z.shape[1] != len(fs):
        raise ValueError('expected boolean Q x sampled-time masks')
    if not len(fs):
        return np.zeros(z.shape[0], dtype=np.float64)
    if np.any(np.diff(fs) <= 0) or pair_end <= fs[-1]:
        raise ValueError('frames must strictly increase inside pair')
    ends = np.r_[fs[1:], int(pair_end)]
    lengths = ends-fs
    a,b = target
    if b <= a: raise ValueError('empty GT interval')
    intersections = np.maximum(0, np.minimum(ends,b)-np.maximum(fs,a))
    intersection = z.astype(np.float64) @ intersections
    predicted = z.astype(np.float64) @ lengths
    return intersection/(predicted+(b-a)-intersection)


def interval_iou(span, target):
    if span is None:
        return 0.0
    a, b = span; c, d = target
    overlap = max(0, min(b, d)-max(a, c))
    union = (b-a)+(d-c)-overlap
    return float(overlap / union) if union > 0 else 0.0


def duration_bin(seconds):
    return 'sub1' if seconds < 1 else '1to3' if seconds < 3 else '3to10' if seconds < 10 else '10plus'


def paired_bootstrap(rows, left, right, metric, reps=2000, seed=79):
    """Resample videos, not correlated GT rows; difference right minus left."""
    vids = sorted({r['video'] for r in rows})
    if not vids:
        return {'videos': 0, 'GT': 0, 'ci95_pp': None}
    sums = np.array([[sum(int(r[right][metric])-int(r[left][metric]) for r in rows if r['video']==v),
                      sum(r['video']==v for r in rows)] for v in vids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vids), size=(reps, len(vids)))
    totals = sums[indices].sum(axis=1)
    values = 100*totals[:,0]/totals[:,1]
    ci = np.quantile(values, [.025, .975]).tolist() if reps else None
    return {'videos': len(vids), 'GT': len(rows), 'net_pp': float(100*sums[:,0].sum()/sums[:,1].sum()),
            'ci95_pp': ci, 'resampling_unit': 'video', 'replicates': reps}
