"""E091 exact, GT-free temporal readouts; valid logits only.

P minimizes discrete CDF W1 over EVERY legal uniform grid interval. For each
length L, a translated uniform interval has quantile a + floor(L*u). Thus
W1(a) = E|X(u)-floor(L*u)-a| is convex in integer a. Its forward difference is
2*Pr[X(u)-floor(L*u) <= a]-1. Binary search finds the earliest minimizing a;
enumerating every L finds the global optimum. Complexity O(T^2 log T), memory
O(T^2). No approximation, GT-dependent search, or candidate-length division.

Computations use np.longdouble (platform precision is test-reported); exact
computed ties use the contract's smaller start then smaller end. No tie epsilon
is introduced. Frame stride 4 multiplies all W1 costs equally and therefore
does not change their argmin.
"""
import numpy as np


def _logits(logits_1d):
    z = np.asarray(logits_1d)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("finite one-dimensional valid native logits required")
    return z


def _interval(a, b, origin):
    a, b, origin = int(a), int(b), int(origin)
    if b <= a:
        return None
    return dict(start=origin + 4*a, end=origin + 4*b + 1,
                first_index=a, last_index=b, frame_length=4*(b-a)+1,
                sample_length=b-a+1)


def _cdf(z):
    z = np.asarray(z, dtype=np.longdouble)
    logp = -np.logaddexp(np.longdouble(0), -z)
    weight = np.exp(logp-logp.max())
    rho = weight/weight.sum(dtype=np.longdouble)
    f = np.cumsum(rho, dtype=np.longdouble)
    f[-1] = 1
    return f


def _costs(f, starts, lengths):
    k = np.arange(len(f)-1, dtype=np.longdouble)[None, :]
    uniform_cdf = np.clip((k-starts[:, None]+1)/lengths[:, None], 0, 1)
    return np.abs(f[None, :-1]-uniform_cdf).sum(axis=1, dtype=np.longdouble)


def project(logits_1d, origin=0):
    """Return one P interval, or None if fewer than two valid grid points.

    Admission is intentionally external: call only for original-H-admitted
    slots. P must not use the updated score's H to change the frozen capacity.
    """
    z = _logits(logits_1d)
    n = len(z)
    if n < 2:
        return None
    f = _cdf(z)
    lengths = np.arange(2, n+1, dtype=np.int64)
    lo = np.zeros(n-1, dtype=np.int64)
    hi = n-lengths
    j = np.arange(n, dtype=np.int64)[None, :]
    included = j < lengths[:, None]
    while np.any(lo < hi):
        a = (lo+hi)//2
        # Quantile coupling mass with X(u)-floor(L*u) <= a.
        indices = np.minimum(a[:, None]+j, n-1)
        mass_times_L = np.clip(f[indices]*lengths[:, None]-j, 0, 1)
        mass_times_L[~included] = 0
        slope_nonnegative = mass_times_L.sum(
            axis=1, dtype=np.longdouble) >= lengths/np.longdouble(2)
        active = lo < hi
        hi = np.where(active & slope_nonnegative, a, hi)
        lo = np.where(active & ~slope_nonnegative, a+1, lo)
    costs = _costs(f, lo, lengths)
    ends = lo+lengths-1
    order = np.lexsort((ends, lo, costs))
    winner = int(order[0])
    return _interval(lo[winner], ends[winner], origin)


def _probabilities_bits(logits_1d):
    z = _logits(logits_1d)
    # Exactly the 088 numerical contract: float32 sigmoid, float64 mean.
    x = z.astype(np.float32, copy=False)
    with np.errstate(over="ignore", under="ignore"):
        p = np.float32(1)/(np.float32(1)+np.exp(-x))
        p64 = 1/(1+np.exp(-x.astype(np.float64)))
    bits = p > np.float32(.5)
    if not np.array_equal(bits, p64 > .5) or not np.array_equal(bits, x > 0):
        raise ValueError("CPU sigmoid threshold boundary mismatch")
    return p, bits


def hull_eligible(logits_1d):
    """Original H has >=5 frames; this must be frozen on zero-update scores."""
    _, bits = _probabilities_bits(logits_1d)
    ids = np.flatnonzero(bits)
    return bool(len(ids) and ids[-1]-ids[0] >= 1)


def component(logits_1d, origin=0):
    """088 C: sigmoid>0.5, min5, max mean then peak then earliest start."""
    p, bits = _probabilities_bits(logits_1d)
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    choices = []
    for a, b in zip(starts, ends):
        if b > a:
            choices.append((-float(p[a:b+1].mean(dtype=np.float64)),
                            -float(p[a:b+1].max()), int(a), int(b)))
    if not choices:
        return None
    _, _, a, b = min(choices)
    return _interval(a, b, origin)
