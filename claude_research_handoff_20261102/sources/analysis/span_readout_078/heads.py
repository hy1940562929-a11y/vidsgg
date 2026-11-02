"""078 frozen-feature span readouts, with an exactly native initial decoder.

This module never reads GT. Both arms have the same parameters and differ only
in whether their low-rank score compares two endpoints or is separable. Decode
uses inclusive sampled endpoints: i < j gives at least five original frames at
stride four. A native mask is a logit array; foreground means sigmoid > .5.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


INVALID_SCORE = -1.0e9


def _indices(record, b, q, device):
    b = torch.as_tensor(b, dtype=torch.long, device=device).reshape(-1)
    q = torch.as_tensor(q, dtype=torch.long, device=device).reshape(-1)
    if b.numel() == 1 and q.numel() != 1:
        b = b.expand(q.numel())
    if q.numel() == 1 and b.numel() != 1:
        q = q.expand(b.numel())
    if b.shape != q.shape:
        raise ValueError("b and q must contain the same number of selected queries")
    batch, queries = record["cls"].shape[1:3]
    if ((b < 0) | (b >= batch)).any() or ((q < 0) | (q >= queries)).any():
        raise IndexError("Selected batch/query index is out of range")
    return b, q


def _selected(record, key, b, q, device, dtype):
    """Index before transferring, avoiding a copy of the entire frozen cache."""
    source = record[key]
    bi, qi = b.to(source.device), q.to(source.device)
    if key == "f":
        value = source[bi]
    elif key == "valid":
        value = source[bi, 0]
    elif key in ("masks", "cls"):
        value = source[-1, bi, qi]
    elif key == "hs":
        layers, batch, queries, _ = record["cls"].shape
        if source.shape != (layers * batch, record["f"].shape[1], queries):
            raise ValueError("Expected hs shaped [layers * batch, dim, queries]")
        value = source.reshape(layers, batch, source.shape[1], queries)[-1]
        value = value[bi, :, qi]
    else:
        raise KeyError(key)
    return value.detach().to(device=device, dtype=dtype)


class SpanHead(nn.Module):
    def __init__(self, mode="independent", dim=256, hidden=32, rank=8,
                 anchor_scale=8.0):
        super().__init__()
        if mode not in ("independent", "joint"):
            raise ValueError("mode must be 'independent' or 'joint'")
        if min(dim, hidden, rank) <= 0 or anchor_scale <= 0:
            raise ValueError("Dimensions and anchor_scale must be positive")
        self.mode = mode
        self.dim, self.rank = dim, rank
        self.anchor_scale = float(anchor_scale)
        self.feature = nn.Linear(dim, hidden)
        self.query = nn.Linear(dim, hidden, bias=False)
        self.classes = nn.Linear(51, hidden, bias=False)
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.endpoint = nn.Linear(hidden, 2)
        self.u = nn.Linear(hidden, rank)
        self.v = nn.Linear(hidden, rank)
        self.gate = nn.Parameter(torch.zeros(()))
        # Only the output residual is zero. Nonzero u/v allow gate gradients
        # at the first step; zeroing all three would permanently kill the term.
        nn.init.zeros_(self.endpoint.weight)
        nn.init.zeros_(self.endpoint.bias)

    @property
    def device(self):
        return self.gate.device

    def _valid(self, record, b, q):
        return _selected(record, "valid", b, q, self.device, torch.bool)

    @torch.no_grad()
    def anchor(self, record, b, q):
        """Native positive hull, or a flagged deterministic feasible fallback.

        Returns three [N] tensors (first, last, fallback). If fewer than two
        samples are valid, first=last=-1 and fallback=True. For an empty or
        singleton foreground, use the highest-logit valid sample and its
        closest valid neighbor; ties choose the earlier neighbor.
        """
        b, q = _indices(record, b, q, self.device)
        valid = self._valid(record, b, q)
        masks = _selected(record, "masks", b, q, self.device, torch.float32)
        n, t = valid.shape
        if masks.shape != valid.shape:
            raise ValueError("Masks and validity must have the same time axis")
        foreground = (masks.sigmoid() > .5) & valid
        positions = torch.arange(t, device=self.device).expand(n, t)
        first = positions.masked_fill(~foreground, t).amin(1)
        last = positions.masked_fill(~foreground, -1).amax(1)
        fallback = first >= last
        for row in torch.nonzero(fallback, as_tuple=False).flatten().tolist():
            available = torch.nonzero(valid[row], as_tuple=False).flatten()
            if available.numel() < 2:
                first[row] = last[row] = -1
                continue
            logits = masks[row, available]
            # nan/inf are not a sensible peak; deterministic first-valid tie.
            logits = torch.nan_to_num(logits, nan=-1.0e9,
                                      posinf=1.0e9, neginf=-1.0e9)
            peak = available[logits.argmax()]
            neighbors = available[available != peak]
            nearest = neighbors[(neighbors - peak).abs().argmin()]
            first[row], last[row] = torch.minimum(peak, nearest), torch.maximum(peak, nearest)
        return first, last, fallback

    def scores(self, record, b, q):
        """Return [N,T,T] scores for selected final-layer queries only."""
        b, q = _indices(record, b, q, self.device)
        dtype = self.gate.dtype
        feature = _selected(record, "f", b, q, self.device, dtype)
        valid = self._valid(record, b, q)
        n, dim, t = feature.shape
        if dim != self.dim or valid.shape != (n, t):
            raise ValueError("Frozen feature dimensions do not match this head")
        if t == 0:
            raise ValueError("A nonempty time axis is required")
        if n == 0:
            return self.gate * torch.zeros((0, t, t), device=self.device, dtype=dtype)
        query = _selected(record, "hs", b, q, self.device, dtype)
        classes = _selected(record, "cls", b, q, self.device, dtype)
        features = F.layer_norm(feature.transpose(1, 2), (dim,))
        queries = F.layer_norm(query, (dim,))
        h = self.feature(features) + self.query(queries)[:, None]
        h = F.gelu(h + self.classes(classes.softmax(-1))[:, None])
        h = h * valid[:, :, None]
        h = F.gelu(self.local(h.transpose(1, 2)).transpose(1, 2))
        h = h * valid[:, :, None]
        residual = self.endpoint(h)
        u, v = self.u(h), self.v(h)
        if self.mode == "joint":
            pair = torch.matmul(u, v.transpose(1, 2)) / math.sqrt(self.rank)
        else:
            diagonal = (u * v).sum(-1) / math.sqrt(self.rank)
            pair = .5 * (diagonal[:, :, None] + diagonal[:, None, :])
        first, last, _ = self.anchor(record, b, q)
        position = torch.arange(t, device=self.device, dtype=dtype)
        start_prior = -(position[None] - first[:, None]).abs() / self.anchor_scale
        end_prior = -(position[None] - last[:, None]).abs() / self.anchor_scale
        score = (start_prior[:, :, None] + end_prior[:, None, :]
                 + residual[:, :, 0, None] + residual[:, None, :, 1]
                 + self.gate * pair)
        legal = (position[:, None] < position[None, :])[None]
        legal = legal & valid[:, :, None] & valid[:, None, :]
        return score.masked_fill(~legal, INVALID_SCORE)

    def decode(self, record, b, q):
        """Maximum-score inclusive endpoints; (-1,-1) when no legal span."""
        score = self.scores(record, b, q)
        n, t, _ = score.shape
        flat = score.reshape(n, t * t)
        ix = flat.argmax(-1)
        first, last = ix // t, ix % t
        feasible = flat.amax(-1) > INVALID_SCORE / 2
        return first.masked_fill(~feasible, -1), last.masked_fill(~feasible, -1)


def target_span_loss(scores, first, last):
    """Mean joint CE over legal nonsingleton targets, plus eligible count.

    Invalid, singleton and padding targets are explicitly excluded. Empty
    support returns a differentiable zero, never a NaN. Callers must separately
    record excluded counts; this function makes no assignment or GT matching.
    """
    if scores.ndim != 3 or scores.shape[-2] != scores.shape[-1]:
        raise ValueError("scores must have shape [N,T,T]")
    n, t, _ = scores.shape
    first = torch.as_tensor(first, device=scores.device, dtype=torch.long).reshape(-1)
    last = torch.as_tensor(last, device=scores.device, dtype=torch.long).reshape(-1)
    if first.numel() != n or last.numel() != n:
        raise ValueError("Need one first/last target per selected query")
    eligible = (first >= 0) & (last < t) & (first < last)
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if rows.numel():
        supported = scores[rows, first[rows], last[rows]] > INVALID_SCORE / 2
        rows = rows[supported]
    count = int(rows.numel())
    if not count:
        # Multiplication before sum avoids overflow of summed invalid logits.
        return (scores * 0).sum(), 0
    target = first[rows] * t + last[rows]
    return F.cross_entropy(scores[rows].reshape(count, t * t), target), count
