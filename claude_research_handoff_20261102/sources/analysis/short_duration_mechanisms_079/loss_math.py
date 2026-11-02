"""079-C interval-risk objective and diagnostics; torch is imported lazily.

Intervals use original-video, half-open frame coordinates.  There is no
normalization inherited from experiment 043, no hard-negative mining, and no
GT-dependent decoding in this module.  GT is used only to train or score outputs.
"""
from __future__ import annotations

import math


def interval_iou(prediction, truth):
    """Half-open interval IoU, including a zero score for no prediction."""
    if prediction is None:
        return 0.0
    a, b = map(float, prediction)
    c, d = map(float, truth)
    if not all(math.isfinite(v) for v in (a, b, c, d)) or b <= a or d <= c:
        return 0.0
    overlap = max(0.0, min(b, d) - max(a, c))
    return overlap / (b - a + d - c - overlap)


def decoded_frame_interval(first, last, origin, stride, n_valid):
    """Mirror the retained VrdONE hull convention, not cell-width expansion.

    first/last are inclusive sampled indices; the end is last*stride+1,
    NOT (last+1)*stride. Native empty/singleton predictions are invalid here.
    """
    if not (0 <= first < last < n_valid):
        return None
    return [int(origin + stride * first), int(origin + stride * last + 1)]


def endpoint_metrics(prediction, truth, fps):
    if fps <= 0:
        raise ValueError("fps must be positive")
    if prediction is None:
        return dict(start_abs_frames=None, end_abs_frames=None,
                    endpoint_abs_seconds=None, endpoint_relative=None)
    duration = truth[1] - truth[0]
    if duration <= 0:
        raise ValueError("GT intervals must have positive duration")
    start = abs(prediction[0] - truth[0])
    end = abs(prediction[1] - truth[1])
    return dict(start_abs_frames=start, end_abs_frames=end,
                endpoint_abs_seconds=(start + end) / (2 * fps),
                endpoint_relative=(start + end) / (2 * duration))


def _eligible(scores, first, last):
    import torch
    if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
        raise ValueError("scores must have shape [N,T,T]")
    n, t, _ = scores.shape
    first = torch.as_tensor(first, dtype=torch.long, device=scores.device).reshape(-1)
    last = torch.as_tensor(last, dtype=torch.long, device=scores.device).reshape(-1)
    if len(first) != n or len(last) != n:
        raise ValueError("one GT span is required per selected query")
    eligible = (first >= 0) & (last < t) & (first < last)
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if len(rows):
        rows = rows[scores[rows, first[rows], last[rows]] > -5.0e8]
    return rows, first, last


def interval_objective(scores, first, last, origins, strides, gt_intervals,
                       risk_weight=0.0):
    """Joint interval CE + fixed-weight expected true-frame IoU risk.

    p(i,j) = softmax of legal span scores; risk = E_p[1-tIoU(span,GT)].
    CE and risk use exactly the same eligible nonsingleton GT rows. The risk is
    averaged per retained event, with lambda=1 for the intervention arm. No
    duration label affects decoding and no development-set scale is fitted.
    """
    import torch
    import torch.nn.functional as F
    if risk_weight not in (0.0, 1.0):
        raise ValueError("079-C pre-registers risk weights 0 or 1 only")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite interval logits")
    rows, first, last = _eligible(scores, first, last)
    n, t, _ = scores.shape
    origins = torch.as_tensor(origins, dtype=scores.dtype, device=scores.device).reshape(-1)
    strides = torch.as_tensor(strides, dtype=scores.dtype, device=scores.device).reshape(-1)
    gt = torch.as_tensor(gt_intervals, dtype=scores.dtype, device=scores.device).reshape(-1, 2)
    if len(origins) != n or len(strides) != n or len(gt) != n:
        raise ValueError("one coordinate mapping and original GT interval per row")
    if (strides <= 0).any() or (gt[:, 1] <= gt[:, 0]).any():
        raise ValueError("invalid stride or GT interval")
    if not len(rows):
        zero = (scores * 0).sum()
        return zero, dict(ce=zero, risk=zero, eligible=0)
    z = scores[rows]
    legal = z > -5.0e8
    flat = z.masked_fill(~legal, float("-inf")).flatten(1)
    ce = F.cross_entropy(flat, first[rows] * t + last[rows])
    # Original-frame support uses the native decode's end+1, at every stride.
    pos = torch.arange(t, dtype=scores.dtype, device=scores.device)
    starts = origins[rows, None] + strides[rows, None] * pos[None]
    ends = starts + 1
    a = starts[:, :, None]
    b = ends[:, None, :]
    g0, g1 = gt[rows, 0, None, None], gt[rows, 1, None, None]
    overlap = (torch.minimum(b, g1) - torch.maximum(a, g0)).clamp(min=0)
    union = (b - a) + (g1 - g0) - overlap
    quality = torch.where(legal, overlap / union.clamp(min=1), 0).clamp(0, 1)
    probability = F.softmax(flat, dim=1).reshape_as(z)
    risk = (probability * (1 - quality)).sum((1, 2)).mean()
    loss = ce + risk_weight * risk
    return loss, dict(ce=ce, risk=risk, eligible=int(len(rows)))


GROUPS = {
    "all": lambda d: True,
    "subsecond": lambda d: d < 1,
    "one_to_three": lambda d: 1 <= d < 3,
    "three_to_ten": lambda d: 3 <= d < 10,
    "ten_plus": lambda d: d >= 10,
    "short": lambda d: d < 3,
    "long": lambda d: d >= 3,
}


def advancement_gate(final):
    """Fixed diagnostic gate; cannot pass by merely losing less than CE.

    This is a single-seed triage criterion, not a significance test. It requires
    gains against each contemporaneous control and the two immutable anchors.
    """
    checks = {}
    for baseline in ("ce", "native", "zero_update"):
        for role, group, strict in (("train", "short", True),
                                    ("adapter_dev", "short", True),
                                    ("adapter_dev", "all", False),
                                    ("adapter_dev", "long", False)):
            change = final[role]["groups"][group]["arms"]["ce_iou"]["comparisons"][baseline]
            checks[f"{role}:{group}:vs_{baseline}"] = dict(
                **change, criterion="net>0" if strict else "net>=0",
                passed=change["net"] > 0 if strict else change["net"] >= 0)
    passed = all(c["passed"] for c in checks.values())
    return dict(passed=passed, checks=checks,
                decision="PROMISING_DIAGNOSTIC_REQUIRES_REVIEW" if passed else
                         "NO_POSITIVE_DEVELOPMENT_GATE", automatically_expand=False)


def summarize_rows(rows, arms=("native", "zero_update", "ce", "ce_iou")):
    """Report event counts with all retained GT, including training exclusions.

    This is fixed matched-query localization, not detection AP/recall. Endpoint
    error means use explicitly reported valid-output denominators; an empty
    prediction remains a zero-IoU miss in every hit/IoU denominator.
    """
    def mean(values):
        return sum(values) / len(values) if values else None
    result = {}
    for name, keep in GROUPS.items():
        selected = [r for r in rows if keep(r["relation_sec"])]
        group = dict(N_GT=len(selected),
                     eligible_loss_GT=sum(r["loss_eligible"] for r in selected),
                     singleton_GT=sum(r["singleton_GT"] for r in selected),
                     N_videos=len({r["video_name"] for r in selected}), arms={})
        for arm in arms:
            entries = [r["arms"][arm] for r in selected]
            hits = sum(v["true_frame_tIoU"] >= .5 for v in entries)
            endpoints = [v for v in entries if v["endpoint_relative"] is not None]
            changes = {}
            for baseline in ("native", "zero_update", "ce"):
                added = sum(r["arms"][arm]["true_frame_tIoU"] >= .5 and
                            r["arms"][baseline]["true_frame_tIoU"] < .5 for r in selected)
                lost = sum(r["arms"][arm]["true_frame_tIoU"] < .5 and
                           r["arms"][baseline]["true_frame_tIoU"] >= .5 for r in selected)
                changes[baseline] = dict(added=added, lost=lost, net=added-lost)
            group["arms"][arm] = dict(hits=hits, denominator=len(selected),
                hit_fraction=hits/len(selected) if selected else None,
                mean_true_frame_tIoU=mean([v["true_frame_tIoU"] for v in entries]),
                valid_endpoint_denominator=len(endpoints),
                mean_abs_endpoint_seconds=mean([v["endpoint_abs_seconds"] for v in endpoints]),
                mean_relative_endpoint_error=mean([v["endpoint_relative"] for v in endpoints]),
                comparisons=changes)
        result[name] = group
    return result
