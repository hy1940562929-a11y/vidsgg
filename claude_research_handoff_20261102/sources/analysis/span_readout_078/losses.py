"""075 frozen-assignment losses; all averages are per matched GT, then per layer.

The new ranking domain excludes every retained episode of the same predicate.
Absence of an annotation is dataset supervision, not a verified visual negative.
"""
import torch
from torch.nn import functional as F

ARMS = ('original', 'balanced', 'ordered')


def point_focal(z, y):
    p = z.sigmoid()
    pt = p * y + (1 - p) * (1 - y)
    return F.binary_cross_entropy_with_logits(z, y, reduction='none') * (1 - pt).square() * (.25 * y + .75 * (1 - y))


def focal_rows(z, y, valid):
    v = valid.bool()
    f = point_focal(z, y) * v
    original = f.mean(-1)
    pos, neg = v & (y > .5), v & (y <= .5)
    np_, nn = pos.sum(-1), neg.sum(-1)
    positive = (f * pos).sum(-1) / np_.clamp_min(1)
    negative = (f * neg).sum(-1) / nn.clamp_min(1)
    balanced = (positive + negative) / ((np_ > 0).to(z.dtype) + (nn > 0).to(z.dtype)).clamp_min(1)
    return original, balanced


def spread(mask, limit=16):
    ix = torch.nonzero(mask, as_tuple=False).flatten()
    if len(ix) <= limit:
        return ix
    positions = torch.linspace(0, len(ix) - 1, limit, device=ix.device).round().long()
    return ix[positions]


def rank_domain(y, same_predicate_union, valid, censored=False):
    """One-cell erosion of the current positive and dilation of all positives."""
    v = valid.bool()
    current = (y > .5) & v
    union = same_predicate_union.bool() & v
    left = F.pad(current[:-1], (1, 0), value=False)
    right = F.pad(current[1:], (0, 1), value=False)
    positive = current & left & right
    protected = F.max_pool1d(union.float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
    negative = v & ~protected
    reason = 'eligible'
    if censored:
        reason = 'censored_pair'
    elif not positive.any():
        reason = 'no_guarded_positive'
    elif not negative.any():
        reason = 'no_guarded_negative'
    eligible = reason == 'eligible'
    return dict(positive=spread(positive) if eligible else torch.empty(0, dtype=torch.long, device=y.device),
                negative=spread(negative) if eligible else torch.empty(0, dtype=torch.long, device=y.device),
                reason=reason, positive_cells=int(positive.sum()), negative_cells=int(negative.sum()))


def rank_rows(z, domains):
    # Batch all <=16x16 point comparisons; placeholders are excluded by masks.
    width = 16
    p = torch.zeros((len(domains), width), dtype=torch.long, device=z.device)
    n = torch.zeros_like(p)
    pm = torch.zeros_like(p, dtype=torch.bool)
    nm = torch.zeros_like(pm)
    for i, d in enumerate(domains):
        pi, ni = d['positive'], d['negative']
        p[i, :len(pi)], n[i, :len(ni)] = pi, ni
        pm[i, :len(pi)], nm[i, :len(ni)] = True, True
    mask = pm[:, :, None] & nm[:, None, :]
    values = F.softplus(1. - z.gather(1, p)[:, :, None] + z.gather(1, n)[:, None, :])
    return (values * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1), mask.any((1, 2))


def compose_focal(original, balanced, rank, eligible, arm, k_b, k_r):
    if arm == 'original':
        return original
    b = k_b * balanced
    if arm == 'balanced':
        return b
    if arm != 'ordered':
        raise ValueError(arm)
    return torch.where(eligible, .5 * b + .5 * k_r * rank, b)


def loss(model, values, record, arm, scales, check_original=False):
    from models.losses import masked_dice_loss
    total = values.sum() * 0
    summaries = []
    for layer, m in enumerate(record['match']):
        z = values[layer, m['b'], m['q']]
        y, valid = m['y'], m['valid']
        a, b = focal_rows(z, y, valid)
        rank, eligible = rank_rows(z, m['rank_domains'])
        focal = compose_focal(a, b, rank, eligible, arm, scales['k_B'], scales['k_R']).mean()
        nt = z.new_tensor(float(m['N']))
        dice = masked_dice_loss(z, y, nt, valid)
        value = model.loss_factor['loss_mask'] * focal + model.loss_factor['loss_dice'] * dice
        if check_original:
            native = model.loss_masks(record['cls'][layer], values[layer], record['preds'], record['targets'], None, m['indices'], nt, valid)
            assert torch.allclose(model.loss_factor['loss_mask'] * a.mean(), native['loss_mask'], atol=1e-7, rtol=1e-6)
            assert torch.allclose(dice * model.loss_factor['loss_dice'], native['loss_dice'], atol=1e-7, rtol=1e-6)
        total = total + value
        summaries.append(torch.stack((a.mean(), b.mean(), rank.mean(), focal, dice)))
    return total, torch.stack(summaries).detach()
