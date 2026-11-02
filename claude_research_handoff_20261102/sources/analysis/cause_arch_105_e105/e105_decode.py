"""E105 soft-mass quantile interval decode vs native hull vs time-reversed soft."""
import torch


def native_hull_candidates(masks, classes, n_valid, origin, label_prefix):
    """masks: [Q,T] probabilities or logits? use sigmoid outside. classes: list[Q]."""
    cs = []
    for q in range(masks.shape[0]):
        ix = torch.where(masks[q, :n_valid] > 0.5)[0]
        if len(ix) and int(ix[-1] - ix[0]) * 4 + 1 >= 5:
            cs.append(
                dict(
                    start=origin + 4 * int(ix[0]),
                    end=origin + 4 * int(ix[-1]) + 1,
                    predicate_id=classes[q],
                    native_query=q,
                    candidate_id=f"{label_prefix}/{q}",
                )
            )
    return cs


def soft_quantile_candidates(logits, classes, n_valid, origin, label_prefix, q_lo=0.1, q_hi=0.9, reverse=False):
    """logits [Q,T]; decode each query via mass quantiles on sigmoid*valid."""
    cs = []
    T = logits.shape[1]
    for q in range(logits.shape[0]):
        z = logits[q, :n_valid].float()
        if reverse:
            z = torch.flip(z, dims=[0])
        p = torch.sigmoid(z)
        Z = float(p.sum().item())
        if Z <= 1e-8:
            continue
        rho = p / Z
        cdf = torch.cumsum(rho, dim=0)
        # first index where cdf >= q_lo / q_hi
        a = int(torch.searchsorted(cdf, torch.tensor(q_lo, device=cdf.device)).item())
        b = int(torch.searchsorted(cdf, torch.tensor(q_hi, device=cdf.device)).item())
        a = max(0, min(a, n_valid - 1))
        b = max(0, min(b, n_valid - 1))
        if b < a:
            a, b = b, a
        if reverse:
            # map indices back to original time
            a_orig = n_valid - 1 - b
            b_orig = n_valid - 1 - a
            a, b = a_orig, b_orig
        if int(b - a) * 4 + 1 < 5:
            # expand symmetrically to min5 frames if possible
            need = 5
            # in sample space: length samples L satisfies (L-1)*4+1 >= 5 => L >= 2
            while int(b - a) * 4 + 1 < 5 and (a > 0 or b < n_valid - 1):
                if a > 0:
                    a -= 1
                if int(b - a) * 4 + 1 >= 5:
                    break
                if b < n_valid - 1:
                    b += 1
        if int(b - a) * 4 + 1 < 5:
            continue
        cs.append(
            dict(
                start=origin + 4 * int(a),
                end=origin + 4 * int(b) + 1,
                predicate_id=classes[q],
                native_query=q,
                candidate_id=f"{label_prefix}/{q}",
            )
        )
    return cs
