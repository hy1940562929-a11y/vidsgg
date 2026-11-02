"""E102: exterior negative focal upweight (T) vs misplaced-window negative upweight (X)."""
import hashlib
import torch
import torch.nn.functional as F


def pointwise_focal(inputs, targets, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def _beta(V, d):
    return V / max(d, 1.0)


def _misplaced_positive_mask(y, m, key: str):
    """Deterministic misplaced contiguous window of length d on valid support."""
    device = y.device
    T = y.numel()
    valid_idx = torch.where(m > 0.5)[0]
    if valid_idx.numel() == 0:
        return y.clone()
    d = int((m * y).sum().item())
    if d <= 0 or d >= int(valid_idx.numel()):
        return y.clone()
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    Vn = int(valid_idx.numel())
    # try shifts until different from true positive set
    true = set(torch.where((m * y) > 0.5)[0].tolist())
    start = h % max(Vn - d + 1, 1)
    for k in range(Vn):
        s = (start + k) % max(Vn - d + 1, 1)
        cand = set(valid_idx[s : s + d].tolist())
        if cand != true:
            out = torch.zeros_like(y)
            out[valid_idx[s : s + d]] = 1.0
            return out
    # fallback circular shift of positives on full timeline
    out = torch.roll(y, shifts=1 + (h % max(T - 1, 1)))
    return out * m


def focal_arm(inputs, targets, num_masks, loss_mask, arm="N", keys=None):
    """inputs/targets/loss_mask: [E,T]. keys: list[str] length E for X."""
    phi = pointwise_focal(inputs, targets) * loss_mask.to(inputs.dtype)
    mask = loss_mask.to(inputs.dtype)
    B = inputs.shape[0]
    out = inputs.new_zeros(())
    stats = []
    for i in range(B):
        m = mask[i]
        y = targets[i]
        V = float(m.sum().item())
        d = float((m * y).sum().item())
        if V <= 0:
            continue
        if arm == "N" or d <= 0 or abs(d - V) < 1e-12:
            val = phi[i].sum() / max(V, 1.0)
            w_neg = 1.0
        elif arm == "T":
            beta = _beta(V, d)
            w = torch.ones_like(phi[i])
            neg = (m > 0.5) & (y <= 0.5)
            w = torch.where(neg, torch.full_like(w, beta), w)
            denom = (w * m).sum().clamp(min=1.0)
            val = (w * phi[i]).sum() / denom
            w_neg = float(beta)
        elif arm == "X":
            beta = _beta(V, d)
            key = keys[i] if keys is not None else f"row{i}"
            y_tilde = _misplaced_positive_mask(y, m, key)
            w = torch.ones_like(phi[i])
            neg = (m > 0.5) & (y_tilde <= 0.5)
            w = torch.where(neg, torch.full_like(w, beta), w)
            denom = (w * m).sum().clamp(min=1.0)
            val = (w * phi[i]).sum() / denom
            w_neg = float(beta)
        else:
            raise RuntimeError(arm)
        out = out + val
        stats.append(dict(V=V, d=d, w_neg=w_neg))
    return out / float(num_masks), stats


def loss_masks_arm(rt, pred_masks, gt_masks, indices, num_masks, loss_mask, arm, keys=None):
    src_idx = rt.model._get_src_permutation_idx(indices)
    pred = pred_masks[src_idx]
    target_masks = torch.cat([mask[indices[idx][1]] for idx, mask in enumerate(gt_masks)])
    assert pred.shape == target_masks.shape
    dice = rt.model.loss_factor["loss_dice"] * rt.dice(pred, target_masks, num_masks, loss_mask)
    if arm == "N":
        focal = rt.model.loss_factor["loss_mask"] * rt.focal(pred, target_masks, num_masks, loss_mask)
        mask_f = loss_mask.to(pred.dtype)
        stats = []
        for i in range(pred.shape[0]):
            V = float(mask_f[i].sum().item())
            d = float((mask_f[i] * target_masks[i]).sum().item())
            if V > 0:
                stats.append(dict(V=V, d=d, w_neg=1.0))
    else:
        raw, stats = focal_arm(pred, target_masks, num_masks, loss_mask, arm=arm, keys=keys)
        focal = rt.model.loss_factor["loss_mask"] * raw
    return {"loss_mask": focal, "loss_dice": dice}, stats


def match_keys(batch, indices):
    """Stable keys per matched target for X misplaced windows."""
    meta = batch["c"]["metadata"]
    if isinstance(meta, dict):
        meta = [meta]
    keys = []
    for bi, (src, tgt) in enumerate(indices):
        m = meta[bi] if bi < len(meta) else meta[0]
        vid = m.get("video_id") or m.get("video_name")
        pair = m.get("pair_key") if m.get("pair_key") is not None else m.get("pair_idx")
        for gi in tgt.tolist():
            keys.append(f"{vid}|{pair}|{int(gi)}")
    return keys


def total_loss_arm(rt, output, batch, matches, arm):
    ce = []
    mask = []
    all_stats = []
    num = output["pred_masks"].new_tensor(float(max(sum(len(x) for x in batch["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(output), matches):
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        keys = match_keys(batch, indices) if arm == "X" else None
        lm_dict, stats = loss_masks_arm(rt, layer["pred_masks"], batch["targets"], indices, num, lm, arm, keys=keys)
        mask.append(sum(lm_dict.values()))
        all_stats.extend(stats)
    return sum(ce) + sum(mask), all_stats
