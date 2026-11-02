"""E101 focal arms: N native mean, T unit-positive mass, X long-biased positives."""
import math
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


def masked_focal_arm(inputs, targets, num_masks, loss_mask, arm="N"):
    """Per-event focal with arm-specific positive weights. Shape [E,T]."""
    loss = pointwise_focal(inputs, targets)
    mask = loss_mask.to(loss.dtype).detach()
    loss = loss * mask
    B = inputs.shape[0]
    out = inputs.new_zeros(())
    weight_stats = []
    V_list = []
    d_list = []
    for i in range(B):
        m = mask[i]
        y = targets[i]
        phi = loss[i]
        V = float(m.sum().item())
        d = float((m * y).sum().item())
        V_list.append(V)
        d_list.append(d)
        if V <= 0:
            continue
        if arm == "N" or d <= 0 or abs(d - V) < 1e-12:
            # Native mean; full-positive mask makes T==N.
            val = phi.sum() / max(V, 1.0)
            w_pos = 1.0
        elif arm == "T":
            w = torch.ones_like(phi)
            w = torch.where((m * y) > 0.5, torch.full_like(w, V / max(d, 1.0)), w)
            denom = (w * m).sum().clamp(min=1.0)
            val = (w * phi).sum() / denom
            w_pos = float(V / max(d, 1.0))
        elif arm == "X":
            # Placeholder weight; batch V_ref applied by caller via arm_x_with_vref
            raise RuntimeError("Use masked_focal_arm_x for X")
        else:
            raise RuntimeError("UNKNOWN_ARM")
        out = out + val
        weight_stats.append(dict(V=V, d=d, w_pos=w_pos, short=d < (V * 0.5)))  # coarse; duration via caller
    return out / float(num_masks), weight_stats


def masked_focal_arm_x(inputs, targets, num_masks, loss_mask, v_ref):
    loss = pointwise_focal(inputs, targets)
    mask = loss_mask.to(loss.dtype).detach()
    loss = loss * mask
    B = inputs.shape[0]
    out = inputs.new_zeros(())
    weight_stats = []
    v_ref = max(float(v_ref), 1.0)
    for i in range(B):
        m = mask[i]
        y = targets[i]
        phi = loss[i]
        V = float(m.sum().item())
        d = float((m * y).sum().item())
        if V <= 0:
            continue
        if d <= 0:
            val = phi.sum() / max(V, 1.0)
            w_pos = 1.0
        else:
            w = torch.ones_like(phi)
            w_pos = max(d, 1.0) / v_ref
            w = torch.where((m * y) > 0.5, torch.full_like(w, w_pos), w)
            denom = (w * m).sum().clamp(min=1.0)
            val = (w * phi).sum() / denom
        out = out + val
        weight_stats.append(dict(V=V, d=d, w_pos=w_pos))
    return out / float(num_masks), weight_stats


def loss_masks_arm(rt, pred_logits, pred_masks, gt_preds, gt_masks, indices, num_masks, loss_mask, arm):
    """Mirror MaskVRD.loss_masks but swap focal for arm variant; Dice native."""
    src_idx = rt.model._get_src_permutation_idx(indices)
    pred = pred_masks[src_idx]
    target_masks = torch.cat([mask[indices[idx][1]] for idx, mask in enumerate(gt_masks)])
    assert pred.shape == target_masks.shape
    # Dice native
    dice = rt.model.loss_factor["loss_dice"] * rt.dice(pred, target_masks, num_masks, loss_mask)
    if arm == "N":
        # Use repository scripted focal for exact native path
        focal = rt.model.loss_factor["loss_mask"] * rt.focal(pred, target_masks, num_masks, loss_mask)
        mask_f = loss_mask.to(pred.dtype)
        stats = []
        for i in range(pred.shape[0]):
            V = float(mask_f[i].sum().item())
            d = float((mask_f[i] * target_masks[i]).sum().item())
            if V > 0:
                stats.append(dict(V=V, d=d, w_pos=1.0))
    elif arm == "T":
        raw, stats = masked_focal_arm(pred, target_masks, num_masks, loss_mask, arm="T")
        focal = rt.model.loss_factor["loss_mask"] * raw
    elif arm == "X":
        # V_ref = mean valid length over matched events in this layer call
        mask_f = loss_mask.to(pred.dtype)
        V_list = [float(mask_f[i].sum().item()) for i in range(pred.shape[0]) if float(mask_f[i].sum().item()) > 0]
        v_ref = sum(V_list) / max(len(V_list), 1)
        raw, stats = masked_focal_arm_x(pred, target_masks, num_masks, loss_mask, v_ref)
        focal = rt.model.loss_factor["loss_mask"] * raw
    else:
        raise RuntimeError(arm)
    return {"loss_mask": focal, "loss_dice": dice}, stats


def total_loss_arm(rt, output, batch, matches, arm):
    ce = []
    mask = []
    all_stats = []
    num = output["pred_masks"].new_tensor(float(max(sum(len(x) for x in batch["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(output), matches):
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        lm_dict, stats = loss_masks_arm(rt, layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], indices, num, lm, arm)
        mask.append(sum(lm_dict.values()))
        all_stats.extend(stats)
    return sum(ce) + sum(mask), all_stats
