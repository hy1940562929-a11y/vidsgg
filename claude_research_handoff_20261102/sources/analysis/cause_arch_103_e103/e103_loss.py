"""E103 unit-mass W1 mask supervision (T) vs uniform target (X); N native focal+dice."""
import torch
import torch.nn.functional as F


def unit_mass_w1(logits, target_mask, valid_mask, mode="true"):
    """Per-event W1 between softmax_valid(logits) and target mass.
    target_mask, valid_mask: [T] or [E,T]; logits [E,T].
    mode: 'true' -> rho* = 1_GT/d; 'uniform' -> rho* = 1_valid/V
    Returns scalar mean over events, and list of per-event w1.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        target_mask = target_mask.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
    E, T = logits.shape
    w1s = []
    total = logits.new_zeros(())
    n = 0
    for i in range(E):
        m = valid_mask[i] > 0.5
        if m.sum() == 0:
            continue
        u = logits[i]
        # masked softmax
        neg_inf = torch.finfo(u.dtype).min
        u_m = torch.where(m, u, torch.full_like(u, neg_inf))
        rho = torch.softmax(u_m, dim=0)
        rho = torch.where(m, rho, torch.zeros_like(rho))
        # renorm numerically
        s = rho.sum().clamp(min=1e-12)
        rho = rho / s
        V = float(m.sum().item())
        d = float((target_mask[i] * valid_mask[i]).sum().item())
        if mode == "uniform" or d <= 0:
            rho_star = torch.where(m, torch.ones_like(rho) / max(V, 1.0), torch.zeros_like(rho))
        else:
            rho_star = torch.where(
                (target_mask[i] > 0.5) & m,
                torch.full_like(rho, 1.0 / d),
                torch.zeros_like(rho),
            )
            # ensure mass 1 on valid
            rho_star = rho_star / rho_star.sum().clamp(min=1e-12)
        # CDF W1 on grid (Δ=1 sample); normalize by max(d,1)
        F = torch.cumsum(rho, dim=0)
        F_star = torch.cumsum(rho_star, dim=0)
        w1 = (F - F_star).abs().sum()
        # duration norm: d samples
        denom = max(d, 1.0) if mode == "true" and d > 0 else max(V, 1.0)
        w1 = w1 / denom
        total = total + w1
        w1s.append(float(w1.detach().item()))
        n += 1
    if n == 0:
        return logits.new_zeros(()), w1s
    return total / n, w1s


def loss_masks_arm(rt, pred_masks, gt_masks, indices, num_masks, loss_mask, arm):
    src_idx = rt.model._get_src_permutation_idx(indices)
    pred = pred_masks[src_idx]
    target_masks = torch.cat([mask[indices[idx][1]] for idx, mask in enumerate(gt_masks)])
    assert pred.shape == target_masks.shape
    COEF = 7.0  # loss_mask(2)+loss_dice(5)
    stats = []
    if arm == "N":
        focal = rt.model.loss_factor["loss_mask"] * rt.focal(pred, target_masks, num_masks, loss_mask)
        dice = rt.model.loss_factor["loss_dice"] * rt.dice(pred, target_masks, num_masks, loss_mask)
        # diagnostic W1 toward true for exposure
        with torch.no_grad():
            _, w1s = unit_mass_w1(pred.detach(), target_masks, loss_mask, mode="true")
        for w in w1s:
            stats.append(dict(w1_true=w, arm="N"))
        return {"loss_mask": focal, "loss_dice": dice}, stats
    mode = "true" if arm == "T" else "uniform"
    w1, w1s = unit_mass_w1(pred, target_masks, loss_mask, mode=mode)
    with torch.no_grad():
        _, true_list = unit_mass_w1(pred.detach(), target_masks, loss_mask, mode="true")
        _, uni_list = unit_mass_w1(pred.detach(), target_masks, loss_mask, mode="uniform")
    for i, w in enumerate(w1s):
        stats.append(dict(
            w1_train=w,
            w1_true=true_list[i] if i < len(true_list) else w,
            w1_uniform=uni_list[i] if i < len(uni_list) else w,
            arm=arm,
        ))
    # single term under loss_mask key; dice zero
    zero = pred.new_zeros(())
    return {"loss_mask": pred.new_tensor(COEF) * w1, "loss_dice": zero}, stats


def total_loss_arm(rt, output, batch, matches, arm):
    ce = []
    mask = []
    all_stats = []
    num = output["pred_masks"].new_tensor(float(max(sum(len(x) for x in batch["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(output), matches):
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        lm_dict, stats = loss_masks_arm(rt, layer["pred_masks"], batch["targets"], indices, num, lm, arm)
        mask.append(sum(lm_dict.values()))
        all_stats.extend(stats)
    return sum(ce) + sum(mask), all_stats
