"""Temporal GIoU matching cost for E114. Loss coeffs stay native; decode stays hull."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models.losses import batch_masked_sigmoid_focal_loss, batch_masked_dice_loss


def mask_to_span01(mask, valid, eps=1e-6):
    """Soft center-width in [0,1] from a 1D mask logit/prob row. mask,valid: [T]."""
    if mask.dim() != 1:
        raise RuntimeError("mask_to_span01 expects [T]")
    T = mask.numel()
    t = torch.linspace(0.0, 1.0, T, device=mask.device, dtype=mask.dtype)
    m = mask.sigmoid() * valid.float()
    mass = m.sum().clamp_min(eps)
    cx = (m * t).sum() / mass
    w = (2.0 * (m * (t - cx).abs()).sum() / mass).clamp_min(eps)
    w = w.clamp(max=1.0)
    return cx, w


def giou_1d_cost(cx_a, w_a, cx_b, w_b, eps=1e-6):
    """1 - GIoU for 1D intervals given as center/width in [0,1]. Returns scalar tensor."""
    a0 = cx_a - 0.5 * w_a
    a1 = cx_a + 0.5 * w_a
    b0 = cx_b - 0.5 * w_b
    b1 = cx_b + 0.5 * w_b
    inter = (torch.minimum(a1, b1) - torch.maximum(a0, b0)).clamp_min(0.0)
    union = (a1 - a0) + (b1 - b0) - inter
    iou = inter / union.clamp_min(eps)
    enclose = (torch.maximum(a1, b1) - torch.minimum(a0, b0)).clamp_min(eps)
    giou = iou - (enclose - union) / enclose
    return 1.0 - giou


def gt_spans_from_targets(targets, valid):
    """List over batch of list of (cx,w) from GT masks."""
    spans = []
    for bi, tgt in enumerate(targets):
        v = valid[bi, 0]
        row = []
        for g in range(tgt.shape[0]):
            row.append(mask_to_span01(tgt[g], v))
        spans.append(row)
    return spans


def shuffle_gt_spans(spans, generator):
    """Permute temporal (cx,w) across GTs within each batch item (X control)."""
    out = []
    for row in spans:
        if len(row) <= 1:
            out.append(list(row))
            continue
        order = torch.randperm(len(row), generator=generator).tolist()
        cxws = [(row[i][0], row[i][1]) for i in order]
        # keep count; assign shuffled geometry
        out.append(cxws)
    return out


def bipartite_match_tgiou(model, pred_logits, gt_preds, pred_masks, gt_masks, _mask, gt_spans,
                          lambda_tgiou=2.0):
    """Native class+mask+dice plus λ·(1−GIoU) between pred soft-span and GT span."""
    bs, num_queries = pred_logits.shape[:2]
    T = pred_masks.shape[-1]
    out_logits = pred_logits.flatten(0, 1)
    out_mask = pred_masks.flatten(0, 1)
    batch_out_mask = _mask.repeat(1, num_queries, 1).flatten(0, 1)
    tgt_ids = torch.cat(gt_preds, dim=0)
    tgt_mask = torch.cat(gt_masks, dim=0)
    batch_tgt_mask = []
    for i in range(bs):
        batch_tgt_mask += [_mask[i, 0]] * len(gt_masks[i])
    batch_tgt_mask = torch.stack(batch_tgt_mask, dim=0)

    out_logits_exp = out_logits[:, :, None].repeat(1, 1, len(tgt_ids))
    tgt_ids_exp = tgt_ids[None, :].repeat(out_logits_exp.shape[0], 1)
    cost_class = F.cross_entropy(out_logits_exp, tgt_ids_exp, reduction="none")
    cost_mask = batch_masked_sigmoid_focal_loss(
        out_mask, tgt_mask, batch_out_mask=batch_out_mask, batch_tgt_mask=batch_tgt_mask)
    cost_dice = batch_masked_dice_loss(
        out_mask, tgt_mask, batch_out_mask=batch_out_mask, batch_tgt_mask=batch_tgt_mask)

    # cost_tgiou: [Q_total, G_total]
    device = pred_masks.device
    dtype = pred_masks.dtype
    Q = bs * num_queries
    G = int(tgt_ids.numel())
    cost_tgiou = torch.zeros(Q, G, device=device, dtype=dtype)
    # precompute pred spans per query
    pred_cxw = []
    for qi in range(Q):
        b = qi // num_queries
        pred_cxw.append(mask_to_span01(out_mask[qi], batch_out_mask[qi]))
    flat_gt = []
    for row in gt_spans:
        flat_gt.extend(row)
    assert len(flat_gt) == G
    for qi in range(Q):
        pcx, pw = pred_cxw[qi]
        for gi, (gcx, gw) in enumerate(flat_gt):
            # gcx/gw may be tensors
            cost_tgiou[qi, gi] = giou_1d_cost(pcx, pw, gcx.to(device=device, dtype=dtype),
                                              gw.to(device=device, dtype=dtype))

    _C = (
        model.cost_factor["cost_class"] * cost_class
        + model.cost_factor["cost_mask"] * cost_mask
        + model.cost_factor["cost_dice"] * cost_dice
        + lambda_tgiou * cost_tgiou
    )
    _C = _C.view(bs, num_queries, -1).detach().cpu()
    sizes = [len(p) for p in gt_preds]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(_C.split(sizes, -1))]
    loss_mask = torch.cat(
        [torch.stack([_mask[idx].squeeze(0)] * sizes[idx], dim=0) for idx in range(len(sizes))],
        dim=0,
    )
    return (
        [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],
        loss_mask,
    )


def matches_arm(rt, output, batch, arm, lambda_tgiou=2.0, seed_key=0):
    layers = rt.layers(output)
    spans = gt_spans_from_targets(batch["targets"], batch["valid"])
    if arm == "X":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed_key) % (2**31))
        spans = shuffle_gt_spans(spans, gen)
    out = []
    for layer in layers:
        if arm == "N":
            out.append(rt.model.bipartite_match(
                layer["pred_logits"], batch["preds"], layer["pred_masks"], batch["targets"],
                None, batch["valid"]))
        else:
            out.append(bipartite_match_tgiou(
                rt.model, layer["pred_logits"], batch["preds"], layer["pred_masks"], batch["targets"],
                batch["valid"], spans, lambda_tgiou=lambda_tgiou))
    return out


def native_loss_with_matches(rt, o, b, matches):
    ce, mask = [], []
    num = o["pred_masks"].new_tensor(float(max(sum(len(x) for x in b["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(o), matches):
        args = (layer["pred_logits"], layer["pred_masks"], b["preds"], b["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        mask.append(sum(rt.model.loss_masks(*args).values()))
    return sum(ce) + sum(mask)


def match_disagreement(rt, o, b):
    mN = matches_arm(rt, o, b, "N")
    mT = matches_arm(rt, o, b, "T", seed_key=0)
    mX = matches_arm(rt, o, b, "X", seed_key=1)
    def as_maps(matches):
        maps = []
        for indices, _ in matches:
            layer_maps = []
            for src, tgt in indices:
                d = {int(t): int(s) for s, t in zip(src.tolist(), tgt.tolist())}
                layer_maps.append(d)
            maps.append(layer_maps)
        return maps
    n, t, x = as_maps(mN), as_maps(mT), as_maps(mX)
    diffs_tn = diffs_tx = total = 0
    for li in range(len(n)):
        for bi in range(len(n[li])):
            keys = set(n[li][bi]) | set(t[li][bi]) | set(x[li][bi])
            for k in keys:
                total += 1
                if n[li][bi].get(k) != t[li][bi].get(k):
                    diffs_tn += 1
                if t[li][bi].get(k) != x[li][bi].get(k):
                    diffs_tx += 1
    return diffs_tn / max(total, 1), diffs_tx / max(total, 1)
