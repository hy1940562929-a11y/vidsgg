"""Duration-biased Hungarian matching for E110. Loss coeffs stay native."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models.losses import batch_masked_sigmoid_focal_loss, batch_masked_dice_loss


def gt_durations(batch):
    """List[List[float]] aligned with batch['preds']."""
    out = []
    for meta in batch["c"]["metadata"]:
        out.append([float(e["duration_seconds"]) for e in meta["events"]])
    return out


def duration_weight(d, arm, alpha=1.0, d_ref=3.0):
    d = max(float(d), 0.25)
    if arm == "N":
        return 1.0
    if arm == "T":
        return (d / d_ref) ** alpha  # short → smaller cost multiplier
    if arm == "X":
        return (d_ref / d) ** alpha  # long → smaller
    raise RuntimeError(arm)


def bipartite_match_biased(model, pred_logits, gt_preds, pred_masks, gt_masks, _mask, durations, arm,
                           alpha=1.0, d_ref=3.0):
    """Same as MaskVRD.bipartite_match but scales GT columns by duration weight."""
    bs, num_queries = pred_logits.shape[:2]
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
    _C = (
        model.cost_factor["cost_class"] * cost_class
        + model.cost_factor["cost_mask"] * cost_mask
        + model.cost_factor["cost_dice"] * cost_dice
    )
    _C = _C.view(bs, num_queries, -1).detach().cpu()
    sizes = [len(p) for p in gt_preds]
    # scale columns by duration (global concat order)
    col = 0
    for b, durs in enumerate(durations):
        require_len = sizes[b]
        assert len(durs) == require_len, (len(durs), require_len)
        for d in durs:
            w = duration_weight(d, arm, alpha=alpha, d_ref=d_ref)
            _C[b, :, col] = _C[b, :, col] * w
            col += 1
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(_C.split(sizes, -1))]
    loss_mask = torch.cat(
        [torch.stack([_mask[idx].squeeze(0)] * sizes[idx], dim=0) for idx in range(len(sizes))],
        dim=0,
    )
    return (
        [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],
        loss_mask,
    )


def matches_arm(rt, output, batch, arm):
    durations = gt_durations(batch)
    layers = rt.layers(output)
    out = []
    for layer in layers:
        if arm == "N":
            out.append(rt.model.bipartite_match(
                layer["pred_logits"], batch["preds"], layer["pred_masks"], batch["targets"],
                None, batch["valid"]))
        else:
            out.append(bipartite_match_biased(
                rt.model, layer["pred_logits"], batch["preds"], layer["pred_masks"], batch["targets"],
                batch["valid"], durations, arm))
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
    """Fraction of (layer,batch,gt) where T vs N matched query differs; for admission."""
    mN = matches_arm(rt, o, b, "N")
    mT = matches_arm(rt, o, b, "T")
    mX = matches_arm(rt, o, b, "X")
    def as_maps(matches):
        # list over layers of list over batch of dict gt->query
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
