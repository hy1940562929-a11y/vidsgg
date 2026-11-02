"""E104: depth-mean mask aggregation (T) vs earliest-layer mask (X) vs native (N)."""
import torch


def stack_masks(rt, output):
    layers = rt.layers(output)  # 4 dicts with pred_masks, pred_logits
    masks = torch.stack([L["pred_masks"] for L in layers], dim=0)  # [4,B,Q,T]
    return layers, masks


def fused_mask(masks, mode="mean"):
    if mode == "mean":
        return masks.mean(dim=0)
    if mode == "early":
        return masks[0]
    if mode == "last":
        return masks[-1]
    raise RuntimeError(mode)


def native_total_loss(rt, output, batch, matches):
    """Full native aux+final class/mask loss (same as E101 native_loss_direct)."""
    ce = []
    mask = []
    num = output["pred_masks"].new_tensor(float(max(sum(len(x) for x in batch["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(output), matches):
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        mask.append(sum(rt.model.loss_masks(*args).values()))
    return sum(ce) + sum(mask)


def agg_match_and_loss(rt, output, batch, mode):
    """Single Hungarian on (last class, fused mask); one class+mask loss; no aux."""
    layers, masks = stack_masks(rt, output)
    last = layers[-1]
    fm = fused_mask(masks, mode=mode)
    # temporary output-like for matcher: use last logits + fused masks
    indices, loss_mask = rt.model.bipartite_match(
        last["pred_logits"], batch["preds"], fm, batch["targets"], None, batch["valid"]
    )
    num = fm.new_tensor(float(max(sum(len(x) for x in batch["preds"]), 1)))
    args = (last["pred_logits"], fm, batch["preds"], batch["targets"], None, indices, num, loss_mask)
    loss = rt.model.loss_labels(*args)["loss_class"] + sum(rt.model.loss_masks(*args).values())
    return loss, [(indices, loss_mask)], fm


def total_loss_arm(rt, output, batch, matches, arm):
    stats = []
    if arm == "N":
        loss = native_total_loss(rt, output, batch, matches)
        # exposure: L1 between mean and last
        with torch.no_grad():
            _, masks = stack_masks(rt, output)
            diff = (masks.mean(0) - masks[-1]).abs().mean().item()
        stats.append(dict(fuse_last_l1=diff))
        return loss, stats
    mode = "mean" if arm == "T" else "early"
    loss, _, fm = agg_match_and_loss(rt, output, batch, mode)
    with torch.no_grad():
        _, masks = stack_masks(rt, output)
        diff = (fm - masks[-1]).abs().mean().item()
    stats.append(dict(fuse_last_l1=diff, mode=mode))
    return loss, stats


def infer_masks(rt, output, arm):
    """Return [B,Q,T] mask logits for hull."""
    layers, masks = stack_masks(rt, output)
    if arm == "N" or arm == "C0":
        return layers[-1]["pred_masks"], layers[-1]["pred_logits"]
    if arm == "T":
        return fused_mask(masks, "mean"), layers[-1]["pred_logits"]
    if arm == "X":
        return fused_mask(masks, "early"), layers[-1]["pred_logits"]
    raise RuntimeError(arm)
