"""ActionFormer/FCOS-style centerness weights for mask focal+dice (E117)."""
import torch


def gt_center_width_frames(target_row, valid_row, eps=1e-6):
    """Center/width in frame indices from binary GT mask row [T] and valid [T]."""
    v = valid_row.float()
    y = target_row.float() * v
    T = y.numel()
    idx = torch.arange(T, device=y.device, dtype=y.dtype)
    mass = y.sum().clamp_min(eps)
    cx = (y * idx).sum() / mass
    # width ≈ 2 E[|t-cx|] on positive mass; fallback to mass count
    w = (2.0 * (y * (idx - cx).abs()).sum() / mass).clamp_min(1.0)
    if float(mass) < 0.5:
        # empty GT: neutral center
        cx = (idx * v).sum() / v.sum().clamp_min(eps)
        w = v.sum().clamp_min(1.0)
    return cx, w


def gaussian_centerness(T, cx, w, valid, device, dtype):
    """Gaussian centerness on frames; renormalize so sum(w)=sum(valid)."""
    idx = torch.arange(T, device=device, dtype=dtype)
    sigma = (0.5 * w).clamp_min(1.0)
    g = torch.exp(-0.5 * ((idx - cx) / sigma) ** 2)
    g = g * valid.float().to(dtype=dtype)
    s = g.sum()
    vs = valid.float().sum().clamp_min(1.0)
    if float(s) < 1e-8:
        return valid.float().to(dtype=dtype)
    return g * (vs / s)


def build_weight_maps(batch, mode, generator=None):
    """Per-batch-item list of [G,T] weight maps for each GT.
    mode: 'T' true GT cxw; 'X' shuffled cxw within item; 'N' unused.
    """
    weights = []
    for bi, tgt in enumerate(batch["targets"]):
        valid = batch["valid"][bi, 0]
        T = tgt.shape[-1]
        G = tgt.shape[0]
        cxws = [gt_center_width_frames(tgt[g], valid) for g in range(G)]
        if mode == "X" and G > 1:
            order = torch.randperm(G, generator=generator).tolist()
            cxws = [cxws[i] for i in order]
        rows = []
        for g in range(G):
            cx, w = cxws[g]
            rows.append(gaussian_centerness(T, cx, w, valid, tgt.device, tgt.dtype))
        weights.append(torch.stack(rows, dim=0) if rows else tgt.new_zeros(0, T))
    return weights


def losses_arm(rt, output, batch, arm, seed_key=0):
    """Native CE + mask focal/dice; T/X replace loss_mask with centerness maps."""
    matches = rt.matches(output, batch)
    ce, mask_terms = [], []
    num = output["pred_masks"].new_tensor(float(max(sum(len(p) for p in batch["preds"]), 1)))
    weight_maps = None
    if arm in ("T", "X"):
        gen = None
        if arm == "X":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed_key) % (2 ** 31))
        weight_maps = build_weight_maps(batch, arm, generator=gen)
    for layer, (indices, lm) in zip(rt.layers(output), matches):
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        if arm == "N":
            mask_terms.append(sum(rt.model.loss_masks(*args).values()))
            continue
        # Rebuild mask loss with centerness loss_mask (coeffs 2/5 native).
        layer_loss = layer["pred_masks"].new_zeros(())
        for b, (queries, targets) in enumerate(indices):
            for q, target in zip(queries.tolist(), targets.tolist()):
                z = layer["pred_masks"][b, q:q + 1]
                y = batch["targets"][b][target:target + 1]
                w = weight_maps[b][target:target + 1]
                one = z.new_tensor(1.0)
                layer_loss = layer_loss + 2 * rt.focal(z, y, one, w) + 5 * rt.dice(z, y, one, w)
        # Match native loss_masks scale: per-event terms summed then / num
        mask_terms.append(layer_loss / num)
    return sum(ce) + sum(mask_terms), matches, weight_maps


def weight_disagreement(rt, output, batch):
    """Mean L1 between T and N effective masks on matched short GTs (admission)."""
    matches = rt.matches(output, batch)
    # last layer only
    indices, _ = matches[-1]
    maps_t = build_weight_maps(batch, "T")
    diffs, n = [], 0
    for b, (queries, targets) in enumerate(indices):
        valid = batch["valid"][b, 0].float()
        for q, target in zip(queries.tolist(), targets.tolist()):
            meta = batch["c"]["metadata"][b]["events"][target]
            if float(meta.get("duration_seconds", 99)) >= 3:
                continue
            w = maps_t[b][target]
            # N uses binary valid broadcast
            d = (w - valid).abs().sum() / valid.sum().clamp_min(1.0)
            diffs.append(float(d))
            n += 1
    if not diffs:
        return 0.0, 0
    return sum(diffs) / len(diffs), n
