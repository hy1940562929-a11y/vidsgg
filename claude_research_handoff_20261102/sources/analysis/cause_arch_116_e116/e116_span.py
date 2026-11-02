"""Moment-DETR-style (cx, w) span head for E113."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanHead(nn.Module):
    def __init__(self, n_embd=256, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hs):
        # hs: (B, Q, C) → (B, Q, 2) as (cx, w) via sigmoid
        return self.mlp(hs).sigmoid()


def mask_to_cxw(target_row, valid_row):
    t = target_row.float()
    v = valid_row.reshape(-1).float()
    L = int(v.sum().item())
    if L <= 0:
        return 0.5, max(1.0 / 512.0, 1e-3)
    m = (t[:L] > 0.5).float()
    if float(m.sum()) <= 0:
        return 0.5, max(1.0 / max(L, 1), 1e-3)
    idx = torch.nonzero(m, as_tuple=False).reshape(-1).float()
    start = float(idx.min())
    end = float(idx.max()) + 1.0
    w = max((end - start) / float(L), 1e-3)
    cx = ((start + end) * 0.5) / float(L)
    return min(max(cx, 0.0), 1.0), min(max(w, 1e-3), 1.0)


def gt_cxw_batch(batch):
    """List of (G,2) tensors per batch item."""
    out = []
    for b in range(batch["x"].shape[0]):
        targets = batch["targets"][b]
        valid = batch["valid"][b, 0]
        rows = []
        for g in range(int(targets.shape[0])):
            cx, w = mask_to_cxw(targets[g], valid)
            rows.append([cx, w])
        out.append(torch.tensor(rows, device=batch["x"].device, dtype=batch["x"].dtype))
    return out


def shuffle_cxw(gt_list, generator=None):
    out = []
    for g in gt_list:
        if g.numel() == 0:
            out.append(g)
            continue
        G = g.shape[0]
        if G == 1:
            # random span
            if generator is not None:
                cx = float(torch.rand((), generator=generator))
                w = float(torch.rand((), generator=generator)) * 0.5 + 0.05
            else:
                cx = float(torch.rand(())); w = float(torch.rand(())) * 0.5 + 0.05
            out.append(torch.tensor([[cx, w]], device=g.device, dtype=g.dtype))
        else:
            if generator is not None:
                perm = torch.randperm(G, generator=generator)
            else:
                perm = torch.randperm(G)
            out.append(g[perm.to(g.device)])
    return out


def forward_with_span(predictor, batch, span_head):
    """Native-like forward returning outputs + hs for span (B,Q,C)."""
    x = batch["x"]
    mask_features = batch["f"]
    key_mask = batch["key_mask"]
    output_mask = batch["valid"]
    non_attn_const = -10
    src = predictor.input_norm(x)
    if predictor.input_proj is not None:
        src = predictor.input_proj(src)
        src = src * key_mask.to(src.dtype)
    B = src.shape[0]
    content = predictor.query_embed.weight.permute(1, 0).unsqueeze(0).expand(B, -1, -1).contiguous()
    Q = content.shape[-1]
    tgt = torch.zeros_like(content)
    tgt_mask = torch.ones(B, 1, Q, dtype=torch.bool, device=src.device)
    hs, _ = predictor.transformer.decoder(
        tgt, memory=src, tgt_mask=tgt_mask, memory_mask=key_mask, pos=None,
        query_pos=content, cross_first=False,
    )
    # hs: (n_layer, B, C, Q)
    n_layer, bs, n_channel, n_query = hs.shape
    hs_embed = hs.view(n_layer * bs, n_channel, n_query)
    outputs_class = predictor.class_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    mask_embed = predictor.mask_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    outputs_seg_masks = torch.einsum("lbqc,bcm->lbqm", mask_embed, mask_features)
    outputs_seg_masks = outputs_seg_masks.masked_fill(torch.logical_not(output_mask[None, ...]), float(non_attn_const))
    # last layer query features (B, Q, C)
    hs_last = hs[-1].permute(0, 2, 1).contiguous()
    pred_spans = span_head(hs_last)  # (B,Q,2)
    out = {
        "pred_logits": outputs_class[-1],
        "pred_masks": outputs_seg_masks[-1],
        "pred_spans": pred_spans,
        "aux_outputs": [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ],
        "output_mask": output_mask,
    }
    return out



def giou_1d(cx_a, w_a, cx_b, w_b, eps=1e-6):
    a0 = cx_a - 0.5 * w_a; a1 = cx_a + 0.5 * w_a
    b0 = cx_b - 0.5 * w_b; b1 = cx_b + 0.5 * w_b
    inter = (torch.minimum(a1, b1) - torch.maximum(a0, b0)).clamp_min(0.0)
    union = (a1 - a0) + (b1 - b0) - inter
    iou = inter / union.clamp_min(eps)
    enclose = (torch.maximum(a1, b1) - torch.minimum(a0, b0)).clamp_min(eps)
    return iou - (enclose - union) / enclose


def span_l1_giou_loss(pred_spans, matches_last, gt_cxw, lambda_l1=5.0, lambda_giou=2.0):
    indices, _ = matches_last
    l1s, gious = [], []
    for b, (src, tgt) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred = pred_spans[b, src]
        gt = gt_cxw[b][tgt]
        l1s.append(F.l1_loss(pred, gt, reduction="mean"))
        gious.append((1.0 - giou_1d(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])).mean())
    if not l1s:
        return pred_spans.new_zeros(())
    return lambda_l1 * sum(l1s) / len(l1s) + lambda_giou * sum(gious) / len(gious)
