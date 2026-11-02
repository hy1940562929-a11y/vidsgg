"""Train-only DN short queries for E108 (parallel DN forward; eval stays native 9)."""
import math
import torch
import torch.nn as nn


class DNEncoder(nn.Module):
    def __init__(self, n_embd=256, num_classes=50):
        super().__init__()
        self.n_embd = n_embd
        self.class_embed = nn.Embedding(num_classes + 1, n_embd)
        self.span_mlp = nn.Sequential(
            nn.Linear(2, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )
        nn.init.normal_(self.class_embed.weight, std=0.02)
        for m in self.span_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, cx, log_w, class_id):
        span = torch.stack([cx, log_w], dim=-1)
        return self.class_embed(class_id) + self.span_mlp(span)


def _span_from_mask(target_row, valid_row):
    t = target_row.float()
    v = valid_row.reshape(-1).float()
    L = int(v.sum().item())
    if L <= 0:
        return 0.5, math.log(1.0 / 512.0)
    m = (t[:L] > 0.5).float()
    if float(m.sum()) <= 0:
        return 0.5, math.log(1.0 / max(L, 1))
    idx = torch.nonzero(m, as_tuple=False).reshape(-1).float()
    start = float(idx.min())
    end = float(idx.max()) + 1.0
    w = max(end - start, 1.0)
    cx = (start + end) * 0.5 / float(L)
    log_w = math.log(w / float(L))
    return cx, log_w


def build_dn_specs(batch, arm, noise_scale=0.25, generator=None):
    specs = []
    B = batch["x"].shape[0]
    for b in range(B):
        preds = batch["preds"][b]
        targets = batch["targets"][b]
        valid = batch["valid"][b, 0]
        G = int(preds.numel())
        spans = []
        for g in range(G):
            cx, log_w = _span_from_mask(targets[g], valid)
            spans.append([cx, log_w, int(preds[g].item()), g])
        if arm == "X" and G > 1:
            if generator is not None:
                order = torch.randperm(G, generator=generator).tolist()
            else:
                order = torch.randperm(G).tolist()
            new = []
            for g in range(G):
                cx, log_w = spans[order[g]][0], spans[order[g]][1]
                cls, gi = spans[g][2], spans[g][3]
                new.append([cx, log_w, cls, gi])
            spans = new
        elif arm == "X" and G == 1:
            cls, gi = spans[0][2], spans[0][3]
            if generator is not None:
                rc = float(torch.rand((), generator=generator))
                rw = float(torch.rand((), generator=generator)) * 0.5 + 0.05
            else:
                rc = float(torch.rand(()))
                rw = float(torch.rand(())) * 0.5 + 0.05
            spans = [[rc, math.log(max(rw, 1e-4)), cls, gi]]
        out = []
        for cx, log_w, cls, gi in spans:
            if arm == "T":
                if generator is not None:
                    n1 = float(torch.randn((), generator=generator)) * noise_scale
                    n2 = float(torch.randn((), generator=generator)) * noise_scale
                else:
                    n1 = float(torch.randn(())) * noise_scale
                    n2 = float(torch.randn(())) * noise_scale
                cx = min(max(cx + n1 * 0.15, 0.0), 1.0)
                log_w = log_w + n2 * 0.25
            out.append((cx, log_w, cls, gi))
        specs.append(out)
    return specs


def pack_dn_queries(encoder, specs, device, dtype):
    B = len(specs)
    K = max((len(s) for s in specs), default=0)
    C = encoder.n_embd
    if K == 0:
        return (torch.zeros(B, C, 0, device=device, dtype=dtype),
                torch.zeros(B, 0, device=device, dtype=torch.bool),
                torch.full((B, 0), -1, device=device, dtype=torch.long))
    q = torch.zeros(B, C, K, device=device, dtype=dtype)
    active = torch.zeros(B, K, device=device, dtype=torch.bool)
    gt_index = torch.full((B, K), -1, device=device, dtype=torch.long)
    for b, spans in enumerate(specs):
        if not spans:
            continue
        cx = torch.tensor([s[0] for s in spans], device=device, dtype=dtype)
        log_w = torch.tensor([s[1] for s in spans], device=device, dtype=dtype)
        cls = torch.tensor([s[2] for s in spans], device=device, dtype=torch.long)
        emb = encoder.encode(cx, log_w, cls)
        kb = emb.shape[0]
        q[b, :, :kb] = emb.T
        active[b, :kb] = True
        gt_index[b, :kb] = torch.tensor([s[3] for s in spans], device=device, dtype=torch.long)
    return q, active, gt_index


def forward_dn_only(predictor, batch, dn_encoder, arm, generator=None):
    """Parallel DN-only decoder forward (no content queries → no self-attn crosstalk)."""
    x = batch["x"]
    mask_features = batch["f"]
    key_mask = batch["key_mask"]
    output_mask = batch["valid"]
    non_attn_const = -10

    src = predictor.input_norm(x)
    if predictor.input_proj is not None:
        src = predictor.input_proj(src)
        src = src * key_mask.to(src.dtype)

    specs = build_dn_specs(batch, arm, generator=generator)
    dn_q, dn_active, dn_gt_index = pack_dn_queries(dn_encoder, specs, src.device, src.dtype)
    B, C, K = dn_q.shape
    if K == 0:
        return None

    tgt = torch.zeros_like(dn_q)
    tgt_mask = torch.ones(B, 1, K, dtype=torch.bool, device=src.device)
    # zero-out inactive query slots in tgt_mask so they do not attend
    tgt_mask = tgt_mask & dn_active[:, None, :]
    hs, _ = predictor.transformer.decoder(
        tgt, memory=src, tgt_mask=tgt_mask, memory_mask=key_mask, pos=None,
        query_pos=dn_q, cross_first=False,
    )
    n_layer, bs, n_channel, n_query = hs.shape
    hs_embed = hs.view(n_layer * bs, n_channel, n_query)
    outputs_class = predictor.class_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    mask_embed = predictor.mask_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    outputs_seg_masks = torch.einsum("lbqc,bcm->lbqm", mask_embed, mask_features)
    outputs_seg_masks = outputs_seg_masks.masked_fill(torch.logical_not(output_mask[None, ...]), float(non_attn_const))
    out = {
        "pred_logits": outputs_class[-1],
        "pred_masks": outputs_seg_masks[-1],
        "aux_outputs": [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ],
        "dn_active": dn_active,
        "dn_gt_index": dn_gt_index,
    }
    return out


def dn_indices(dn_out):
    active = dn_out["dn_active"]
    gt_index = dn_out["dn_gt_index"]
    indices = []
    B, K = active.shape
    for b in range(B):
        src, tgt = [], []
        for k in range(K):
            if bool(active[b, k]):
                src.append(k)
                tgt.append(int(gt_index[b, k].item()))
        indices.append((
            torch.tensor(src, dtype=torch.int64, device=active.device),
            torch.tensor(tgt, dtype=torch.int64, device=active.device),
        ))
    return indices


def dn_aux_loss(rt, dn_out, batch):
    if dn_out is None:
        return batch["x"].new_zeros(())
    indices = dn_indices(dn_out)
    if all(len(s) == 0 for s, _ in indices):
        return batch["x"].new_zeros(())
    layers = list(dn_out.get("aux_outputs", [])) + [
        {"pred_logits": dn_out["pred_logits"], "pred_masks": dn_out["pred_masks"]}
    ]
    sizes = [len(p) for p in batch["preds"]]
    loss_mask = torch.cat(
        [torch.stack([batch["valid"][i].squeeze(0)] * sizes[i], dim=0) for i in range(len(sizes))],
        dim=0,
    )
    num = dn_out["pred_masks"].new_tensor(float(max(sum(sizes), 1)))
    ce, mask = [], []
    for layer in layers:
        args = (layer["pred_logits"], layer["pred_masks"], batch["preds"], batch["targets"], None, indices, num, loss_mask)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        mask.append(sum(rt.model.loss_masks(*args).values()))
    return sum(ce) + sum(mask)
