"""Expand query bank 9→12 with optional short/long priors on new slots."""
import math
import torch
import torch.nn as nn


def expand_query_embed(predictor, extra=3):
    """In-place expand predictor.query_embed from 9 to 9+extra; copy first 9."""
    old = predictor.query_embed
    n_embd = old.embedding_dim
    n_old = old.num_embeddings
    n_new = n_old + extra
    new = nn.Embedding(n_new, n_embd)
    with torch.no_grad():
        new.weight[:n_old].copy_(old.weight.data)
        nn.init.normal_(new.weight[n_old:], std=0.02)
    new = new.to(old.weight.device)
    predictor.query_embed = new
    predictor.num_queries = n_new
    return n_new


class SpanPrior(nn.Module):
    """Additive prior on extra query slots from a scalar log-duration prototype."""
    def __init__(self, n_embd=256, extra=3):
        super().__init__()
        self.extra = extra
        self.proj = nn.Linear(1, n_embd * extra)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def bias_for(self, kind):
        # kind: short → log(0.5/3), long → log(12/3) relative to 3s ref in log-seconds
        if kind == "short":
            x = math.log(0.5 / 3.0)
        elif kind == "long":
            x = math.log(12.0 / 3.0)
        else:
            raise RuntimeError(kind)
        t = self.proj.weight.new_tensor([[x]])
        return self.proj(t).view(self.extra, -1)  # (extra, C)


def apply_prior_to_extra(predictor, prior_module, kind):
    """Return query weight view with prior added to last extra rows (not in-place on embed)."""
    w = predictor.query_embed.weight
    extra = prior_module.extra
    if kind == "none":
        return w
    bias = prior_module.bias_for(kind)  # (extra, C)
    w2 = w.clone()
    w2[-extra:] = w2[-extra:] + bias
    return w2


def forward_with_query_weight(predictor, batch, query_weight):
    """Forward using provided query_weight (Q, C) or (n_embd rows). query_embed.weight is (Q,C)."""
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
    # query_weight: (Q, C) like Embedding.weight; decoder wants (C, Q)
    content = query_weight.permute(1, 0).unsqueeze(0).expand(B, -1, -1).contiguous()
    Q = content.shape[-1]
    tgt = torch.zeros_like(content)
    tgt_mask = torch.ones(B, 1, Q, dtype=torch.bool, device=src.device)
    hs, _ = predictor.transformer.decoder(
        tgt, memory=src, tgt_mask=tgt_mask, memory_mask=key_mask, pos=None,
        query_pos=content, cross_first=False,
    )
    n_layer, bs, n_channel, n_query = hs.shape
    hs_embed = hs.view(n_layer * bs, n_channel, n_query)
    outputs_class = predictor.class_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    mask_embed = predictor.mask_embed(hs_embed).view(n_layer, bs, -1, n_query).permute(0, 1, 3, 2)
    outputs_seg_masks = torch.einsum("lbqc,bcm->lbqm", mask_embed, mask_features)
    outputs_seg_masks = outputs_seg_masks.masked_fill(torch.logical_not(output_mask[None, ...]), float(non_attn_const))
    return {
        "pred_logits": outputs_class[-1],
        "pred_masks": outputs_seg_masks[-1],
        "aux_outputs": [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ],
        "output_mask": output_mask,
    }
