"""Coexistence-length FiLM on content query embeddings for E109."""
import torch
import torch.nn as nn


class CoexistFiLM(nn.Module):
    def __init__(self, n_embd=256, hidden=128):
        super().__init__()
        self.n_embd = n_embd
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_embd),
        )
        # near-identity: last layer zeros → γ=1, β=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, coexist):
        """coexist: (B,) in ~[0,1] → gamma, beta each (B, C)."""
        h = self.mlp(coexist.reshape(-1, 1).to(dtype=self.mlp[0].weight.dtype))
        dgamma, beta = h.chunk(2, dim=-1)
        gamma = 1.0 + dgamma
        return gamma, beta


def coexist_lengths(batch):
    """Per-sample support length / 512 from valid mask."""
    # valid: (B,1,T)
    v = batch["valid"].reshape(batch["valid"].shape[0], -1).float()
    return v.sum(dim=-1) / float(v.shape[-1])


def prepare_coexist(batch, arm, generator=None):
    c = coexist_lengths(batch)
    if arm == "N":
        return None
    if arm == "T":
        return c
    if arm == "X":
        B = c.shape[0]
        if B == 1:
            return (1.0 - c).clamp(0.0, 1.0)
        # Always permute on CPU (generator is CPU), then index on device
        if generator is not None:
            perm = torch.randperm(B, generator=generator)
        else:
            perm = torch.randperm(B)
        return c[perm.to(c.device)]
    raise RuntimeError(arm)


def forward_with_film(predictor, batch, film, arm, generator=None):
    """Native-like predictor forward; T/X modulate query embeds via FiLM."""
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
    # content queries (B,C,9)
    content = predictor.query_embed.weight.permute(1, 0).unsqueeze(0).expand(B, -1, -1).contiguous()
    if arm in ("T", "X") and film is not None:
        coexist = prepare_coexist(batch, arm, generator=generator)
        gamma, beta = film(coexist)  # (B,C)
        content = gamma[:, :, None] * content + beta[:, :, None]

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
    out = {
        "pred_logits": outputs_class[-1],
        "pred_masks": outputs_seg_masks[-1],
        "aux_outputs": [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ],
        "output_mask": output_mask,
    }
    return out
