"""Unvalidated049 prototype. Frozen-query endpoint baseline, not a novelty claim."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def project(head, record):
    layers, batch, queries, _ = record['cls'].shape
    emb = head(record['hs']).view(layers, batch, -1, queries).permute(0, 1, 3, 2)
    return torch.einsum('lbqc,bct->lbqt', emb, record['f'])


class EndpointProjection(nn.Module):
    def __init__(self, original_mask_projection):
        super().__init__()
        self.start = copy.deepcopy(original_mask_projection)
        self.end = copy.deepcopy(original_mask_projection)
        self.requires_grad_(True)

    def forward(self, record):
        valid = record['valid'][None, ...]
        s = project(self.start, record).masked_fill(~valid, -torch.inf)
        e = project(self.end, record).masked_fill(~valid, -torch.inf)
        return s, e


def best_interval(start, end, valid):
    """Legal sampled endpoints j>=i+1 (stride4/min5 frames), O(T).

    Equal scores: latest maximizing start for each prefix, then earliest end.
    No valid legal pair raises, so caller must handle empty/singleton input.
    """
    if start.shape != end.shape or valid.shape != start.shape:
        raise ValueError('Matching [...,T] arrays required')
    if valid.dtype != torch.bool or start.shape[-1] < 2:
        raise ValueError('At least two samples and bool validity required')
    s = start.masked_fill(~valid, -torch.inf)
    e = end.masked_fill(~valid, -torch.inf)
    values, indices = torch.cummax(s[..., :-1], dim=-1)
    joint = values + e[..., 1:]
    j = joint.argmax(-1)
    if not torch.isfinite(joint.gather(-1, j[..., None])).all():
        raise ValueError('No legal finite interval')
    i = indices.gather(-1, j[..., None]).squeeze(-1)
    return i, j + 1


def endpoint_ce(start, end, first, last):
    """Independent endpoint CE per matched target; caller validates targets.

    Singleton targets remain visible endpoint labels. They cannot be decoded
    exactly under min5 and must be counted separately in the frozen protocol.
    """
    return F.cross_entropy(start, first, reduction='none') + F.cross_entropy(end, last, reduction='none')
