"""CDSE / KTE / LGSAD research implementation, not a validated research result.

All public feature tensors use [batch, time, channels]. Predicate logits use
[batch, predicate, time]. Predicate indices here are ZERO-based; native VrdONE
foreground category IDs are ONE-based (0 is its background class).

Forward never accepts annotations. Text vectors must come from a real frozen
text encoder in real experiments; synthetic vectors are for software tests only.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class EnhancementConfig:
    pair_dim: int
    roi_dim: int
    geometry_dim: int
    text_dim: int
    hidden_dim: int = 128
    context_seconds: float = 2.0
    exclude_seconds: float = 0.25
    background_threshold: float = 0.1
    min_background_steps: int = 2
    predicate_chunk_size: int = 8
    residual_limit: float = 0.5
    use_cdse: bool = True
    use_kte: bool = True
    use_language: bool = True

    def __post_init__(self):
        for name in ("pair_dim", "roi_dim", "geometry_dim", "text_dim", "hidden_dim",
                     "min_background_steps", "predicate_chunk_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if (not math.isfinite(self.context_seconds) or not math.isfinite(self.exclude_seconds)
                or not 0 <= self.exclude_seconds < self.context_seconds):
            raise ValueError("Require 0 <= exclude_seconds < context_seconds")
        if not 0 < self.background_threshold < 0.5:
            raise ValueError("background_threshold must lie in (0, 0.5)")
        if not 0 < self.residual_limit <= 1:
            raise ValueError("residual_limit must lie in (0, 1]")

    def to_dict(self):
        return asdict(self)


def load_text_anchors(path, expected_predicates: Sequence[str], allow_synthetic=False) -> Tensor:
    """Load a locally generated cache with exact category-order verification.

    Requires a PyTorch version supporting weights_only=True (tested on 2.2.2).
    No encoder download or random-prototype fallback is performed.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("predicates") != list(expected_predicates):
        raise ValueError("Text anchor predicate names/order differ from native IDs 1..P")
    if payload.get("synthetic") is not False and not allow_synthetic:
        raise ValueError("Real runs require a nonsynthetic, traceable text-anchor cache")
    if not allow_synthetic and not payload.get("checkpoint_sha256"):
        raise ValueError("Missing text-checkpoint provenance")
    embeddings = payload.get("embeddings")
    if (not isinstance(embeddings, Tensor) or embeddings.ndim != 2
            or embeddings.shape[0] != len(expected_predicates)
            or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()
            or (embeddings.norm(dim=-1) <= 1e-8).any()):
        raise ValueError("Invalid cached text embeddings")
    return embeddings.detach().float()


def time_steps(value: Union[float, Tensor], batch: int, device) -> Tensor:
    """Seconds per feature step, not seconds per raw video frame."""
    dt = torch.as_tensor(value, device=device, dtype=torch.float32).flatten()
    if dt.numel() == 1:
        dt = dt.expand(batch)
    if dt.shape != (batch,) or not torch.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("seconds_per_step must be a positive scalar or [B]")
    return dt


def masked_difference(x: Tensor, valid: Tensor, dt: Tensor) -> Tensor:
    """No difference crosses a missing/padded position; derivative per second."""
    if x.ndim != 3 or valid.shape != x.shape[:2]:
        raise ValueError("Expected x [B,T,C] and valid [B,T]")
    x = torch.where(valid[..., None], x, torch.zeros_like(x))
    previous = F.pad(x[:, :-1], (0, 0, 1, 0))
    adjacent = valid & F.pad(valid[:, :-1], (1, 0), value=False)
    delta = (x - previous) / dt[:, None, None].to(x.dtype)
    return torch.where(adjacent[..., None], delta, torch.zeros_like(delta))


def local_background_reference(
    features: Tensor, probabilities: Tensor, valid: Tensor,
    seconds_per_step: Union[float, Tensor], context_seconds: float = 2.0,
    exclude_seconds: float = 0.25, threshold: float = 0.1,
    min_steps: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Prediction-only, predicate-conditioned weighted local reference.

    features [B,T,H], probabilities [B,P,T], valid [B,T]. Returns reference
    [B,P,T,H] and availability [B,P,T]. All selectors are detached. No [T,T]
    attention map is constructed: prefix sums implement the two-sided window.
    A confident *prediction* is still not guaranteed to be genuine background.
    """
    if features.ndim != 3 or probabilities.ndim != 3:
        raise ValueError("Invalid feature/probability dimensions")
    b, t, h = features.shape
    if t < 1 or valid.dtype != torch.bool or valid.shape != (b, t):
        raise ValueError("valid must be bool [B,T], T >= 1")
    if probabilities.shape[0] != b or probabilities.shape[2] != t:
        raise ValueError("probabilities must be [B,P,T]")
    if not 0 <= exclude_seconds < context_seconds or not 0 < threshold < 0.5:
        raise ValueError("Invalid context parameters")
    if min_steps < 1:
        raise ValueError("min_steps must be >= 1")
    dt = time_steps(seconds_per_step, b, features.device)
    def seconds_to_steps(seconds):
        ratio = seconds / dt
        # Keep an exact physical boundary (e.g. 2s / (4/30s) == 15)
        # from losing one feature step through float32 round-off.
        tolerance = 4 * torch.finfo(ratio.dtype).eps * ratio.abs().clamp_min(1)
        return (ratio + tolerance).floor().long().clamp(max=t)
    radius = seconds_to_steps(context_seconds)
    exclusion = seconds_to_steps(exclude_seconds)
    idx = torch.arange(t, device=features.device)[None, :].expand(b, -1)
    lo = (idx - radius[:, None]).clamp(min=0)
    left_end = (idx - exclusion[:, None]).clamp(min=0)
    right_begin = (idx + exclusion[:, None] + 1).clamp(max=t)
    hi = (idx + radius[:, None] + 1).clamp(max=t)
    # If rounding removes the entire context, produce an empty interval.
    left_end = torch.maximum(lo, left_end)
    right_begin = torch.minimum(hi, right_begin)
    p = probabilities.detach().float()
    trusted = (p < threshold) & valid[:, None, :]
    weights = torch.where(trusted, 1.0 - p, torch.zeros_like(p))
    clean = torch.where(valid[..., None], features, torch.zeros_like(features)).float()

    def window_sum(values: Tensor) -> Tensor:
        prefix = torch.cat((torch.zeros_like(values[:, :, :1]), values.cumsum(2)), 2)
        def at(index):
            shape = (b, values.shape[1], t) + values.shape[3:]
            gather_index = index[:, None]
            for _ in values.shape[3:]:
                gather_index = gather_index.unsqueeze(-1)
            return prefix.gather(2, gather_index.expand(shape))
        return at(left_end) - at(lo) + at(hi) - at(right_begin)

    numerator = window_sum(weights[..., None] * clean[:, None])
    denominator = window_sum(weights)
    count = window_sum(trusted.float())
    available = (count >= min_steps) & valid[:, None, :] & (denominator > 0)
    reference = numerator / denominator.clamp_min(1e-8)[..., None]
    reference = torch.where(available[..., None], reference, torch.zeros_like(reference))
    return reference.to(features.dtype), available


class MaskedTemporalBlock(nn.Module):
    """Stride-one local encoder; channel-only normalization avoids padding bias."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.output = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        x = torch.where(valid[..., None], x, torch.zeros_like(x))
        x = F.gelu(self.norm1(self.input(x)))
        x = x * valid[..., None]
        delta = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.output(F.gelu(delta)))
        return x * valid[..., None]


class KinematicTransientEncoder(nn.Module):
    """Appearance derivatives plus relative-box derivatives; not optical flow."""
    def __init__(self, roi_dim: int, geometry_dim: int, hidden: int):
        super().__init__()
        self.subject = nn.Linear(roi_dim, hidden)
        self.object = nn.Linear(roi_dim, hidden)
        self.relative = nn.Linear(geometry_dim, hidden)
        self.encoder = MaskedTemporalBlock(hidden * 3, hidden)

    def forward(self, subject, object_, geometry, valid, dt):
        motion = torch.cat((
            self.subject(masked_difference(subject, valid, dt)),
            self.object(masked_difference(object_, valid, dt)),
            self.relative(masked_difference(geometry, valid, dt)),
        ), -1)
        return self.encoder(motion, valid)


class RelationEnhancement(nn.Module):
    """Complete three-module head on top of pre-downsampling BSA features.

    It retains the appearance stream. Predicate-conditioned residuals are
    aggregated into high-resolution memory, while dense relation and boundary
    logits remain predicate-specific. Native integration reads this memory and
    uses the dense logits as additive (not multiplicative) corrections.
    """
    def __init__(self, config: EnhancementConfig, text_embeddings: Tensor):
        super().__init__()
        self.config = config
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != config.text_dim:
            raise ValueError("text_embeddings must be [P,text_dim]")
        if not text_embeddings.is_floating_point() or not torch.isfinite(text_embeddings).all():
            raise ValueError("Text embeddings must be finite floating point vectors")
        if text_embeddings.shape[0] < 1 or (text_embeddings.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Each foreground predicate needs a nonzero text vector")
        self.num_predicates = text_embeddings.shape[0]
        self.register_buffer("text_embeddings", F.normalize(text_embeddings.detach().float(), dim=-1))
        h = config.hidden_dim
        self.appearance = MaskedTemporalBlock(config.pair_dim, h)
        self.motion = KinematicTransientEncoder(config.roi_dim, config.geometry_dim, h)
        self.text_projection = nn.Sequential(nn.Linear(config.text_dim, h), nn.LayerNorm(h))
        self.predicate_ids = nn.Parameter(torch.randn(self.num_predicates, h) * 0.02)
        self.coarse_head = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, 1))
        self.motion_gate_head = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, 1))
        self.context_gate_head = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, 1))
        self.context_projection = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.semantic_head = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, 3))
        self.log_similarity_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        # New residual strengths start at zero. Dense auxiliary heads still train.
        self.motion_strength = nn.Parameter(torch.zeros(()))
        self.context_strength = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _condition(visual, text):
        return torch.cat((visual, text, visual * text), dim=-1)

    def forward(
        self, pair_features: Tensor, subject_features: Tensor,
        object_features: Tensor, relative_geometry: Tensor, valid_mask: Tensor,
        seconds_per_step: Union[float, Tensor], context_enabled: bool = False,
        text_embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        cfg = self.config
        if pair_features.ndim != 3 or pair_features.shape[1] < 1:
            raise ValueError("pair_features must be [B,T,C], T >= 1")
        b, t, _ = pair_features.shape
        expected = ((pair_features, cfg.pair_dim), (subject_features, cfg.roi_dim),
                    (object_features, cfg.roi_dim), (relative_geometry, cfg.geometry_dim))
        if valid_mask.shape != (b, t) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,T]")
        for value, channels in expected:
            if value.shape != (b, t, channels) or value.device != pair_features.device:
                raise ValueError("Feature shape/device does not match configuration")
            if not value.is_floating_point() or not torch.isfinite(value[valid_mask]).all():
                raise ValueError("Valid feature positions must be finite floating point values")
        dt = time_steps(seconds_per_step, b, pair_features.device)
        appearance = self.appearance(pair_features, valid_mask)
        motion = (self.motion(subject_features, object_features, relative_geometry, valid_mask, dt)
                  if cfg.use_kte else torch.zeros_like(appearance))
        text = self.text_embeddings if text_embeddings is None else text_embeddings.detach()
        if text.shape not in ((self.num_predicates, cfg.text_dim),
                              (b, self.num_predicates, cfg.text_dim)):
            raise ValueError("Text override must be [P,E] or [B,P,E], ordered by predicate")
        if not torch.isfinite(text).all() or (text.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Invalid text override")
        if text.device != pair_features.device:
            raise ValueError("Text embeddings and features must be on the same device")
        if cfg.use_language:
            text = self.text_projection(text.to(appearance.dtype))
        else:
            # Explicit learned-ID control, not an assertion of cross-modal alignment.
            text = self.predicate_ids
        if text.ndim == 2:
            text = text[None].expand(b, -1, -1)

        result: Dict[str, List[Tensor]] = {key: [] for key in (
            "coarse_logits", "relation_logits", "start_logits", "end_logits",
            "motion_gate_logits", "context_available")}
        residual_sum = torch.zeros_like(appearance)
        mass_sum = appearance.new_zeros(b, t, 1)
        for p0 in range(0, self.num_predicates, cfg.predicate_chunk_size):
            p1 = min(self.num_predicates, p0 + cfg.predicate_chunk_size)
            original = appearance[:, None].expand(-1, p1 - p0, -1, -1)
            anchor = text[:, p0:p1, None].expand(-1, -1, t, -1)
            coarse = self.coarse_head(self._condition(original, anchor)).squeeze(-1)
            gate_logits = self.motion_gate_head(torch.cat((
                original, motion[:, None].expand_as(original), anchor), -1)).squeeze(-1)
            motion_residual = (cfg.residual_limit * self.motion_strength.tanh()
                               * gate_logits.sigmoid()[..., None] * motion[:, None])
            enhanced = original + motion_residual
            if cfg.use_cdse and context_enabled:
                reference, available = local_background_reference(
                    appearance, coarse.sigmoid(), valid_mask, dt,
                    cfg.context_seconds, cfg.exclude_seconds,
                    cfg.background_threshold, cfg.min_background_steps)
                difference = original - reference
                context_gate = self.context_gate_head(torch.cat((
                    original, difference, anchor), -1)).sigmoid()
                context_residual = (cfg.residual_limit * self.context_strength.tanh()
                                    * context_gate * self.context_projection(difference)
                                    * available[..., None])
                enhanced = enhanced + context_residual
            else:
                available = torch.zeros_like(coarse, dtype=torch.bool)
            local_logits = self.semantic_head(self._condition(enhanced, anchor))
            similarity = (F.normalize(enhanced, dim=-1)
                          * F.normalize(anchor, dim=-1)).sum(-1)
            scale = self.log_similarity_scale.clamp(max=math.log(50)).exp()
            relation = local_logits[..., 0] + scale * similarity
            # A soft selector averages residuals, never deletes the original path.
            mass = coarse.detach().sigmoid() * valid_mask[:, None]
            residual_sum = residual_sum + ((enhanced - original) * mass[..., None]).sum(1)
            mass_sum = mass_sum + mass.sum(1)[..., None]
            values = (coarse, relation, local_logits[..., 1], local_logits[..., 2],
                      gate_logits, available)
            for key, value in zip(result, values):
                result[key].append(value * valid_mask[:, None])
        outputs = {key: torch.cat(values, 1) for key, values in result.items()}
        outputs["high_res_features"] = (appearance + residual_sum / mass_sum.clamp_min(1e-8)) * valid_mask[..., None]
        outputs["motion_features"] = motion
        outputs["valid_mask"] = valid_mask
        return outputs


def build_dense_targets(
    class_ids: Sequence[Tensor], instance_masks: Sequence[Tensor], valid_mask: Tensor,
    num_predicates: int, known_masks: Optional[Sequence[Tensor]] = None,
    assume_exhaustive_annotations: bool = False, censor_window_edges: bool = True,
) -> Dict[str, Tensor]:
    """Training-only conversion from native VrdONE ONE-based foreground IDs.

    Repeated instances of a predicate are unioned for frame supervision. Their
    individual boundaries are retained. Unknown labels are never negatives.
    known_masks: one bool [P,valid_length] per sample, or explicit complete-label
    assumption. Cropped/unknown boundaries are excluded, not fabricated.
    """
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool or num_predicates < 1:
        raise ValueError("Invalid valid_mask or num_predicates")
    b, t = valid_mask.shape
    if len(class_ids) != b or len(instance_masks) != b:
        raise ValueError("GT batch size mismatch")
    if known_masks is None and not assume_exhaustive_annotations:
        raise ValueError("Provide known_masks or explicitly verify exhaustive annotations")
    if known_masks is not None and len(known_masks) != b:
        raise ValueError("known_masks batch size mismatch")
    device = valid_mask.device
    frame = torch.zeros((b, num_predicates, t), device=device)
    start, end = torch.zeros_like(frame), torch.zeros_like(frame)
    known = torch.zeros_like(frame, dtype=torch.bool)
    for i in range(b):
        ids = torch.as_tensor(class_ids[i], device=device)
        masks = torch.as_tensor(instance_masks[i], device=device)
        if ids.ndim != 1 or masks.ndim != 2 or masks.shape[0] != ids.numel():
            raise ValueError("Expected class IDs [N] and instance masks [N,T_i]")
        if ids.is_floating_point() or ids.dtype == torch.bool or ((ids < 1) | (ids > num_predicates)).any():
            raise ValueError("GT foreground class IDs must be integers in 1..P")
        width = masks.shape[1]
        if width > t or valid_mask[i, width:].any():
            raise ValueError("GT temporal grid does not cover valid feature positions")
        if not torch.isfinite(masks).all() or not ((masks == 0) | (masks == 1)).all():
            raise ValueError("instance_masks must be binary on the SAME sampled grid")
        if known_masks is None:
            known[i] = valid_mask[i, None]
        else:
            item_known = torch.as_tensor(known_masks[i], device=device)
            if item_known.dtype != torch.bool or item_known.shape != (num_predicates, width):
                raise ValueError("Each known mask must be bool [P,T_i]")
            known[i, :, :width] = item_known & valid_mask[i, None, :width]
        for label, instance in zip(ids.tolist(), masks.bool()):
            p = label - 1
            instance = F.pad(instance, (0, t - width)) & valid_mask[i]
            frame[i, p] = torch.maximum(frame[i, p], instance.float())
            known[i, p] |= instance
            previous = F.pad(instance[:-1], (1, 0), value=False)
            following = F.pad(instance[1:], (0, 1), value=False)
            start[i, p] = torch.maximum(start[i, p], (instance & ~previous).float())
            end[i, p] = torch.maximum(end[i, p], (instance & ~following).float())
    start_known, end_known = known.clone(), known.clone()
    # Boundaries beside an unobserved position cannot be inferred from a crop.
    if t > 1:
        start_known[:, :, 1:] &= known[:, :, :-1]
        end_known[:, :, :-1] &= known[:, :, 1:]
    if censor_window_edges:
        start_known[:, :, 0] &= frame[:, :, 0] == 0
        # Applies to each sample's actual last valid position, not padded T-1.
        for i in range(b):
            indices = valid_mask[i].nonzero().flatten()
            if indices.numel():
                last = int(indices[-1])
                end_known[i, :, last] &= frame[i, :, last] == 0
    for value in (frame, start, end):
        value.mul_(valid_mask[:, None])
    return {"frame": frame, "start": start, "end": end,
            "known": known, "start_known": start_known, "end_known": end_known}


def balanced_binary_loss(logits: Tensor, targets: Tensor, known: Tensor) -> Tensor:
    """Equal positive/negative mass per sample/predicate where both exist."""
    if logits.shape != targets.shape or known.shape != targets.shape:
        raise ValueError("Loss shape mismatch")
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    positive = known & targets.bool()
    negative = known & ~targets.bool()
    np = positive.sum(-1)
    nn_ = negative.sum(-1)
    lp = (losses * positive).sum(-1) / np.clamp_min(1)
    ln = (losses * negative).sum(-1) / nn_.clamp_min(1)
    terms = (np > 0).float() + (nn_ > 0).float()
    per = (lp + ln) / terms.clamp_min(1)
    active = terms > 0
    return (per * active).sum() / active.sum().clamp_min(1)


def temporal_ranking_loss(logits, targets, known, margin=0.5):
    """Same-pair/predicate positives vs known negatives; log-MEAN-exp negatives.

    All repeated positive intervals are excluded from negatives. Normalizing by
    negative count avoids a built-in preference for shorter coexistence windows.
    """
    x = logits.float()
    pos = known & targets.bool()
    neg = known & ~targets.bool()
    np, nn_ = pos.sum(-1), neg.sum(-1)
    positive_mean = (x * pos).sum(-1) / np.clamp_min(1)
    negative_lme = torch.logsumexp(x.masked_fill(~neg, -1e4), -1) - nn_.clamp_min(1).float().log()
    active = (np > 0) & (nn_ > 0)
    loss = F.relu(margin + negative_lme - positive_mean)
    return (loss * active).sum() / active.sum().clamp_min(1)


@dataclass
class LossWeights:
    coarse: float = 0.5
    relation: float = 1.0
    start: float = 0.25
    end: float = 0.25
    motion_gate: float = 0.1
    temporal_rank: float = 0.1
    consistency: float = 0.1


class EnhancementCriterion(nn.Module):
    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.weights = weights or LossWeights()
        if any(v < 0 or not math.isfinite(v) for v in asdict(self.weights).values()):
            raise ValueError("Loss weights must be finite and nonnegative")

    def forward(self, outputs, targets, use_motion_gate=True) -> Dict[str, Tensor]:
        known = targets["known"] & outputs["valid_mask"][:, None]
        frame = targets["frame"]
        w = self.weights
        return {
            "enh_coarse": w.coarse * balanced_binary_loss(outputs["coarse_logits"], frame, known),
            "enh_relation": w.relation * balanced_binary_loss(outputs["relation_logits"], frame, known),
            "enh_start": w.start * balanced_binary_loss(outputs["start_logits"], targets["start"], targets["start_known"] & known),
            "enh_end": w.end * balanced_binary_loss(outputs["end_logits"], targets["end"], targets["end_known"] & known),
            "enh_motion_gate": (w.motion_gate if use_motion_gate else 0.0) * balanced_binary_loss(outputs["motion_gate_logits"], frame, known),
            "enh_rank": w.temporal_rank * temporal_ranking_loss(outputs["relation_logits"], frame, known),
        }

    def consistency(self, clean, perturbed, known: Tensor) -> Tensor:
        """Optional aligned, label-preserving nuisance augmentation control.

        This is not automatically a camera-invariance guarantee. The caller must
        ensure that the augmentation preserves predicates, identities and times.
        """
        if clean["relation_logits"].shape != perturbed["relation_logits"].shape:
            raise ValueError("Consistency requires identical temporal/predicate grids")
        valid = known & clean["valid_mask"][:, None] & perturbed["valid_mask"][:, None]
        error = (clean["relation_logits"].float().detach().sigmoid()
                 - perturbed["relation_logits"].float().sigmoid()).square()
        return self.weights.consistency * (error * valid).sum() / valid.sum().clamp_min(1)


def soft_refine_predictions(
    class_logits: Tensor, mask_logits: Tensor, dense: Dict[str, Tensor],
    class_scale: Tensor, mask_scale: Tensor, boundary_scale: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Bounded additive corrections; keep Q, P, ranking budget and decoder rules.

    Boundary evidence scores a predicted interval; it never hard-trims it. There
    is no guarantee that corrections improve detection or preserve calibration.
    """
    relation = dense["relation_logits"].to(mask_logits.dtype)
    valid = dense["valid_mask"]
    if class_logits.shape[-1] != relation.shape[1] + 1:
        raise ValueError("Native class dimension must be P+1 with background at 0")
    if mask_logits.shape[:2] != class_logits.shape[:2] or mask_logits.shape[-1] != relation.shape[-1]:
        raise ValueError("Prediction dimensions do not match dense evidence")
    class_weights = class_logits.detach().float().softmax(-1)[..., 1:].to(relation.dtype)
    evidence = torch.einsum("bqp,bpt->bqt", class_weights, relation.tanh())
    refined_mask = mask_logits + mask_scale.tanh() * evidence * valid[:, None]
    temporal_weights = mask_logits.detach().float().sigmoid() * valid[:, None]
    temporal_weights = temporal_weights / temporal_weights.sum(-1, keepdim=True).clamp_min(1e-8)
    local_semantics = torch.einsum("bqt,bpt->bqp", temporal_weights.to(relation.dtype), relation.tanh())
    selected = (refined_mask.detach().sigmoid() > 0.5) & valid[:, None]
    exists = selected.any(-1)
    first = selected.long().argmax(-1)
    last = selected.shape[-1] - 1 - selected.flip(-1).long().argmax(-1)
    start = dense["start_logits"].transpose(1, 2)
    end = dense["end_logits"].transpose(1, 2)
    begin_score = start.gather(1, first[..., None].expand(-1, -1, start.shape[-1]))
    end_score = end.gather(1, last[..., None].expand(-1, -1, end.shape[-1]))
    boundary = (begin_score.tanh() + end_score.tanh()) * 0.5 * exists[..., None]
    correction = class_scale.tanh() * local_semantics + boundary_scale.tanh() * boundary
    refined_class = class_logits + F.pad(correction.to(class_logits.dtype), (1, 0))
    return refined_class, refined_mask
