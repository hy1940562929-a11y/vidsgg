"""Optional native VrdONE integration.

Put this directory and your ORIGINAL VrdONE checkout on PYTHONPATH. This adapter
imports, but does not vendor or modify, models.maskvrd.MaskVRD. Training/inference
flow follows lucaspk512/vrdone. No GPU selection, dataset download or experiment
launcher is included. See README.md for the exact supported input layout.
"""
from typing import Optional
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.maskvrd import MaskVRD
from relation_enhancement import (
    EnhancementConfig, EnhancementCriterion, LossWeights, RelationEnhancement,
    build_dense_targets, soft_refine_predictions,
)


class HighResolutionPredictor(nn.Module):
    """Keep native Q/P outputs; add full-resolution query reads and soft scores."""
    def __init__(self, native, hidden_dim: int, num_heads: int):
        super().__init__()
        self.native = native
        channels = native.query_embed.embedding_dim
        if channels % num_heads:
            raise ValueError("Query dimension must be divisible by num_heads")
        self.memory_projection = nn.Linear(hidden_dim, channels)
        self.mask_projection = nn.Linear(hidden_dim, channels)
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.read = nn.MultiheadAttention(channels, num_heads, dropout=0.0, batch_first=True)
        self.query_strength = nn.Parameter(torch.zeros(()))
        self.feature_strength = nn.Parameter(torch.zeros(()))
        self.class_strength = nn.Parameter(torch.zeros(()))
        self.mask_strength = nn.Parameter(torch.zeros(()))
        self.boundary_strength = nn.Parameter(torch.zeros(()))

    def forward(self, x, mask_features, mask, output_mask, dense, non_attn_const=-10):
        native = self.native
        valid = dense["valid_mask"]
        if (~valid.any(-1)).any() or (~mask.flatten(1).any(-1)).any():
            raise ValueError("Native VrdONE requires a nonempty sequence for each pair")
        src = native.input_norm(x)
        if native.input_proj is not None:
            src = native.input_proj(src) * mask.to(src.dtype)
        hs, _ = native.transformer(src, mask, query_embed=native.query_embed.weight.permute(1, 0))
        nl, b, c, nq = hs.shape
        high = dense["high_res_features"]
        memory = self.memory_norm(self.memory_projection(high)) * valid[..., None]
        mask_delta = self.mask_projection(high).transpose(1, 2) * valid[:, None]
        if mask_delta.shape != mask_features.shape:
            raise ValueError("FPN feature channels/time grid do not match native predictor")
        fused_mask_features = mask_features + self.feature_strength.tanh() * mask_delta
        refined = []
        for layer in hs:
            query = layer.transpose(1, 2)
            update, _ = self.read(self.query_norm(query), memory, memory,
                                  key_padding_mask=~valid, need_weights=False)
            refined.append((query + self.query_strength.tanh() * update).transpose(1, 2))
        refined = torch.stack(refined)
        flat = refined.reshape(nl * b, c, nq)
        classes = native.class_embed(flat).view(nl, b, -1, nq).permute(0, 1, 3, 2)
        embedding = native.mask_embed(flat).view(nl, b, -1, nq).permute(0, 1, 3, 2)
        masks = torch.einsum("lbqc,bct->lbqt", embedding, fused_mask_features)
        outputs = []
        for layer in range(nl):
            cls, msk = soft_refine_predictions(
                classes[layer], masks[layer], dense,
                self.class_strength, self.mask_strength, self.boundary_strength)
            msk = msk.masked_fill(~output_mask, float(non_attn_const))
            outputs.append({"pred_logits": cls, "pred_masks": msk})
        result = dict(outputs[-1])
        if native.aux_loss:
            result["aux_outputs"] = outputs[:-1]
        result["output_mask"] = output_mask
        return result


class EnhancedMaskVRD(MaskVRD):
    """Drop-in subclass for the inspected native MaskVRD API.

    seconds_per_step is required and must match ACTUAL feature sampling. The
    native wrapper currently supports a uniform sampling interval for a dataset.
    The standalone core additionally supports per-sample intervals.

    All original weights can be strictly loaded BEFORE wrapping via
    native_state_dict. Enhanced checkpoints should be loaded normally AFTER
    construction. Do not load an original checkpoint with silent strict=False.
    """
    def __init__(
        self, config, device, enhancement_config: EnhancementConfig,
        text_embeddings: Tensor, seconds_per_step: float,
        native_state_dict: Optional[dict] = None,
        loss_weights: Optional[LossWeights] = None,
        assume_exhaustive_annotations: bool = False,
        censor_window_edges: bool = True,
        enabled: bool = True,
    ):
        super().__init__(config, device)
        if native_state_dict is not None:
            self.load_state_dict(native_state_dict, strict=True)
        if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
            raise ValueError("seconds_per_step must be positive and match the dataset")
        if enhancement_config.pair_dim != self.embd_dim:
            raise ValueError("pair_dim must match native BSA embedding channels")
        if enhancement_config.roi_dim != self.visual_dim:
            raise ValueError("roi_dim must match native per-entity visual_dim")
        if enhancement_config.geometry_dim != config["bbox_so_dim"]:
            raise ValueError("geometry_dim must match native relative bbox features")
        if len(text_embeddings) != config["num_classes"]:
            raise ValueError("Text anchors must cover every foreground predicate in native ID order")
        self.enabled = enabled
        self.seconds_per_step = float(seconds_per_step)
        self.assume_exhaustive_annotations = assume_exhaustive_annotations
        self.censor_window_edges = censor_window_edges
        self.enhancement = RelationEnhancement(enhancement_config, text_embeddings)
        self.enhancement_criterion = EnhancementCriterion(loss_weights)
        self.predictor = HighResolutionPredictor(self.predictor, enhancement_config.hidden_dim,
                                                 config["predictor"]["n_head"])
        self.register_buffer("context_enabled", torch.tensor(False))

    def set_context_enabled(self, enabled: bool):
        """Enable only after a training-only warm-up and held-out calibration."""
        self.context_enabled.fill_(bool(enabled))

    def loss_masks(self, pred_logits, pred_masks, gt_preds, gt_masks, gt_segs,
                   indices, num_masks, loss_mask):
        # The imported scripted native loss infers num_masks as Tensor, whereas
        # native.loss() calls .item(). Keep the same scalar normalizer/formula,
        # but satisfy its type. This repair also applies to enabled=False.
        normalizer = torch.as_tensor(num_masks, dtype=torch.float32, device=pred_masks.device)
        return super().loss_masks(pred_logits, pred_masks, gt_preds, gt_masks,
                                  gt_segs, indices, normalizer, loss_mask)

    def _mask_vrd(self, batched_inputs, batched_masks):
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn, _ = self.neck(feats, masks)
        if not self.enabled:
            return self.predictor.native(feats[-1], fpn, masks[-1], output_mask=masks[0])
        v = self.visual_dim
        # CLIP-augmented native layout: [s_visual,o_visual,s_clip,o_clip,rel_box,...].
        relative_start = 2 * v + (2 * self.clip_dim if self.with_clip_feature else 0)
        relative = batched_inputs[:, relative_start:relative_start + self.bbox_so_dim]
        dense = self.enhancement(
            feats[0].transpose(1, 2), batched_inputs[:, :v].transpose(1, 2),
            batched_inputs[:, v:2 * v].transpose(1, 2), relative.transpose(1, 2),
            masks[0][:, 0].bool(), self.seconds_per_step,
            context_enabled=bool(self.context_enabled.item()),
        )
        predictions = self.predictor(feats[-1], fpn, masks[-1], masks[0], dense)
        predictions["enhancement"] = dense
        return predictions

    def forward_training(self, input_data):
        features, masks = self.preprocessing(input_data["so_features_list"])
        gt_preds = [torch.as_tensor(item, device=features.device) for item in input_data["preds_list"]]
        original_masks = [torch.as_tensor(item, device=features.device) for item in input_data["masks_list"]]
        targets = None
        if self.enabled:
            targets = build_dense_targets(
                gt_preds, original_masks, masks[:, 0].bool(), self.enhancement.num_predicates,
                input_data.get("predicate_known_list"), self.assume_exhaustive_annotations,
                self.censor_window_edges)
        elif any(item.ndim != 2 or item.shape[-1] > features.shape[-1] for item in original_masks):
            raise ValueError("Native masks must use the sampled feature grid")
        # Native matcher concatenates masks and needs a common padded grid. This
        # only appends zeros; it never interpolates raw-frame annotations.
        gt_masks = [F.pad(item.float(), (0, features.shape[-1] - item.shape[-1]))
                    for item in original_masks]
        predictions = self._mask_vrd(features, masks)
        gt_segs = input_data.get("segs_list")
        if self.with_fuzzy and gt_segs is None:
            raise ValueError("Native fuzzy loss requires segs_list")
        indices, loss_mask = self.bipartite_match(
            predictions["pred_logits"], gt_preds, predictions["pred_masks"],
            gt_masks, gt_segs, _mask=predictions["output_mask"])
        # Keyword arguments deliberately follow the METHOD SIGNATURE. In the
        # inspected snapshot, parent.forward_training swaps gt_preds/pred_masks
        # positionally. Both the enhanced and disabled control use this same fix.
        losses = self.loss(
            indices=indices, pred_logits=predictions["pred_logits"],
            pred_masks=predictions["pred_masks"], gt_preds=gt_preds,
            gt_masks=gt_masks, gt_segs=gt_segs,
            _mask=predictions["output_mask"], loss_mask=loss_mask,
            aux_outputs=predictions.get("aux_outputs") if self.deep_supervision else None)
        if self.enabled:
            losses.update(self.enhancement_criterion(
                predictions["enhancement"], targets,
                use_motion_gate=self.enhancement.config.use_kte))
        if self.enabled and "so_features_aug_list" in input_data:
            if input_data.get("augmentation_preserves_relations") is not True:
                raise ValueError("Explicitly certify label/time-preserving augmentations")
            aug_features, aug_masks = self.preprocessing(input_data["so_features_aug_list"])
            if not torch.equal(masks, aug_masks):
                raise ValueError("Nuisance augmentation changed the temporal support")
            augmented = self._mask_vrd(aug_features, aug_masks)
            losses["enh_consistency"] = self.enhancement_criterion.consistency(
                predictions["enhancement"], augmented["enhancement"], targets["known"])
        losses["total_loss"] = torch.stack(list(losses.values())).sum()
        return losses
