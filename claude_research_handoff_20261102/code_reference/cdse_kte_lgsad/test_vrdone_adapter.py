"""Native API tests using an existing source checkout, random weights, CPU only.

Set VRDONE_ROOT to a checkout/snapshot containing models/maskvrd.py. No real
checkpoint, annotation, dataset, evaluation or remote machine is accessed.
"""
import copy
import importlib
import os
from pathlib import Path
import sys
import unittest

import torch

from relation_enhancement import EnhancementConfig


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = os.environ.get("VRDONE_ROOT")
        if not root or not (Path(root) / "models/maskvrd.py").is_file():
            raise unittest.SkipTest("Set VRDONE_ROOT to an existing native VrdONE source tree")
        sys.path.insert(0, root)
        cls.Native = importlib.import_module("models.maskvrd").MaskVRD
        cls.Enhanced = importlib.import_module("vrdone_adapter").EnhancedMaskVRD

    def setUp(self):
        torch.manual_seed(19)
        self.cfg = dict(visual_dim=8, clip_dim=4, bbox_entity_dim=8, bbox_so_dim=5,
            embd_dim=16, num_classes=3, backbone_arch=[1, 1, 2], scale_factor=2,
            fpn_start_level=0, max_seq_len=64, n_mha_win_size=9, use_abs_pe=False,
            use_rel_pe=False, use_local=False, max_so_pair=200, fuse_ks=1,
            fuse_head=2, fuse_qx_stride=1, fuse_kv_stride=1, fuse_path_drop=0.,
            n_head=2, embd_kernel_size=3, embd_with_ln=True, dropattn=0.,
            dropout=0., droppath=0., fpn_dim=16, fpn_with_ln=True,
            fpn_norm_first=True, loss_types=["labels", "masks"],
            predictor=dict(n_input=16, n_embd=16, n_head=2, n_hidden=32,
                num_queries=9, num_classes=3, attn_pdrop=0., proj_pdrop=0.,
                path_pdrop=0., cls_prior_prob=0.01, n_qx_stride=0, n_kv_stride=1,
                num_layers=2, deep_supervision=True, enforce_input_project=False),
            cost_coeff_dict=dict(cost_class=2., cost_mask=2., cost_dice=5.),
            loss_coeff_dict=dict(eos_coef=0.1, loss_class=2., loss_mask=2., loss_dice=5.),
            with_clip_feature=False)
        self.enh_cfg = EnhancementConfig(16, 8, 5, 10, hidden_dim=8,
                                          predicate_chunk_size=2, min_background_steps=1)
        self.anchors = torch.randn(3, 10)

    def pair(self):
        native = self.Native(self.cfg, "cpu")
        enhanced = self.Enhanced(self.cfg, "cpu", self.enh_cfg, self.anchors, 4 / 30,
                                 native_state_dict=copy.deepcopy(native.state_dict()),
                                 assume_exhaustive_annotations=True)
        return native, enhanced

    def input_data(self):
        channels = 2 * 8 + 5 + 2 * 8 + (8 if self.cfg["with_clip_feature"] else 0)
        masks0 = torch.zeros(2, 48)
        masks0[0, 8:12] = 1
        masks0[1, 30:35] = 1
        masks1 = torch.zeros(1, 32)
        masks1[0, 10:15] = 1
        return {"so_features_list": [torch.randn(channels, 48), torch.randn(channels, 32)],
                "preds_list": [torch.tensor([1, 1]), torch.tensor([3])],
                "masks_list": [masks0, masks1]}

    def test_initial_predictions_match_native_exactly(self):
        original, enhanced = self.pair()
        original.eval()
        enhanced.eval()
        x = torch.randn(2, 37, 64)
        valid = torch.arange(64)[None, None] < torch.tensor([48, 32])[:, None, None]
        with torch.no_grad():
            a = original._mask_vrd(x, valid)
            b = enhanced._mask_vrd(x, valid)
        self.assertTrue(torch.equal(a["pred_logits"], b["pred_logits"]))
        self.assertTrue(torch.equal(a["pred_masks"], b["pred_masks"]))
        self.assertEqual(b["pred_logits"].shape, (2, 9, 4))
        for aa, bb in zip(a["aux_outputs"], b["aux_outputs"]):
            self.assertTrue(torch.equal(aa["pred_masks"], bb["pred_masks"]))
        enhanced.enabled = False
        with torch.no_grad():
            c = enhanced._mask_vrd(x, valid)
        self.assertTrue(torch.equal(a["pred_logits"], c["pred_logits"]))

    def test_native_training_loss_and_backward(self):
        _, model = self.pair()
        model.train()
        losses = model(self.input_data())
        self.assertIn("enh_relation", losses)
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertGreater(model.enhancement.appearance.input.weight.grad.abs().sum(), 0)
        self.assertIsNotNone(model.predictor.mask_strength.grad)

    def test_native_training_with_clip_input_layout(self):
        self.cfg["with_clip_feature"] = True
        _, model = self.pair()
        model.train()
        losses = model(self.input_data())
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(losses["total_loss"]))

    def test_all_new_branches_connect_to_native_predictions(self):
        _, model = self.pair()
        model.eval()
        model.set_context_enabled(True)
        with torch.no_grad():
            for name in ("query_strength", "feature_strength", "class_strength", "mask_strength", "boundary_strength"):
                getattr(model.predictor, name).fill_(0.2)
            model.enhancement.motion_strength.fill_(0.2)
            model.enhancement.context_strength.fill_(0.2)
            model.enhancement.coarse_head[-1].weight.zero_()
            model.enhancement.coarse_head[-1].bias.fill_(-8)
        x = torch.randn(1, 37, 64)
        valid = torch.ones(1, 1, 64, dtype=torch.bool)
        out = model._mask_vrd(x, valid)
        self.assertTrue(out["enhancement"]["context_available"].any())
        (out["pred_logits"].square().mean() + out["pred_masks"].square().mean()).backward()
        for parameter in (model.enhancement.text_projection[0].weight,
                          model.enhancement.context_projection[0].weight,
                          model.enhancement.motion.subject.weight,
                          model.predictor.read.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum(), 0)

    def test_nuisance_augmentation_path(self):
        _, model = self.pair()
        data = self.input_data()
        data["so_features_aug_list"] = [f + torch.randn_like(f) * 0.01 for f in data["so_features_list"]]
        with self.assertRaises(ValueError):
            model(data)
        data["augmentation_preserves_relations"] = True
        losses = model(data)
        self.assertIn("enh_consistency", losses)
        self.assertTrue(torch.isfinite(losses["enh_consistency"]))

    def test_disabled_control_has_finite_native_training_loss(self):
        _, model = self.pair()
        model.enabled = False
        losses = model(self.input_data())
        self.assertFalse(any(key.startswith("enh_") for key in losses))
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(losses["total_loss"]))

    def test_native_inference_short_long_buckets_and_budget(self):
        _, model = self.pair()
        model.eval()
        model._config_eval(dict(topk=2, n_max_pair=5, feat_stride=4, pred_min_frames=5))
        model.set_context_enabled(True)
        counts = [48, 80]
        raw_counts = [(count - 1) * 4 + 1 for count in counts]
        boxes = [torch.tensor([0., 0., 10., 10.]).repeat(count, 1)
                 for count in (raw_counts[0], raw_counts[0], raw_counts[1], raw_counts[1])]
        data = dict(so_features_list=[torch.randn(37, count) for count in counts],
                    sids=torch.tensor([0, 2]), oids=torch.tensor([1, 3]),
                    bboxes_list=boxes, cat_ids=torch.tensor([1, 2, 1, 2]),
                    cat_scores=torch.ones(4) * 0.9,
                    traj_durations=torch.tensor([[0, count] for count in
                        (raw_counts[0], raw_counts[0], raw_counts[1], raw_counts[1])]),
                    so_offset=[torch.tensor(0), torch.tensor(0)])
        result = model(data)
        if result is not None:
            self.assertLessEqual(len(result["triplets"]), 5)
            self.assertEqual(len(result["triplets"]), len(result["pred_durations"]))
            self.assertTrue(all(end - begin >= 5 for begin, end in result["pred_durations"]))

    def test_enhanced_checkpoint_round_trip_and_context_flag(self):
        _, a = self.pair()
        _, b = self.pair()
        a.eval()
        b.eval()
        a.set_context_enabled(True)
        with torch.no_grad():
            a.predictor.mask_strength.fill_(0.2)
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertTrue(bool(b.context_enabled))
        x = torch.randn(1, 37, 64)
        valid = torch.ones(1, 1, 64, dtype=torch.bool)
        with torch.no_grad():
            aa, bb = a._mask_vrd(x, valid), b._mask_vrd(x, valid)
        self.assertTrue(torch.equal(aa["pred_logits"], bb["pred_logits"]))
        self.assertTrue(torch.equal(aa["pred_masks"], bb["pred_masks"]))

    def test_no_deep_supervision_training(self):
        self.cfg["predictor"]["deep_supervision"] = False
        _, model = self.pair()
        losses = model(self.input_data())
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertNotIn("loss_class_0", losses)


if __name__ == "__main__":
    unittest.main()
