"""Synthetic software regression tests, never research evidence."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from relation_enhancement import (
    EnhancementConfig, RelationEnhancement, EnhancementCriterion, build_dense_targets,
    masked_difference, local_background_reference, temporal_ranking_loss,
    soft_refine_predictions, load_text_anchors,
)


class EnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)
        self.cfg = EnhancementConfig(12, 8, 5, 10, hidden_dim=8,
                                     predicate_chunk_size=2, min_background_steps=1)
        self.text = torch.randn(3, 10)
        self.valid = torch.tensor([[True] * 12, [True] * 8 + [False] * 4])
        self.inputs = (torch.randn(2, 12, 12), torch.randn(2, 12, 8),
                       torch.randn(2, 12, 8), torch.randn(2, 12, 5), self.valid)

    def model(self):
        return RelationEnhancement(self.cfg, self.text)

    def forward_model(self, model=None, **kw):
        return (model or self.model())(*self.inputs, seconds_per_step=0.25, **kw)

    def targets(self):
        masks0 = torch.zeros(3, 12)
        masks0[0, 2:4] = 1
        masks0[1, 7:9] = 1
        masks0[2, :] = 1
        masks1 = torch.zeros(1, 8)
        masks1[0, 3:5] = 1
        return build_dense_targets([torch.tensor([1, 1, 2]), torch.tensor([3])],
                                   [masks0, masks1], self.valid, 3,
                                   assume_exhaustive_annotations=True)

    def test_forward_shapes_and_finite_backward(self):
        model = self.model()
        out = self.forward_model(model)
        losses = EnhancementCriterion()(out, self.targets())
        torch.stack(list(losses.values())).sum().backward()
        self.assertEqual(out["relation_logits"].shape, (2, 3, 12))
        self.assertEqual(out["high_res_features"].shape, (2, 12, 8))
        self.assertGreater(model.appearance.input.weight.grad.abs().sum(), 0)
        self.assertGreater(model.text_projection[0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(model.text_embeddings.requires_grad)

    def test_repeated_events_multilabel_and_individual_boundaries(self):
        gt = self.targets()
        self.assertEqual(gt["frame"][0, 0].sum(), 4)
        self.assertTrue(gt["frame"][0, 0, 7])
        self.assertTrue(gt["frame"][0, 1, 7])
        self.assertEqual(gt["start"][0, 0].nonzero().flatten().tolist(), [2, 7])
        self.assertEqual(gt["end"][0, 0].nonzero().flatten().tolist(), [3, 8])
        self.assertFalse(gt["start_known"][0, 1, 0])
        self.assertFalse(gt["end_known"][1, 2, 8:].any())

    def test_unknown_annotation_requires_explicit_contract(self):
        with self.assertRaisesRegex(ValueError, "known_masks"):
            build_dense_targets([torch.tensor([1])], [torch.ones(1, 12)], self.valid[:1], 3)
        known = torch.zeros(3, 12, dtype=torch.bool)
        gt = build_dense_targets([torch.tensor([1])], [torch.ones(1, 12)], self.valid[:1], 3, [known])
        self.assertTrue(gt["known"][0, 0].all())
        self.assertFalse(gt["known"][0, 1:].any())

    def test_wrong_class_ids_and_wrong_temporal_grid_rejected(self):
        for label in (torch.tensor([0]), torch.tensor([4]), torch.tensor([1.0])):
            with self.assertRaises(ValueError):
                build_dense_targets([label], [torch.ones(1, 12)], self.valid[:1], 3,
                                    assume_exhaustive_annotations=True)
        with self.assertRaises(ValueError):
            build_dense_targets([torch.tensor([1])], [torch.ones(1, 6)], self.valid[:1], 3,
                                assume_exhaustive_annotations=True)

    def test_context_keeps_exact_boundary_at_real_sampling_interval(self):
        features = torch.zeros(1, 20, 1)
        features[0, 15, 0] = 1
        ref, available = local_background_reference(
            features, torch.full((1, 1, 20), 0.01),
            torch.ones(1, 20, dtype=torch.bool), 4 / 30,
            context_seconds=2.0, exclude_seconds=0.0, min_steps=1)
        self.assertTrue(available[0, 0, 0])
        self.assertAlmostEqual(ref[0, 0, 0, 0].item(), 1 / 15, places=6)

    def test_context_matches_direct_window_and_detaches_selector(self):
        features = torch.arange(12.0).view(1, 6, 2).requires_grad_()
        p = torch.tensor([[[0.02, 0.03, 0.8, 0.04, 0.6, 0.05]]], requires_grad=True)
        valid = torch.tensor([[True, True, True, True, False, True]])
        ref, available = local_background_reference(features, p, valid, 0.5, 1.0, 0.0, 0.1, 1)
        for t in range(6):
            candidates = [j for j in range(max(0, t - 2), min(6, t + 3))
                          if j != t and valid[0, j] and p[0, 0, j] < 0.1]
            if candidates and valid[0, t]:
                weights = 1 - p.detach()[0, 0, candidates]
                expected = (features[0, candidates] * weights[:, None]).sum(0) / weights.sum()
                torch.testing.assert_close(ref[0, 0, t], expected)
                self.assertTrue(available[0, 0, t])
            else:
                self.assertFalse(available[0, 0, t])
                self.assertEqual(ref[0, 0, t].abs().sum(), 0)
        ref.sum().backward()
        self.assertIsNone(p.grad)
        self.assertGreater(features.grad.abs().sum(), 0)

    def test_context_no_background_exact_fallback(self):
        model = self.model()
        with torch.no_grad():
            model.coarse_head[-1].weight.zero_()
            model.coarse_head[-1].bias.fill_(8)
            model.context_strength.fill_(1)
        on = self.forward_model(model, context_enabled=True)
        off = self.forward_model(model, context_enabled=False)
        self.assertFalse(on["context_available"].any())
        torch.testing.assert_close(on["high_res_features"], off["high_res_features"], rtol=0, atol=0)
        torch.testing.assert_close(on["relation_logits"], off["relation_logits"], rtol=0, atol=0)

    def test_context_branch_has_gradient_when_available(self):
        model = self.model()
        with torch.no_grad():
            model.coarse_head[-1].weight.zero_()
            model.coarse_head[-1].bias.fill_(-8)
            model.context_strength.fill_(0.3)
        out = self.forward_model(model, context_enabled=True)
        self.assertTrue(out["context_available"].any())
        out["relation_logits"].sum().backward()
        self.assertGreater(model.context_projection[0].weight.grad.abs().sum(), 0)

    def test_predicate_specific_background_not_global_relationness(self):
        f = torch.randn(1, 6, 2)
        p = torch.stack((torch.ones(6) * 0.9, torch.ones(6) * 0.01))[None]
        _, available = local_background_reference(f, p, torch.ones(1, 6, dtype=torch.bool), 0.25,
                                                  min_steps=1)
        self.assertFalse(available[:, 0].any())
        self.assertTrue(available[:, 1].all())

    def test_motion_does_not_cross_missing_positions(self):
        x = torch.tensor([[[1.], [3.], [1000.], [7.], [9.]]])
        valid = torch.tensor([[True, True, False, True, True]])
        delta = masked_difference(x, valid, torch.tensor([0.5]))
        torch.testing.assert_close(delta.flatten(), torch.tensor([0., 4., 0., 0., 4.]))

    def test_padding_cannot_contaminate_valid_outputs(self):
        model = self.model()
        original = self.forward_model(model)
        changed = [x.clone() for x in self.inputs[:-1]]
        for x in changed:
            x[~self.valid] = float("nan")
        other = model(*changed, self.valid, seconds_per_step=0.25)
        for key in ("relation_logits", "start_logits", "high_res_features"):
            torch.testing.assert_close(original[key], other[key])
        self.assertEqual(other["relation_logits"][1, :, 8:].abs().sum(), 0)

    def test_empty_support_and_single_step(self):
        model = self.model()
        for t in (1, 7):
            mask = torch.zeros(1, t, dtype=torch.bool)
            out = model(torch.randn(1, t, 12), torch.randn(1, t, 8),
                        torch.randn(1, t, 8), torch.randn(1, t, 5), mask, 0.1, True)
            self.assertTrue(torch.isfinite(out["relation_logits"]).all())
            self.assertEqual(out["high_res_features"].abs().sum(), 0)

    def test_chunking_is_numerically_equivalent(self):
        model = self.model()
        one = RelationEnhancement(replace(self.cfg, predicate_chunk_size=1), self.text)
        one.load_state_dict(model.state_dict())
        a = self.forward_model(model, context_enabled=True)
        b = self.forward_model(one, context_enabled=True)
        torch.testing.assert_close(a["relation_logits"], b["relation_logits"])

    def test_shuffled_text_changes_conditioning_but_not_target_order(self):
        model = self.model()
        a = self.forward_model(model)
        b = self.forward_model(model, text_embeddings=self.text.flip(0))
        self.assertGreater((a["relation_logits"] - b["relation_logits"]).abs().max(), 1e-5)
        self.assertEqual(a["relation_logits"].shape, b["relation_logits"].shape)

    def test_consistency_is_finite_and_asymmetric(self):
        model = self.model()
        clean = self.forward_model(model)
        noisy_inputs = list(self.inputs)
        noisy_inputs[0] = noisy_inputs[0] + 0.1 * torch.randn_like(noisy_inputs[0])
        noisy = model(*noisy_inputs, seconds_per_step=0.25)
        criterion = EnhancementCriterion()
        self.assertEqual(criterion.consistency(clean, clean, self.targets()["known"]).item(), 0)
        loss = criterion.consistency(clean, noisy, self.targets()["known"])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss, 0)

    def test_log_mean_exp_does_not_reward_more_identical_negatives(self):
        a = temporal_ranking_loss(torch.tensor([[[0., 2., 2.]]]),
                                  torch.tensor([[[1., 0., 0.]]]), torch.ones(1, 1, 3, dtype=torch.bool))
        b = temporal_ranking_loss(torch.tensor([[[0., 2., 2., 2., 2.]]]),
                                  torch.tensor([[[1., 0., 0., 0., 0.]]]), torch.ones(1, 1, 5, dtype=torch.bool))
        torch.testing.assert_close(a, b)

    def test_soft_refinement_zero_is_exact_identity_and_can_add_peak(self):
        dense = self.forward_model()
        cls = torch.randn(2, 9, 4)
        masks = torch.full((2, 9, 12), -0.01)
        z = torch.tensor(0.)
        a, b = soft_refine_predictions(cls, masks, dense, z, z, z)
        self.assertTrue(torch.equal(a, cls))
        self.assertTrue(torch.equal(b, masks))
        dense["relation_logits"] = torch.ones_like(dense["relation_logits"]) * 5
        _, b = soft_refine_predictions(cls, masks, dense, z, torch.tensor(1.), z)
        self.assertTrue((b[self.valid[:, None].expand_as(b)] > 0).any())

    def test_text_cache_rejects_synthetic_and_wrong_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anchors.pt"
            torch.save({"embeddings": self.text, "predicates": ["a", "b", "c"], "synthetic": True}, path)
            with self.assertRaises(ValueError):
                load_text_anchors(path, ["a", "b", "c"])
            with self.assertRaises(ValueError):
                load_text_anchors(path, ["b", "a", "c"], allow_synthetic=True)
            self.assertTrue(torch.equal(load_text_anchors(path, ["a", "b", "c"], True), self.text))

    def test_default_vrdone_feature_dimensions_and_50_predicates(self):
        cfg = EnhancementConfig(512, 1024, 5, 512, hidden_dim=128)
        model = RelationEnhancement(cfg, torch.randn(50, 512)).eval()
        with torch.no_grad():
            model.coarse_head[-1].weight.zero_()
            model.coarse_head[-1].bias.fill_(-8)
            out = model(torch.randn(1, 512, 512), torch.randn(1, 512, 1024),
                        torch.randn(1, 512, 1024), torch.randn(1, 512, 5),
                        torch.ones(1, 512, dtype=torch.bool), 4 / 30, True)
        self.assertEqual(out["relation_logits"].shape, (1, 50, 512))
        self.assertEqual(out["high_res_features"].shape, (1, 512, 128))
        self.assertTrue(out["context_available"].all())
        self.assertTrue(torch.isfinite(out["relation_logits"]).all())

    def test_invalid_sampling_metadata_rejected(self):
        model = self.model()
        for dt in (0., -1., float("nan"), float("inf"), torch.ones(3)):
            with self.assertRaises(ValueError):
                model(*self.inputs, seconds_per_step=dt)


if __name__ == "__main__":
    unittest.main()
