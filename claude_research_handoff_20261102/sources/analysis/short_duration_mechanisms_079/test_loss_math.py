"""Synthetic tests only; no dataset results or benchmark claims."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_math import (advancement_gate, decoded_frame_interval, endpoint_metrics, interval_iou,
                       interval_objective, summarize_rows)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class IntervalMathTests(unittest.TestCase):
    def test_gate_rejects_losing_less_than_ce(self):
        import copy
        comparisons = {a: dict(added=2, lost=1, net=1) for a in ("ce", "native", "zero_update")}
        group = dict(arms=dict(ce_iou=dict(comparisons=comparisons)))
        final = {role: dict(groups={g: copy.deepcopy(group) for g in ("short", "long", "all")})
                 for role in ("train", "adapter_dev")}
        self.assertTrue(advancement_gate(final)["passed"])
        final["adapter_dev"]["groups"]["short"]["arms"]["ce_iou"]["comparisons"]["native"] = dict(added=0, lost=1, net=-1)
        self.assertFalse(advancement_gate(final)["passed"])

    def test_half_open_geometry(self):
        self.assertEqual(interval_iou([3, 5], [3, 5]), 1.)
        self.assertEqual(interval_iou([3, 5], [5, 8]), 0.)
        self.assertAlmostEqual(interval_iou([2, 8], [4, 10]), .5)
        self.assertEqual(interval_iou(None, [3, 5]), 0.)

    def test_uniform_shift_formula(self):
        for duration in (3, 10, 30):
            delta = 1
            self.assertAlmostEqual(interval_iou([delta, duration + delta], [0, duration]),
                                   (duration - delta) / (duration + delta))

    def test_native_frame_mapping(self):
        self.assertEqual(decoded_frame_interval(2, 5, 100, 4, 8), [108, 121])
        self.assertIsNone(decoded_frame_interval(2, 2, 100, 4, 8))
        self.assertIsNone(decoded_frame_interval(-1, -1, 100, 4, 8))
        self.assertIsNone(decoded_frame_interval(2, 8, 100, 4, 8))

    def test_endpoint_denominator(self):
        metrics = endpoint_metrics([2, 12], [0, 10], 10)
        self.assertAlmostEqual(metrics["endpoint_abs_seconds"], .2)
        self.assertAlmostEqual(metrics["endpoint_relative"], .2)
        self.assertIsNone(endpoint_metrics(None, [0, 10], 10)["endpoint_relative"])

    def test_summary_retains_excluded_events_and_losses(self):
        rows = []
        for k, duration in enumerate((.5, 2., 5., 11.)):
            row = dict(relation_sec=duration, loss_eligible=k != 0,
                       singleton_GT=k == 0, video_name="video", arms={})
            values = dict(native=.6 if k in (0, 2) else .2,
                          zero_update=.6 if k in (0, 2) else .2,
                          ce=.6 if k in (0, 2) else .2,
                          ce_iou=.6 if k in (1, 2) else .2)
            for arm, iou in values.items():
                row["arms"][arm] = dict(true_frame_tIoU=iou,
                    endpoint_relative=None if k == 3 else .1,
                    endpoint_abs_seconds=None if k == 3 else .2)
            rows.append(row)
        result = summarize_rows(rows)
        self.assertEqual(result["all"]["N_GT"], 4)
        self.assertEqual(result["all"]["eligible_loss_GT"], 3)
        self.assertEqual(result["all"]["singleton_GT"], 1)
        self.assertEqual(result["all"]["arms"]["ce_iou"]["denominator"], 4)
        self.assertEqual(result["all"]["arms"]["ce_iou"]["valid_endpoint_denominator"], 3)
        self.assertEqual(result["short"]["arms"]["ce_iou"]["comparisons"]["ce"],
                         dict(added=1, lost=1, net=0))
        self.assertEqual(result["one_to_three"]["arms"]["ce_iou"]["comparisons"]["ce"]["added"], 1)


@unittest.skipUnless(HAS_TORCH, "torch unavailable; run synthetic CPU gradient tests on server")
class RiskObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def logits(self, n=1, t=3):
        torch = self.torch
        z = torch.zeros((n, t, t), dtype=torch.float64, requires_grad=True)
        legal = torch.triu(torch.ones((t, t), dtype=torch.bool), diagonal=1)
        return z, z.masked_fill(~legal[None], -1e9)

    def test_expected_risk_uses_true_frame_coordinates(self):
        _, scores = self.logits()
        loss, parts = interval_objective(scores, [0], [1], [100], [4], [[100, 105]], 1.)
        # Three legal native spans: [100,105), [100,109), [104,109).
        expected = ((1-1) + (1-5/9) + (1-1/9)) / 3
        self.assertAlmostEqual(float(parts["risk"]), expected)
        self.assertAlmostEqual(float(parts["ce"]), math.log(3))
        self.assertAlmostEqual(float(loss), math.log(3) + expected)

    def test_ce_arm_is_exact_same_joint_cross_entropy(self):
        torch = self.torch
        _, scores = self.logits()
        loss, parts = interval_objective(scores, [0], [1], [0], [4], [[0, 5]], 0.)
        expected = torch.nn.functional.cross_entropy(scores.flatten(1), torch.tensor([1]))
        self.assertAlmostEqual(float(loss), float(expected))
        self.assertEqual(parts["eligible"], 1)

    def test_risk_gradient_prefers_better_intervals(self):
        z, scores = self.logits()
        _, parts = interval_objective(scores, [0], [1], [0], [4], [[0, 5]], 1.)
        parts["risk"].backward()
        self.assertLess(float(z.grad[0, 0, 1]), 0)
        self.assertGreater(float(z.grad[0, 1, 2]), 0)
        self.assertEqual(float(z.grad[0, 2, 0]), 0.)

    def test_singletons_excluded_same_way_from_both_losses(self):
        z, scores = self.logits(n=2)
        loss, parts = interval_objective(scores, [0, 1], [1, 1], [0, 0], [4, 4], [[0, 5], [4, 5]], 1.)
        self.assertEqual(parts["eligible"], 1)
        loss.backward()
        self.assertEqual(float(z.grad[1].abs().sum()), 0.)

    def test_empty_support_is_finite_differentiable_zero(self):
        z, scores = self.logits()
        loss, parts = interval_objective(scores, [1], [1], [0], [4], [[4, 5]], 1.)
        self.assertEqual(parts["eligible"], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(z.grad.abs().sum()), 0.)

    def test_perfect_distribution_has_zero_risk(self):
        torch = self.torch
        _, scores = self.logits()
        scores = scores.detach().clone()
        scores[0, 0, 1] = 1000
        _, parts = interval_objective(scores, [0], [1], [0], [4], [[0, 5]], 1.)
        self.assertEqual(float(parts["risk"]), 0.)

    def test_real_objective_changes_gradient_from_ce(self):
        torch = self.torch
        z1, s1 = self.logits()
        z2, s2 = self.logits()
        l1, _ = interval_objective(s1, [0], [1], [0], [4], [[0, 5]], 0.)
        l2, _ = interval_objective(s2, [0], [1], [0], [4], [[0, 5]], 1.)
        l1.backward()
        l2.backward()
        self.assertFalse(torch.equal(z1.grad, z2.grad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
