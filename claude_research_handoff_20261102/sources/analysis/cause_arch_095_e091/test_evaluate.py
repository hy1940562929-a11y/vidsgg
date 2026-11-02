"""Synthetic contract checks; not E091 experimental results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import ARMS, VideoStats, decide, evaluate, integer_change, position_null, tiou


class EvaluationContractTests(unittest.TestCase):
    def test_half_open_tiou(self):
        self.assertEqual(tiou(None, (10, 15)), 0)
        self.assertEqual(tiou((10, 15), (10, 15)), 1)
        self.assertEqual(tiou((0, 10), (10, 15)), 0)
        self.assertEqual(tiou((10, 20), (10, 15)), .5)

    def test_gains_and_losses_are_both_counted(self):
        self.assertEqual(integer_change([1, 0, 1, 0], [0, 1, 1, 0]),
                         dict(N=4, B_hits=2, reference_hits=2, gain=1, loss=1, net=0,
                              both_hit=1, neither_hit=1))

    def test_video_equal_is_not_event_equal(self):
        s = VideoStats(["v1", "v1", "v1", "v2"])
        result = s.difference([1, 1, 1, 0], [0, 0, 0, 0])
        self.assertEqual(result["point"], .5)
        nr = s.residual_net([1, 1, 1, 0])
        self.assertEqual(nr["point"], .75)
        self.assertEqual(nr["resampled_denominator_min"], 2)
        self.assertEqual(nr["resampled_denominator_max"], 6)
        self.assertEqual(s.receipt()["indices_sha256"], VideoStats(["v1", "v1", "v1", "v2"]).receipt()["indices_sha256"])

    def test_zero_reference_error_cannot_pass(self):
        result = VideoStats(["v1", "v2"]).ratio([0, 0], [0, 0])
        self.assertFalse(result["passes"])
        self.assertIsNone(result["point"])

    def test_null_is_deterministic_length_preserving_no_gt(self):
        draws = [position_null((12, 21), "v1", 2, 4, 0, 20, rep) for rep in range(100)]
        self.assertTrue(all(b-a == 9 and a % 4 == 0 and (b-1) % 4 == 0 for a, b in draws))
        self.assertTrue(all(a >= 0 and b <= 77 for a, b in draws))
        self.assertEqual(draws[12], position_null((12, 21), "v1", 2, 4, 0, 20, 12))
        self.assertIsNone(position_null(None, "v1", 2, 4, 0, 20, 12))

    def test_verdict_priority(self):
        gates = dict(technical=True, primary_ci_upper_below_5pp=False, long_clear_harm=False,
                     primary=True, seed_direction=True, protection=True, null=True,
                     endpoint_E=True, score_C=True)
        self.assertEqual(decide(gates), "VERIFIED_INTERVENTION_EFFECT")
        gates["score_C"] = False
        self.assertEqual(decide(gates), "METHOD_ONLY_NOT_VERIFIED_CAUSE")
        gates["protection"] = False
        self.assertEqual(decide(gates), "UNRESOLVED_STOP_WITHOUT_EXPANSION")
        gates["primary_ci_upper_below_5pp"] = True
        self.assertEqual(decide(gates), "NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE")
        gates["technical"] = False
        self.assertEqual(decide(gates), "TECHNICAL_FAILURE")

    def fixture(self):
        events = [dict(video_name=v, gt_index=i, pair_idx=i, query=0, origin=0, N_valid=10,
                       begin_fid=8, end_fid=13, main400=True, sub163=i % 2 == 0,
                       residual334=True, long_protection=True, admitted=True,
                       slot_id=f"{v}/{i}/0", fps=30, relation_sec=5/30)
                  for i, v in enumerate(("v1", "v1", "v2", "v2"))]
        predictions = {key: [dict(P=[8, 13], C=[8, 13]) for e in events] for key in ARMS}
        return dict(events=events, predictions=predictions)

    def test_end_to_end_null_and_no_gain_rejection(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            result = evaluate(data, d, expected_counts=False)
            self.assertEqual(result["verdict"], "NOT_SUPPORTED_STOP_THIS_CAUSE_RECIPE")
            self.assertEqual(result["primary_P"]["Z"]["point"], 0)
            self.assertEqual(result["NR334"]["A"]["denominator"], 4)
            self.assertTrue((Path(d)/"per_gt.csv").is_file())
            saved = json.loads((Path(d)/"null_intervals.json").read_text())
            self.assertEqual(len(saved["libraries"]["B90"][0]), 100)

    def test_admission_or_slot_reuse_change_fails(self):
        data = self.fixture()
        data["predictions"]["B90"][0]["P"] = None
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "admission"):
                evaluate(data, d, expected_counts=False)
        data = self.fixture()
        data["events"][1]["pair_idx"] = data["events"][0]["pair_idx"]
        data["predictions"]["B90"][1]["P"] = [12, 17]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "same slot"):
                evaluate(data, d, expected_counts=False)

    def test_descriptive_native_score_receipt(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            scores = {e["slot_id"]: np.array([-2, -2, 2, 2, -2, -2, -2, -2, -2, -2], np.float32)
                      for e in data["events"]}
            np.savez(base/"scores.npz", **scores)
            data["score_files"] = {key: "scores.npz" for key in ARMS}
            result = evaluate(data, base/"out", expected_counts=False, score_base=base)
            self.assertEqual(result["descriptive_scores"]["status"], "DESCRIPTIVE_ONLY")
            desc = result["descriptive_scores"]["aggregates"]["Z"]["main400"]
            self.assertEqual(desc["positive_components"]["mean"], 1)
            self.assertEqual(desc["P_duration_ratio"]["mean"], 1)
            self.assertEqual(desc["cdf_w1_over_gt_duration"]["N_defined"], 4)


if __name__ == "__main__":
    unittest.main()
