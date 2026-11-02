"""Synthetic semantics checks; never run model inference or production data."""
import unittest
import numpy as np
from analyze_history import (binary_support_iou, decoded_hull, response_delay,
                             bootstrap_video_mean, decide, tiou)


class HistoryMetricsTests(unittest.TestCase):
    def test_mask_support_is_four_frames_but_hull_last_is_plus_one(self):
        bits = np.array([False, True, False, True])
        times = np.arange(4) * 4
        self.assertAlmostEqual(binary_support_iou(bits, times, 14, [4, 14]), 6 / 10)
        self.assertEqual(decoded_hull(bits, 0), [4, 13])
        self.assertAlmostEqual(tiou([4, 13], [4, 14]), .9)
        self.assertIsNone(decoded_hull(np.array([False, True]), 0))

    def test_binary_union_counts_target_outside_pair(self):
        self.assertAlmostEqual(binary_support_iou([True, True], [0, 4], 8, [0, 12]), 8 / 12)

    def test_delay_keeps_censored_failure(self):
        times = np.array([0, 4, 8, 12, 16])
        never = response_delay(np.ones(5), times, 8, 16, 4, False)
        self.assertTrue(never["censored"])
        self.assertIsNone(never["delay_seconds"])
        reached = response_delay(np.array([1, 1, 1, 0, 0]), times, 8, 20, 4, False)
        self.assertEqual(reached["delay_seconds"], 1)

    def test_video_weights_not_case_weights(self):
        rows = [dict(video_name="one", off=1)] + [dict(video_name="two", off=0)] * 10
        result = bootstrap_video_mean(rows, "off")
        self.assertEqual(result["point"], .5)

    def test_gate_cannot_promote_small_or_descriptive_sample(self):
        effect = dict(N_cases=30, N_videos=10, point=.04, CI_97_5_percent=[.01, .07])
        baseline = dict(point=.01)
        self.assertEqual(decide(effect, baseline), "SUPPORT_ANNOTATION_CONDITIONED_SENSITIVITY_ONLY")
        self.assertEqual(decide(dict(effect, N_cases=29), baseline), "INCONCLUSIVE_INSUFFICIENT_CASES_OR_VIDEOS")
        self.assertEqual(decide(effect, dict(point=-.01)), "INCONCLUSIVE")
        self.assertEqual(decide(effect, baseline, True), "DESCRIPTIVE_SUBGROUP_NOT_A_PRIMARY_GATE")
        self.assertEqual(decide(dict(effect, point=.01, CI_97_5_percent=[-.01, .02]), baseline), "BELOW_PRESPECIFIED_EFFECT")


if __name__ == "__main__":
    unittest.main()
