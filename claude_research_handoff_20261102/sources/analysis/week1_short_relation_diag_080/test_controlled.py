"""Synthetic statistical unit tests; no project experiment results."""
import json
import unittest

import numpy as np

from controlled import (_design, _outcome, _weighted_estimate, compare,
                        judgment, video_bootstrap)


def row(index, *, short=True, coexist=15, competitor=False, video=None,
        predicate="watch", tiou=0, **kwargs):
    result = dict(video_name=video or f"video{index}", gt_index=index,
                  relation_sec=1.1 if short else 5., coexist_sec=coexist,
                  predicate=predicate, video_GT_load=100, pred_available=True,
                  tiou=tiou, duration_ratio=2., pred_interval_disjoint=False,
                  native_F=True, chosen_pair_covers_gt=True,
                  pair_predicate_annotation_count=1,
                  competition_flags={"overlap_other_pred_ge3": competitor},
                  source_scope="synthetic", readout="fixed")
    result.update(kwargs)
    return result


class ControlledTests(unittest.TestCase):
    def test_h1_never_controls_away_duration(self):
        rows = [row(0, short=True), row(1, short=False, tiou=1)]
        result = compare(rows, "H1", repetitions=20)
        self.assertEqual(result["N_common_support"], 2)
        self.assertEqual(result["standardized_gap"], 1)
        self.assertEqual(result["judgment"]["prediction_status"], "未决")
        self.assertEqual(result["judgment"]["mechanism_status"], "未决")

    def test_h2_fine_duration_controls_and_strict_control(self):
        rows = [row(0), row(1, coexist=5, competitor=True)]
        original = compare(rows, "H2", repetitions=20)
        strict = compare(rows, "H2", h2_no_comp_control=True, repetitions=20)
        self.assertEqual(original["N_common_support"], 2)
        self.assertEqual(strict["N_common_support"], 0)
        unmatched = compare([row(0), row(1, coexist=5, relation_sec=1.4)], "H2", repetitions=20)
        self.assertEqual(unmatched["N_common_support"], 0)

    def test_h5_competition_and_load_sensitivity(self):
        rows = [row(0, competitor=True, video_GT_load=20), row(1, video_GT_load=100)]
        self.assertEqual(compare(rows, "H5", repetitions=20)["N_common_support"], 2)
        self.assertEqual(compare(rows, "H5", load_control=True, repetitions=20)["N_common_support"], 0)
        rows[1]["coexist_sec"] = 25
        self.assertEqual(compare(rows, "H5", repetitions=20)["N_common_support"], 0)

    def test_json_flag_roundtrip(self):
        item = row(0, competitor=True)
        item["competition_flags"] = json.dumps(item["competition_flags"])
        side, fields, status = _design(item, "H5", definition="overlap_other_pred_ge3",
                                       h2_no_comp_control=False, load_control=False)
        self.assertEqual((side, status), (1, "eligible"))
        self.assertEqual(fields, ("watch", 4, 2))

    def test_missing_group_and_missing_metric(self):
        missing_group = compare([row(0)], "H5", repetitions=20)
        self.assertEqual(missing_group["N_common_support"], 0)
        self.assertIsNone(missing_group["standardized_gap"])
        missing = compare([row(0, competitor=True, tiou=None), row(1)], "H5", repetitions=20)
        self.assertEqual(missing["N_missing_outcome"], 1)
        self.assertEqual(missing["judgment"]["reason"], "missing_outcomes_not_counted_as_failures")

    def test_numeric_tiou_with_unknown_availability_stays_unavailable(self):
        item = row(0, pred_available=None, tiou=.9)
        self.assertEqual(_outcome(item, "miss_tiou05"), (None, "missing_outcome"))
        self.assertEqual(_outcome(item, "tiou"), (None, "missing_outcome"))

    def test_official_miss_is_spatial_tube_detection_and_requires_provenance(self):
        rows = [row(0, short=True, VrdONE_hit=0, BIG_hit="1", readout="official_Top100_vIoU50"),
                row(1, short=False, VrdONE_hit="1", BIG_hit=0, readout="official_Top100_vIoU50")]
        vrd = compare(rows, "H1", outcome="VrdONE_official_miss", repetitions=20)
        big = compare(rows, "H1", outcome="BIG_official_miss", repetitions=20)
        self.assertEqual(vrd["standardized_gap"], 1)
        self.assertEqual(big["standardized_gap"], -1)
        self.assertIn("NOT temporal-only tIoU", vrd["metric_protocol"])
        rows[0]["source_scope"] = rows[1]["source_scope"] = "UNSPECIFIED"
        with self.assertRaises(ValueError):
            compare(rows, "H1", outcome="VrdONE_official_miss", repetitions=20)

    def test_official_hit_values_validate_and_do_not_require_temporal_prediction(self):
        item = row(0, VrdONE_hit=0, pred_available=None, tiou=None)
        self.assertEqual(_outcome(item, "VrdONE_official_miss"), (1., "observed"))
        item["VrdONE_hit"] = 2
        with self.assertRaises(ValueError):
            _outcome(item, "VrdONE_official_miss")
        item["VrdONE_hit"] = ""
        self.assertEqual(_outcome(item, "VrdONE_official_miss"), (None, "missing_outcome"))
        with self.assertRaises(ValueError):
            compare([row(0, VrdONE_hit=0)], "H1", outcome="VrdONE_official_miss", repetitions=20)

    def test_pooled_weights_correct_simpson_case(self):
        rows = [row(0, short=False, predicate="A"),
                row(1, short=True, predicate="A"), row(2, short=True, predicate="A"),
                row(3, short=False, predicate="B", tiou=1), row(4, short=False, predicate="B", tiou=1),
                row(5, short=True, predicate="B", tiou=1)]
        result = compare(rows, "H1", repetitions=20)
        self.assertAlmostEqual(result["raw_same_support_side1"] - result["raw_same_support_side0"], 1/3)
        self.assertEqual(result["standardized_gap"], 0)
        self.assertEqual(result["weighted_numerator_side0"], 3)
        self.assertEqual(result["weighted_denominator"], 6)

    def test_no_prediction_counts_as_miss_but_not_zero_length(self):
        item = row(0, pred_available=False, tiou=None, duration_ratio=None)
        self.assertEqual(_outcome(item, "miss_tiou05"), (1., "observed"))
        self.assertEqual(_outcome(item, "stretch"), (None, "no_prediction_conditional_exclusion"))

    def test_stretch_must_separate_disjoint_and_overlapping(self):
        item = row(0, duration_ratio=2, pred_interval_disjoint=True)
        self.assertEqual(_outcome(item, "stretch"), (1., "observed"))
        self.assertEqual(_outcome(item, "overlap_stretch"), (0., "observed"))
        self.assertEqual(_outcome(item, "no_overlap_stretch"), (1., "observed"))
        item["pred_interval_disjoint"] = False
        self.assertEqual(_outcome(item, "overlap_stretch"), (1., "observed"))

    def test_no_prediction_ratio_exclusion_is_reported(self):
        result = compare([row(0, competitor=True, pred_available=False, tiou=None, duration_ratio=None),
                          row(1)], "H5", outcome="stretch", repetitions=20)
        self.assertEqual(result["exclusions"]["no_prediction_conditional_exclusion"], 1)
        self.assertEqual(result["N_missing_outcome"], 0)

    def test_excludes_invalid_common_support_and_wrong_population(self):
        rows = [row(0, relation_sec=1, coexist=.5), row(1, native_F=False)]
        result = compare(rows, "H1", population="native_F", repetitions=20)
        self.assertEqual(result["N_population"], 1)
        self.assertEqual(result["exclusions"]["invalid_time_covariates"], 1)

    def test_duplicate_and_mixed_sources_rejected(self):
        with self.assertRaises(ValueError):
            compare([row(0), row(0)], "H1", repetitions=20)
        with self.assertRaises(ValueError):
            compare([row(0), row(1, readout="oracle")], "H1", repetitions=20)

    def test_exact_duration_boundaries(self):
        item = row(0, relation_sec=1.25)
        _, fields, _ = _design(item, "H2", definition="overlap_other_pred_ge3",
                               h2_no_comp_control=False, load_control=False)
        self.assertEqual(fields[-1], 5)
        item["relation_sec"] = 3
        side, _, _ = _design(item, "H1", definition="overlap_other_pred_ge3",
                              h2_no_comp_control=False, load_control=False)
        self.assertEqual(side, 0)


class BootstrapTests(unittest.TestCase):
    def test_repeated_video_draws_have_multiplicity_not_set_membership(self):
        # Video A has y=(0,1), video B has y=(1,0). Drawing A,A,B must
        # give means 1/3,2/3 rather than the unduplicated means .5,.5.
        cell = np.asarray([0, 1, 0, 1])
        values = np.asarray([0., 1., 1., 0.])
        weights = np.asarray([2., 2., 1., 1.])
        means, retained, _ = _weighted_estimate(cell, values, weights, 1, np.asarray([4]))
        self.assertAlmostEqual(means[0], 1/3)
        self.assertAlmostEqual(means[1], 2/3)
        self.assertEqual(retained, 1)
        replicated_cell = np.asarray([0, 1, 0, 1, 0, 1])
        replicated_y = np.asarray([0., 1., 0., 1., 1., 0.])
        means2, _, _ = _weighted_estimate(replicated_cell, replicated_y, np.ones(6), 1, np.asarray([6]))
        self.assertEqual(means, means2)

    def test_sparse_strata_disappearance_reported(self):
        data = [dict(video_name="a", side=0, outcome=0., stratum_id=0),
                dict(video_name="b", side=1, outcome=1., stratum_id=0)]
        main = video_bootstrap(data, 1, repetitions=100, seed=80)
        self.assertLess(main["repetitions_valid"], 100)
        self.assertEqual(main["retained_original_support_mass_p05"], 0)
        positive = video_bootstrap(data, 1, repetitions=100, seed=80, method="exp_multiplier")
        self.assertEqual(positive["repetitions_valid"], 100)
        self.assertEqual(positive["retained_original_support_mass_p05"], 1)
        self.assertEqual(main, video_bootstrap(data, 1, repetitions=100, seed=80))

    def test_no_support_and_one_video_no_ci(self):
        self.assertIsNone(video_bootstrap([], 0, repetitions=20)["ci_low"])
        data = [dict(video_name="a", side=0, outcome=0., stratum_id=0),
                dict(video_name="a", side=1, outcome=1., stratum_id=0)]
        self.assertIsNone(video_bootstrap(data, 1, repetitions=20)["ci_low"])

    def test_decision_requires_each_side_videos_and_stability(self):
        result = dict(N_missing_outcome=0, N_common_support=100, N_videos_side0=20,
                      N_videos_side1=20, outcome="miss_tiou05", standardized_gap=.05,
                      bootstrap=dict(ci_low=.01, ci_high=.09, repetitions_valid=2000,
                                     repetitions_requested=2000, retained_original_support_mass_p05=1))
        self.assertEqual(judgment(result)["prediction_status"], "暂成立")
        self.assertEqual(judgment(dict(result, N_videos_side0=19))["prediction_status"], "未决")
        result["bootstrap"]["retained_original_support_mass_p05"] = .79
        self.assertEqual(judgment(result)["prediction_status"], "未决")
        result["bootstrap"].update(retained_original_support_mass_p05=1, ci_low=-.02, ci_high=.02)
        result["standardized_gap"] = 0
        self.assertEqual(judgment(result)["prediction_status"], "证伪")
        self.assertEqual(judgment(result)["mechanism_status"], "未决")


if __name__ == "__main__":
    unittest.main()
