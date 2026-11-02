"""Synthetic correctness tests; not research results."""
import csv
import tempfile
import unittest
from pathlib import Path

from stats import build_summary, cluster_bootstrap_mean, write_summary_csv


def row(index=0, **kwargs):
    result = dict(video_name="v1", gt_index=index, predicate="hold",
                  relation_sec=1, coexist_sec=20, slice="S-L-solo",
                  pred_available=True, tiou=0.5, duration_ratio=1,
                  source_scope="synthetic", readout="fixed_query")
    result.update(kwargs)
    return result


def one(rows):
    return build_summary(rows, include_empty_slices=False)[0]


class SummaryTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(build_summary([]), [])
        result = build_summary([row()])
        empty = next(r for r in result if r["slice"] == "S-S")
        self.assertEqual(empty["N"], 0)
        self.assertEqual(empty["status"], "EMPTY")
        self.assertIsNone(empty["miss_rate_tiou05"])

    def test_exact_thresholds_and_stretch_tie(self):
        result = one([row(0, tiou=.3, duration_ratio=2), row(1, tiou=.5), row(2, tiou=.7)])
        self.assertEqual(result["miss_count_tiou03"], 0)
        self.assertEqual(result["miss_count_tiou05"], 1)
        self.assertEqual(result["miss_count_tiou07"], 2)
        self.assertEqual(result["stretch_count"], 1)
        self.assertEqual(result["stretch_denominator"], 3)
        self.assertAlmostEqual(result["duration_bias"], 4 / 3)

    def test_explicit_missing_prediction_is_zero_tiou_not_zero_ratio(self):
        result = one([row(0, pred_available=False, tiou=None, duration_ratio=None),
                      row(1, tiou=1, duration_ratio=2)])
        self.assertEqual(result["N"], 2)
        self.assertEqual(result["N_pred"], 1)
        self.assertEqual(result["no_pred_count"], 1)
        self.assertEqual(result["mean_tiou"], .5)
        self.assertEqual(result["miss_rate_tiou05"], .5)
        self.assertEqual(result["stretch_rate"], 1)
        self.assertEqual(result["stretch_rate_all"], .5)
        self.assertEqual(result["duration_bias"], 2)

    def test_no_predictions(self):
        result = one([row(pred_available=False, tiou=None, duration_ratio=None)])
        self.assertEqual(result["miss_rate_tiou05"], 1)
        self.assertEqual(result["stretch_rate_all"], 0)
        self.assertIsNone(result["duration_bias"])
        self.assertIsNone(result["stretch_rate"])

    def test_missing_prediction_columns_not_failures(self):
        item = row()
        for key in ("pred_available", "tiou", "duration_ratio"):
            del item[key]
        result = one([item])
        self.assertEqual(result["prediction_availability_missing"], 1)
        self.assertIsNone(result["N_pred"])
        self.assertIsNone(result["miss_count_tiou05"])
        self.assertIsNone(result["mean_tiou"])
        self.assertIsNone(result["duration_bias"])

    def test_partial_time_or_ratio_columns_do_not_silently_shrink_denominator(self):
        result = one([row(0), row(1, tiou=None, duration_ratio=None)])
        self.assertEqual(result["N_pred"], 2)
        self.assertEqual(result["N_tiou_valid"], 1)
        self.assertEqual(result["N_ratio_missing"], 1)
        self.assertIsNone(result["mean_tiou"])
        self.assertIsNone(result["duration_bias"])
        self.assertIsNone(result["stretch_count"])

    def test_ratio_unavailable_does_not_invalidate_complete_tiou(self):
        result = one([row(duration_ratio=None)])
        self.assertEqual(result["miss_rate_tiou05"], 0)
        self.assertEqual(result["mean_tiou"], .5)
        self.assertIsNone(result["duration_bias"])

    def test_different_sources_never_pooled(self):
        result = build_summary([row(), row(source_scope="second"), row(readout="oracle")],
                               include_empty_slices=False)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r["N"] == 1 for r in result))

    def test_count_balance(self):
        rows = [row(i, slice=name, pred_available=False, tiou=None, duration_ratio=None)
                for i, name in enumerate(("S-S", "S-L-solo", "S-L-comp", "L-L", "L-S", "UNKNOWN"))]
        result = build_summary(rows, include_all=True)
        self.assertEqual(sum(r["N"] for r in result[:-1]), result[-1]["N"])
        self.assertEqual(result[-1]["N_pred"] + result[-1]["no_pred_count"], 6)

    def test_definitions_and_populations_not_pooled(self):
        result = build_summary([row(), row(slice_definition="strict"),
                                row(analysis_population="chosen_pair_covers_gt")],
                               include_empty_slices=False)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r["N"] == 1 for r in result))
        default = next(r for r in result if r["analysis_population"] == "all" and r["slice_definition"] == "literal")
        self.assertEqual(default["N"], 1)

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):
            build_summary([row(), row(slice="S-S")])

    def test_invalid_values_rejected(self):
        for kwargs in (dict(tiou=1.1), dict(tiou=float("nan")), dict(duration_ratio=-1),
                       dict(duration_ratio=0), dict(pred_available="nope"),
                       dict(pred_available=False), dict(slice="bad")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_summary([row(**kwargs)])

    def test_csv_is_na_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_summary_csv(path, build_summary([row(duration_ratio=None)], include_empty_slices=False))
            with path.open() as handle:
                saved = list(csv.DictReader(handle))[0]
            self.assertEqual(saved["duration_bias"], "NA")
            self.assertEqual(saved["N"], "1")
            with self.assertRaises(FileExistsError):
                write_summary_csv(path, [])


class BootstrapTests(unittest.TestCase):
    def test_reproducible_video_resampling(self):
        data = [dict(video_name="a", value=0), dict(video_name="a", value=0),
                dict(video_name="b", value=1)]
        result = cluster_bootstrap_mean(data, "value", repetitions=100)
        self.assertEqual(result, cluster_bootstrap_mean(data, "value", repetitions=100))
        self.assertAlmostEqual(result["mean"], 1 / 3)
        self.assertEqual(result["N_videos"], 2)
        self.assertEqual((result["ci_low"], result["ci_high"]), (0, 1))

    def test_missing_and_single_video(self):
        result = cluster_bootstrap_mean([dict(video_name="a", value=None)], "value")
        self.assertIsNone(result["mean"])
        result = cluster_bootstrap_mean([dict(video_name="a", value=1)], "value")
        self.assertEqual(result["mean"], 1)
        self.assertIsNone(result["ci_low"])


if __name__ == "__main__":
    unittest.main()
