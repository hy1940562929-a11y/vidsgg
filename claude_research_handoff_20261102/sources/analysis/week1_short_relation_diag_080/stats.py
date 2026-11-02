"""Auditable aggregation for Week1 temporal diagnostics.

This module only consumes already verified, one-row-per-GT records. It never
selects a prediction, infers a slice, or reads annotations. Rates are fractions
in [0, 1], not percentages. Missing evidence is never treated as a prediction
failure. An explicit ``pred_available=False`` *is* a prediction failure.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SLICES = ("S-S", "S-L-solo", "S-L-comp", "L-L", "L-S", "UNKNOWN")
THRESHOLDS = ((0.3, "03"), (0.5, "05"), (0.7, "07"))
SUMMARY_COLUMNS = [
    "source_scope", "readout", "slice_definition", "analysis_population", "slice", "N", "N_videos", "status",
    "N_pred", "N_pred_observed", "no_pred_count", "no_pred_observed",
    "prediction_availability_missing", "N_tiou_valid", "N_tiou_missing",
    "tiou_sum", "mean_tiou", "mean_tiou_denominator",
]
for _threshold, _tag in THRESHOLDS:
    SUMMARY_COLUMNS += [f"miss_count_tiou{_tag}", f"miss_rate_tiou{_tag}",
                        f"miss_denominator_tiou{_tag}"]
SUMMARY_COLUMNS += [
    "N_ratio_valid", "N_ratio_missing", "stretch_count", "stretch_rate",
    "stretch_denominator", "stretch_rate_all", "stretch_all_denominator",
    "ratio_sum", "duration_bias", "duration_bias_denominator",
    "median_ratio", "p90_ratio",
]


def _number(value: Any, label: str, *, maximum: float | None = None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def _availability(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    # A CSV round trip is accepted, but arbitrary truthy strings are not.
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid pred_available: {value!r}")


def _quantile(values: list[float], probability: float) -> float | None:
    """Linear quantile, identical to NumPy's default for finite values."""
    if not values:
        return None
    values = sorted(values)
    index = probability * (len(values) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def _validated_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if not str(row.get("video_name", "")):
        raise ValueError("Each row requires a nonempty video_name")
    if row.get("slice") not in SLICES:
        raise ValueError(f"Unknown slice: {row.get('slice')!r}")
    result["source_scope"] = str(row.get("source_scope") or "UNSPECIFIED")
    result["readout"] = str(row.get("readout") or "UNSPECIFIED")
    result["slice_definition"] = str(row.get("slice_definition") or "literal")
    result["analysis_population"] = str(row.get("analysis_population") or "all")
    available = _availability(row.get("pred_available"))
    tiou = _number(row.get("tiou"), "tiou", maximum=1)
    ratio = _number(row.get("duration_ratio"), "duration_ratio")
    if available is False:
        if tiou not in (None, 0.0):
            raise ValueError("A missing prediction cannot have positive tIoU")
        if ratio is not None:
            raise ValueError("A missing prediction must have duration_ratio=None, not zero")
        tiou = 0.0
    elif available is None:
        # Do not infer existence from an oracle tIoU or partially imported field.
        tiou = None
        ratio = None
    elif ratio == 0:
        raise ValueError("An available, nonempty prediction must have positive duration_ratio")
    result.update(pred_available=available, tiou=tiou, duration_ratio=ratio)
    return result


def _summarize(rows: list[dict[str, Any]], scope: str, readout: str,
               slice_definition: str, analysis_population: str, slice_name: str) -> dict[str, Any]:
    n = len(rows)
    available = [r for r in rows if r["pred_available"] is True]
    absent = [r for r in rows if r["pred_available"] is False]
    availability_missing = n - len(available) - len(absent)
    tiou_values = [r["tiou"] for r in rows if r["tiou"] is not None]
    ratio_values = [r["duration_ratio"] for r in available if r["duration_ratio"] is not None]
    complete_availability = availability_missing == 0
    complete_tiou = len(tiou_values) == n
    complete_ratio = complete_availability and len(ratio_values) == len(available)
    result = {key: None for key in SUMMARY_COLUMNS}
    result.update(
        source_scope=scope, readout=readout, slice_definition=slice_definition,
        analysis_population=analysis_population, slice=slice_name,
        N=n, N_videos=len({r["video_name"] for r in rows}),
        status=("EMPTY" if not n else "COMPLETE" if complete_tiou and complete_ratio else "INCOMPLETE_METRICS"),
        N_pred=len(available) if complete_availability else None,
        N_pred_observed=len(available),
        no_pred_count=len(absent) if complete_availability else None,
        no_pred_observed=len(absent),
        prediction_availability_missing=availability_missing,
        N_tiou_valid=len(tiou_values), N_tiou_missing=n - len(tiou_values),
        N_ratio_valid=len(ratio_values), N_ratio_missing=len(available) - len(ratio_values),
    )
    if complete_tiou:
        total = math.fsum(tiou_values)
        result.update(tiou_sum=total, mean_tiou=total / n if n else None,
                      mean_tiou_denominator=n)
        for threshold, tag in THRESHOLDS:
            misses = sum(value < threshold for value in tiou_values)
            result.update({f"miss_count_tiou{tag}": misses,
                           f"miss_rate_tiou{tag}": misses / n if n else None,
                           f"miss_denominator_tiou{tag}": n})
    if complete_ratio:
        n_pred = len(available)
        stretch = sum(value >= 2 for value in ratio_values)
        ratio_sum = math.fsum(ratio_values)
        result.update(
            stretch_count=stretch,
            stretch_rate=stretch / n_pred if n_pred else None,
            stretch_denominator=n_pred,
            stretch_rate_all=stretch / n if n else None,
            stretch_all_denominator=n,
            ratio_sum=ratio_sum,
            duration_bias=ratio_sum / n_pred if n_pred else None,
            duration_bias_denominator=n_pred,
            median_ratio=_quantile(ratio_values, 0.5),
            p90_ratio=_quantile(ratio_values, 0.9),
        )
    return result


def build_summary(rows: Iterable[Mapping[str, Any]], *, include_empty_slices: bool = True,
                  include_all: bool = False) -> list[dict[str, Any]]:
    """Aggregate each source/readout separately; duplicate GT identities fail.

    Empty input returns []. For each observed source/readout/definition/population,
    empty slices are emitted with N=0 and undefined rates. An ALL row is opt-in.
    Missing slice_definition and analysis_population default to literal and all.
    """
    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    identities: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = _validated_row(raw)
        group_key = tuple(row[key] for key in ("source_scope", "readout", "slice_definition", "analysis_population"))
        if row.get("gt_index") is not None:
            identity = (*group_key, str(row["video_name"]), str(row["gt_index"]))
            if identity in identities:
                raise ValueError(f"Duplicate GT identity within source/readout: {identity}")
            identities.add(identity)
        grouped[group_key][row["slice"]].append(row)
    result = []
    for group_key, groups in sorted(grouped.items()):
        for slice_name in SLICES:
            if groups.get(slice_name) or include_empty_slices:
                result.append(_summarize(groups.get(slice_name, []), *group_key, slice_name))
        if include_all:
            all_rows = [row for slice_name in SLICES for row in groups.get(slice_name, [])]
            result.append(_summarize(all_rows, *group_key, "ALL"))
    return result


def write_summary_csv(path: str | Path, summary: Iterable[Mapping[str, Any]]) -> None:
    """Write explicit NA for undefined or unavailable metrics, without overwriting."""
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in summary:
            writer.writerow({key: "NA" if row.get(key) is None else row[key] for key in SUMMARY_COLUMNS})


def cluster_bootstrap_mean(rows: Iterable[Mapping[str, Any]], value_key: str, *,
                           repetitions: int = 2000, seed: int = 80) -> dict[str, Any]:
    """GT-weighted mean with whole-video resampling; descriptive, not causal.

    Missing any requested value makes the full-cohort estimate unavailable.
    At least two videos are needed for a confidence interval. This helper does
    not control confounding or claim significance from an incomplete cohort.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    data = list(rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    missing = 0
    for row in data:
        value = row.get(value_key)
        if value is None or value == "":
            missing += 1
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite bootstrap value for {value_key}")
        if not str(row.get("video_name", "")):
            raise ValueError("Each bootstrap row requires video_name")
        grouped[str(row["video_name"])].append(value)
    output = dict(N=len(data), N_missing=missing, N_videos=len(grouped),
                  mean=None, ci_low=None, ci_high=None, repetitions=repetitions,
                  seed=seed, unit="video_cluster", estimand="GT_weighted_mean")
    if missing or not data:
        return output
    output["mean"] = math.fsum(value for values in grouped.values() for value in values) / len(data)
    videos = sorted(grouped)
    if len(videos) < 2:
        return output
    moments = [(math.fsum(grouped[v]), len(grouped[v])) for v in videos]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        selected = [moments[rng.randrange(len(moments))] for _ in moments]
        estimates.append(math.fsum(item[0] for item in selected) / sum(item[1] for item in selected))
    output.update(ci_low=_quantile(estimates, 0.025), ci_high=_quantile(estimates, 0.975))
    return output
