#!/usr/bin/env python3
"""CPU-only descriptive controls for Week1 H1/H2/H5 predictions.

Estimand: within each exact covariate stratum represented on both sides, compute
the difference of outcome means. Weight both sides by the pooled stratum GT
mass, (n0+n1)/N_common. This is a common-support descriptive association, not a
causal effect. It is recomputed separately for every outcome and population.

H1: short (<3s) minus long (>=3s), exact predicate + coexist bins [3,10,20].
H2: short/long-coexist/solo minus short/short-coexist, exact predicate + 0.25s
duration bins. A required sensitivity also excludes competitors on side 0.
H5: short/long-coexist/overlapping-different-long-predicate minus the matching
solo group, predicate + 0.25s duration + coexist bins. A sensitivity adds video
GT-load bins [50,100,200], fixed before viewing controlled results.

Main bootstrap samples whole videos WITH replacement; all GT from a video
share its multiplicity, including duplicated video draws. Rates AND pooled
stratum weights are re-estimated. A stratum losing a side is omitted in that
replicate, and original-support retention is reported. Low retention makes the
judgment inconclusive. The inherited 077 Exp(1) positive video multiplier
bootstrap is a separately named sensitivity preserving all original strata.

Only binary outcomes have the preset 3pp decision. Each side needs >=20 common-
support videos, total support >=60 GT, >=95% valid bootstrap replicates and
retention p05>=80%. No controlled intervention => mechanism always UNDECIDED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

COEXIST_EDGES = (3.0, 10.0, 20.0)
LOAD_EDGES = (50, 100, 200)
DEFINITION = "overlap_other_pred_ge3"
MISSING = (None, "", "NA", "None", "nan")
OUTCOMES = ("miss_tiou03", "miss_tiou05", "miss_tiou07", "stretch",
            "overlap_stretch", "no_overlap_stretch", "pred_interval_disjoint",
            "duration_ratio", "tiou", "VrdONE_official_miss", "BIG_official_miss")
OFFICIAL_OUTCOMES = {"VrdONE_official_miss": "VrdONE_hit", "BIG_official_miss": "BIG_hit"}
OFFICIAL_PROTOCOL = "original Top100 category+subject/object vIoU>=0.5 one-to-one matching; NOT temporal-only tIoU"


def number(value: Any) -> float | None:
    if value in MISSING:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def boolean(value: Any) -> bool | None:
    if value in MISSING:
        return None
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def object_value(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(value) if value not in MISSING else {}


def _competition(row: Mapping, definition: str) -> bool | None:
    flags = object_value(row.get("competition_flags"))
    if definition in flags:
        return boolean(flags[definition])
    if definition in row:
        return boolean(row[definition])
    # Long-coexist slice labels imply a flag; S-S itself does not.
    slices = object_value(row.get("slices"))
    label = slices.get(definition)
    if label in ("S-L-solo", "S-L-comp"):
        return label == "S-L-comp"
    return None


def _in_population(row: Mapping, population: str) -> bool:
    if population == "all":
        return True
    if population in ("F", "native_F"):
        return boolean(row.get("native_F")) is True
    if population == "chosen_pair_covers_gt":
        return boolean(row.get("chosen_pair_covers_gt")) is True
    if population == "chosen_pair_covers_gt_unique_annotation":
        return (boolean(row.get("chosen_pair_covers_gt")) is True
                and number(row.get("pair_predicate_annotation_count")) == 1)
    raise ValueError(f"Unknown population {population!r}")


def _design(row: Mapping, hypothesis: str, *, definition: str,
            h2_no_comp_control: bool, load_control: bool) -> tuple[int | None, tuple | None, str]:
    duration, coexist = number(row.get("relation_sec")), number(row.get("coexist_sec"))
    if duration is None or coexist is None or duration <= 0 or coexist <= 0 or coexist + 1e-6 < duration:
        return None, None, "invalid_time_covariates"
    predicate = row.get("predicate")
    if predicate in MISSING:
        return None, None, "missing_predicate"
    co_bin = sum(coexist >= boundary for boundary in COEXIST_EDGES)
    duration_bin = math.floor(duration / 0.25)
    if hypothesis == "H1":
        side, fields = int(duration < 3), (str(predicate), co_bin)
    elif hypothesis == "H2":
        if duration >= 3:
            return None, None, "outside_contrast"
        competitor = _competition(row, definition)
        if coexist >= 10 or h2_no_comp_control:
            if competitor is None:
                return None, None, "missing_competition_flag"
            if competitor:
                return None, None, "outside_contrast"
        side, fields = int(coexist >= 10), (str(predicate), duration_bin)
    elif hypothesis == "H5":
        if duration >= 3 or coexist < 10:
            return None, None, "outside_contrast"
        competitor = _competition(row, definition)
        if competitor is None:
            return None, None, "missing_competition_flag"
        side, fields = int(competitor), (str(predicate), duration_bin, co_bin)
    else:
        raise ValueError(f"Unknown hypothesis {hypothesis}")
    if load_control:
        load = number(row.get("video_GT_load"))
        if load is None or load <= 0:
            return None, None, "missing_video_load"
        fields += (sum(load >= boundary for boundary in LOAD_EDGES),)
    return side, fields, "eligible"


def _outcome(row: Mapping, name: str) -> tuple[float | None, str]:
    if name in OFFICIAL_OUTCOMES:
        value = number(row.get(OFFICIAL_OUTCOMES[name]))
        if value is None:
            return None, "missing_outcome"
        if value not in (0, 1):
            raise ValueError(f"{OFFICIAL_OUTCOMES[name]} must be exactly 0 or 1")
        return 1.0 - value, "observed"
    available = boolean(row.get("pred_available"))
    if name.startswith("miss_tiou") or name == "tiou":
        if available is None:
            # A numeric oracle value does not establish an actual selected
            # prediction. This comparison intentionally excludes oracle reads.
            return None, "missing_outcome"
        value = number(row.get("tiou"))
        if available is False:
            if value not in (None, 0):
                raise ValueError("No prediction cannot have positive tIoU")
            value = 0.0
        if value is None:
            return None, "missing_outcome"
        if not 0 <= value <= 1:
            raise ValueError("tIoU outside [0,1]")
        if name == "tiou":
            return value, "observed"
        threshold = {"miss_tiou03": .3, "miss_tiou05": .5, "miss_tiou07": .7}[name]
        return float(value < threshold), "observed"
    if available is False:
        return None, "no_prediction_conditional_exclusion"
    if available is None:
        return None, "missing_outcome"
    if name == "pred_interval_disjoint":
        value = boolean(row.get(name))
        return (None, "missing_outcome") if value is None else (float(value), "observed")
    ratio = number(row.get("duration_ratio"))
    if ratio is None:
        return None, "missing_outcome"
    if ratio <= 0:
        raise ValueError("Available prediction requires positive duration ratio")
    if name == "duration_ratio":
        return ratio, "observed"
    if name == "stretch":
        return float(ratio >= 2), "observed"
    if name in ("overlap_stretch", "no_overlap_stretch"):
        # Prefer explicit independently joined boundary labels. Falling back to
        # the disjoint flag is equivalent, but never derive it from tIoU alone.
        explicit = boolean(row.get(name))
        if explicit is not None:
            if explicit and ratio < 2:
                raise ValueError("Stretch flag inconsistent with ratio <2")
            return float(explicit), "observed"
        disjoint = boolean(row.get("pred_interval_disjoint"))
        if disjoint is None:
            return None, "missing_outcome"
        return float(ratio >= 2 and disjoint == (name == "no_overlap_stretch")), "observed"
    raise ValueError(f"Unknown outcome {name}")


def _weighted_estimate(cell, outcomes, row_weights, n_strata, original_mass):
    """Return reweighted means and retained masses; public only for tests."""
    import numpy as np
    sizes = np.bincount(cell, weights=row_weights, minlength=2*n_strata).reshape(-1, 2)
    sums = np.bincount(cell, weights=row_weights*outcomes, minlength=2*n_strata).reshape(-1, 2)
    keep = np.all(sizes > 0, axis=1)
    if not np.any(keep):
        return None, 0.0, 0.0
    mass = sizes[keep].sum(axis=1)
    weights = mass / mass.sum()
    means = (weights[:, None] * sums[keep] / sizes[keep]).sum(axis=0)
    retained_original = float(original_mass[keep].sum() / original_mass.sum())
    retained_draw = float(sizes[keep].sum() / sizes.sum())
    return (float(means[0]), float(means[1])), retained_original, retained_draw


def video_bootstrap(supported: list[dict], n_strata: int, *, repetitions: int = 2000,
                    seed: int = 80, method: str = "resample") -> dict:
    """One multiplier per VIDEO, not per GT or per stratum; duplicate-safe."""
    import numpy as np
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if method not in ("resample", "exp_multiplier"):
        raise ValueError("Unknown bootstrap method")
    videos = sorted({row["video_name"] for row in supported})
    output = dict(method=("whole_video_with_replacement" if method == "resample" else "positive_video_Exp1_multiplier_077"),
                  seed=seed, repetitions_requested=repetitions, repetitions_valid=0,
                  N_videos=len(videos), ci_low=None, ci_high=None,
                  retained_original_support_mass_p05=None, retained_draw_support_mass_p05=None)
    if not supported or len(videos) < 2:
        return output
    indices = {v: i for i, v in enumerate(videos)}
    video_index = np.asarray([indices[r["video_name"]] for r in supported])
    cell = np.asarray([2*r["stratum_id"] + r["side"] for r in supported])
    values = np.asarray([r["outcome"] for r in supported], dtype=float)
    original_mass = np.bincount(cell, minlength=2*n_strata).reshape(-1, 2).sum(axis=1)
    rng = np.random.default_rng(seed)
    gaps, original_retention, draw_retention = [], [], []
    for _ in range(repetitions):
        if method == "resample":
            draws = rng.integers(0, len(videos), size=len(videos))
            multiplicity = np.bincount(draws, minlength=len(videos))
        else:
            multiplicity = np.maximum(rng.exponential(1, len(videos)), np.finfo(float).tiny)
        estimate, original, drawn = _weighted_estimate(cell, values, multiplicity[video_index], n_strata, original_mass)
        # Include failed replicates as zero retention, never hide instability.
        original_retention.append(original)
        draw_retention.append(drawn)
        if estimate is not None:
            gaps.append(estimate[1] - estimate[0])
    output.update(repetitions_valid=len(gaps),
                  ci_low=float(np.quantile(gaps, .025)) if gaps else None,
                  ci_high=float(np.quantile(gaps, .975)) if gaps else None,
                  retained_original_support_mass_p05=float(np.quantile(original_retention, .05)),
                  retained_draw_support_mass_p05=float(np.quantile(draw_retention, .05)))
    return output


def judgment(result: Mapping, *, minimum_videos_each: int = 20, minimum_gt: int = 60,
             minimum_effect: float = .03) -> dict:
    """Judge the observable prediction, never the named causal mechanism."""
    base = dict(mechanism_status="未决", scope="observational common-support association",
                effect_threshold=minimum_effect, prediction_status="未决")
    if result.get("N_missing_outcome", 0):
        return dict(base, reason="missing_outcomes_not_counted_as_failures")
    if result.get("N_common_support", 0) < minimum_gt or min(result.get("N_videos_side0", 0), result.get("N_videos_side1", 0)) < minimum_videos_each:
        return dict(base, reason="insufficient_common_support")
    boot = result.get("bootstrap", {})
    if (boot.get("ci_low") is None or boot.get("ci_high") is None
            or boot.get("repetitions_valid", 0) < .95 * boot.get("repetitions_requested", 1)
            or (boot.get("retained_original_support_mass_p05") or 0) < .8):
        return dict(base, reason="unstable_video_bootstrap_support")
    if result["outcome"] in ("tiou", "duration_ratio"):
        return dict(base, reason="no_preregistered_effect_size_for_continuous_outcome")
    if result["standardized_gap"] >= minimum_effect and boot["ci_low"] > 0:
        return dict(base, prediction_status="暂成立", reason="preset_positive_descriptive_association")
    if boot["ci_high"] < minimum_effect:
        return dict(base, prediction_status="证伪", reason="at_least_3pp_prediction_not_supported_in_this_population")
    return dict(base, reason="imprecise_effect")


def compare(rows: Iterable[Mapping[str, Any]], hypothesis: str, *, outcome: str = "miss_tiou05",
            population: str = "all", definition: str = DEFINITION,
            h2_no_comp_control: bool = False, load_control: bool = False,
            repetitions: int = 2000, seed: int = 80, bootstrap_method: str = "resample",
            minimum_each_stratum: int = 1) -> dict:
    """Return point/CI, stratum sufficient statistics, exclusions and membership.

    Input must be one record per GT and exactly one source/readout. Source CSVs
    from run.py are accepted directly (JSON columns are decoded on demand).
    Official miss outcomes require explicit source_scope and a readout label
    containing 'official', 'Top100' and 'vIoU', e.g. official_Top100_vIoU50.
    They use the original spatial-tube detection hit, never temporal-only IoU.
    For stretch outcomes, population is conditional on available predictions;
    excluded no-prediction GT remain explicitly counted. A missing metric never
    becomes zero. Sparse unsupported strata remain outside the estimand.
    """
    if outcome not in OUTCOMES:
        raise ValueError(outcome)
    if minimum_each_stratum < 1:
        raise ValueError("minimum_each_stratum must be >=1")
    if load_control and hypothesis != "H5":
        raise ValueError("Video-load sensitivity is preregistered for H5 only")
    rows = [dict(row) for row in rows]
    scopes = {(str(r.get("source_scope", "UNSPECIFIED")), str(r.get("readout", "UNSPECIFIED"))) for r in rows}
    if len(scopes) > 1:
        raise ValueError("Run each source_scope/readout separately")
    if outcome in OFFICIAL_OUTCOMES:
        for scope, readout in scopes:
            if scope in ("", "None", "UNSPECIFIED") or not all(token in readout.lower() for token in ("official", "top100", "viou")):
                raise ValueError("Official outcomes require explicit source_scope and readout such as official_Top100_vIoU50")
    seen = set()
    strata = defaultdict(lambda: [[], []])
    exclusions = defaultdict(int)
    population_rows = [r for r in rows if _in_population(r, population)]
    population_side_n = [0, 0]
    for row in population_rows:
        identity = (str(row["video_name"]), str(row["gt_index"]))
        if identity in seen:
            raise ValueError(f"Duplicate GT {identity}")
        seen.add(identity)
        side, key, status = _design(row, hypothesis, definition=definition,
                                    h2_no_comp_control=h2_no_comp_control, load_control=load_control)
        if status != "eligible":
            exclusions[status] += 1
            continue
        population_side_n[side] += 1
        value, status = _outcome(row, outcome)
        if value is None:
            exclusions[status] += 1
            continue
        strata[key][side].append(dict(row, side=side, outcome=value))
    supported, details = [], []
    for key, sides in sorted(strata.items()):
        if min(len(sides[0]), len(sides[1])) < minimum_each_stratum:
            exclusions["no_common_support"] += sum(map(len, sides))
            continue
        sid = len(details)
        details.append(dict(stratum_id=sid, stratum_key=list(key), N_0=len(sides[0]), N_1=len(sides[1]),
                            numerator_0=math.fsum(r["outcome"] for r in sides[0]),
                            numerator_1=math.fsum(r["outcome"] for r in sides[1])))
        for side in (0, 1):
            supported.extend(dict(r, stratum_id=sid) for r in sides[side])
    n = len(supported)
    point = [None, None]
    weighted_numerators = [None, None]
    if n:
        for d in details:
            d.update(pooled_weight_numerator=d["N_0"] + d["N_1"], pooled_weight_denominator=n)
        for side in (0, 1):
            weighted_numerators[side] = math.fsum((d["N_0"]+d["N_1"]) * d[f"numerator_{side}"] / d[f"N_{side}"] for d in details)
            point[side] = weighted_numerators[side] / n
    sides = [[r for r in supported if r["side"] == side] for side in (0, 1)]
    scope, readout = next(iter(scopes)) if scopes else ("UNSPECIFIED", "UNSPECIFIED")
    result = dict(hypothesis=hypothesis, outcome=outcome, source_scope=scope, readout=readout,
                  population=population, definition=definition, h2_no_comp_control=h2_no_comp_control,
                  load_control=load_control, coexist_edges=list(COEXIST_EDGES), duration_bin_width=.25,
                  load_edges=list(LOAD_EDGES) if load_control else None, minimum_each_stratum=minimum_each_stratum,
                  direction="side1 minus side0", side0={"H1":"long", "H2":"short_short_coexist_no_comp" if h2_no_comp_control else "short_short_coexist_all", "H5":"short_long_coexist_solo"}[hypothesis],
                  side1={"H1":"short", "H2":"short_long_coexist_solo", "H5":"short_long_coexist_comp"}[hypothesis],
                  N_input=len(rows), N_population=len(population_rows), N_contrast_side0=population_side_n[0],
                  N_contrast_side1=population_side_n[1], N_common_support=n,
                  N_videos_common_support=len({r["video_name"] for r in supported}),
                  N_side0=len(sides[0]), N_side1=len(sides[1]),
                  N_videos_side0=len({r["video_name"] for r in sides[0]}),
                  N_videos_side1=len({r["video_name"] for r in sides[1]}),
                  N_missing_outcome=exclusions["missing_outcome"], exclusions=dict(exclusions),
                  N_strata=len(details), standardized_side0=point[0], standardized_side1=point[1],
                  weighted_numerator_side0=weighted_numerators[0], weighted_numerator_side1=weighted_numerators[1],
                  weighted_denominator=n, standardized_gap=point[1]-point[0] if n else None,
                  standardized_gap_pp=100*(point[1]-point[0]) if n and outcome not in ("duration_ratio", "tiou") else None,
                  raw_same_support_side0=math.fsum(r["outcome"] for r in sides[0])/len(sides[0]) if sides[0] else None,
                  raw_same_support_side1=math.fsum(r["outcome"] for r in sides[1])/len(sides[1]) if sides[1] else None,
                  strata=details,
                  membership=[{k:r[k] for k in ("video_name", "gt_index", "side", "stratum_id", "outcome")} for r in supported])
    result["bootstrap"] = video_bootstrap(supported, len(details), repetitions=repetitions, seed=seed, method=bootstrap_method)
    result["judgment"] = judgment(result)
    result["metric_protocol"] = OFFICIAL_PROTOCOL if outcome in OFFICIAL_OUTCOMES else "selected-prediction temporal diagnostics; not official detection recall"
    result["interpretation"] = ("Stretch/disjoint outcomes are conditional on a prediction; no-overlap stretches do not establish boundary extension. "
                                "Fractional standardized numerators are weights, not recovered GT instances. Mechanisms remain unverified.")
    return result


def load_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="New JSON file; refuses overwrite")
    parser.add_argument("--hypothesis", choices=("H1", "H2", "H5"), required=True)
    parser.add_argument("--outcome", choices=OUTCOMES, default="miss_tiou05")
    parser.add_argument("--population", default="all")
    parser.add_argument("--source-scope", help="Explicit provenance label for an unlabeled source CSV")
    parser.add_argument("--readout", help="Explicit readout label; official outcomes require official_Top100_vIoU50 or equivalent")
    parser.add_argument("--definition", default=DEFINITION)
    parser.add_argument("--h2-no-comp-control", action="store_true")
    parser.add_argument("--load-control", action="store_true")
    parser.add_argument("--bootstrap-method", choices=("resample", "exp_multiplier"), default="resample")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=80)
    args = parser.parse_args()
    rows = load_csv(args.input)
    for item in rows:
        for field, requested in (("source_scope", args.source_scope), ("readout", args.readout)):
            if requested is not None:
                if item.get(field) not in (None, "", requested):
                    raise ValueError(f"Refusing to overwrite existing {field}: {item[field]}")
                item[field] = requested
    result = compare(rows, args.hypothesis, outcome=args.outcome,
                     population=args.population, definition=args.definition,
                     h2_no_comp_control=args.h2_no_comp_control, load_control=args.load_control,
                     repetitions=args.repetitions, seed=args.seed, bootstrap_method=args.bootstrap_method)
    result["provenance"] = {"input": str(Path(args.input).resolve()),
                            "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "training_or_model_execution": False}
    with Path(args.out).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
