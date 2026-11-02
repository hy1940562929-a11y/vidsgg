#!/usr/bin/env python3
"""081 post-080 CPU exploration; never chooses a new query or runs a model."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "week1_short_relation_diag_080"
SOURCE = PREVIOUS / "results_v1" / "per_gt_boundary_joined.csv"
CONTROLLED = PREVIOUS / "controlled.py"
POPULATIONS = ("all", "chosen_pair_covers_gt")
STATES = ("selected_hit", "oracle_only_hit", "neither")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value):
    if value is True or str(value).lower() in ("true", "1"):
        return True
    if value is False or str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Unspecified boolean {value!r}")


def optional(value):
    if value in (None, "", "None", "NA"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite input")
    return result


def tiou(a, b, c, d):
    intersection = max(0., min(b, d) - max(a, c))
    return intersection / (b-a+d-c-intersection)


def pair_key(row):
    return row["video_name"], row["subject_tid"], row["object_tid"]


def classify_readout(selected, oracle):
    if selected >= .5 and oracle < .5:
        raise ValueError("Selected hit missing from oracle coverage")
    return "selected_hit" if selected >= .5 else "oracle_only_hit" if oracle >= .5 else "neither"


def validate_and_group(raw):
    """Construct labels BEFORE any subgroup filter; require complete videos."""
    rows = []
    identities = set()
    for original in raw:
        row = dict(original)
        for name in ("gt_index", "subject_tid", "object_tid", "begin_fid", "end_fid",
                     "pair_annotation_count", "pair_predicate_annotation_count", "video_GT_load"):
            row[name] = int(row[name])
        for name in ("relation_sec", "coexist_sec", "tiou", "oracle_tiou", "fps"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                raise ValueError(f"Nonfinite {name}")
        for name in ("pred_available", "chosen_pair_covers_gt"):
            row[name] = truth(row[name])
        for name in ("duration_ratio", "pred_begin", "pred_end"):
            row[name] = optional(row.get(name))
        if row["source_scope"] != "GT27_correct_entity_input" or row["readout"] != "target_predicate_maxclass_query_min5":
            raise ValueError("Unexpected source/readout; do not pool or silently replace source")
        identity = row["video_name"], row["gt_index"]
        if identity in identities:
            raise ValueError(f"Duplicate GT {identity}")
        identities.add(identity)
        a, b, fps = row["begin_fid"], row["end_fid"], row["fps"]
        if b <= a or fps <= 0 or abs((b-a)/fps - row["relation_sec"]) > 1e-7:
            raise ValueError("Inconsistent ground truth time")
        if not (0 <= row["tiou"] <= 1 and 0 <= row["oracle_tiou"] <= 1):
            raise ValueError("tIoU outside [0,1]")
        if row["oracle_tiou"] + 1e-8 < row["tiou"]:
            raise ValueError("Oracle best is smaller than fixed selected tIoU")
        if row["pred_available"]:
            p, q = row["pred_begin"], row["pred_end"]
            if p is None or q is None or q-p < 5:
                raise ValueError("Invalid selected interval")
            if abs(tiou(p, q, a, b) - row["tiou"]) > 1e-9:
                raise ValueError("Selected tIoU does not match saved endpoints")
            if row["duration_ratio"] is None or abs((q-p)/(b-a) - row["duration_ratio"]) > 1e-9:
                raise ValueError("Duration ratio mismatch")
        elif row["tiou"] != 0 or row["duration_ratio"] is not None:
            raise ValueError("No prediction must have tIoU0 and undefined duration ratio")
        row["readout_state"] = classify_readout(row["tiou"], row["oracle_tiou"])
        row["duration_group"] = "<1" if row["relation_sec"] < 1 else "1-3" if row["relation_sec"] < 3 else ">=3"
        rows.append(row)
    videos = collections.Counter(row["video_name"] for row in rows)
    pairs = collections.defaultdict(list)
    pair_predicate = collections.Counter((*pair_key(row), row["predicate"]) for row in rows)
    for row in rows:
        pairs[pair_key(row)].append(row)
    for row in rows:
        pair = pairs[pair_key(row)]
        repeat_count = pair_predicate[(*pair_key(row), row["predicate"])]
        if row["video_GT_load"] != videos[row["video_name"]]:
            raise ValueError("Incomplete video GT: cannot reconstruct mixed annotation labels")
        if row["pair_annotation_count"] != len(pair) or row["pair_predicate_annotation_count"] != repeat_count:
            raise ValueError("Pair annotation counts disagree with full-video records")
        durations = {member["relation_sec"] < 3 for member in pair}
        row["repeat_group"] = "unique" if repeat_count == 1 else "repeated"
        row["pair_mix_group"] = "mixed_short_long" if len(durations) == 2 else "short_only" if True in durations else "long_only"
    return rows


def population_rows(rows, population):
    if population == "all":
        return rows
    if population == "chosen_pair_covers_gt":
        return [row for row in rows if row["chosen_pair_covers_gt"]]
    raise ValueError(population)


def decompose(rows, population):
    selected = population_rows(rows, population)
    views = {"overall": (), "duration": ("duration_group",), "repeat": ("repeat_group",),
             "pair_mix": ("pair_mix_group",),
             "duration_x_repeat_x_pair_mix": ("duration_group", "repeat_group", "pair_mix_group")}
    output = []
    for view, fields in views.items():
        groups = collections.defaultdict(list)
        for row in selected:
            groups[tuple(row[field] for field in fields)].append(row)
        for key, members in sorted(groups.items()):
            result = dict(population=population, view=view, duration_group="ALL", repeat_group="ALL", pair_mix_group="ALL",
                          N=len(members), N_videos=len({row["video_name"] for row in members}), population_N=len(selected),
                          group_fraction_numerator=len(members), group_fraction_denominator=len(selected),
                          group_fraction=len(members)/len(selected) if selected else None)
            result.update(zip(fields, key))
            for state in STATES:
                count = sum(row["readout_state"] == state for row in members)
                result[state + "_count"] = count
                result[state + "_denominator"] = len(members)
                result[state + "_rate"] = count/len(members)
            if sum(result[state + "_count"] for state in STATES) != result["N"]:
                raise ValueError("Readout partition not exhaustive")
            output.append(result)
        if sum(record["N"] for record in output if record["view"] == view) != len(selected):
            raise ValueError("View does not partition population")
    return output


def support_status(result):
    b = result["bootstrap"]
    if result["N_common_support"] < 60 or min(result["N_videos_side0"], result["N_videos_side1"]) < 20:
        return "insufficient_support"
    if (result["N_missing_outcome"] or b["ci_low"] is None or b["ci_high"] is None
            or b["repetitions_valid"] < .95*b["repetitions_requested"]
            or (b["retained_original_support_mass_p05"] or 0) < .8):
        return "unstable_or_incomplete"
    return "support_gate_passed_no_prespecified_continuous_effect_threshold"


def flatten(result):
    b = result["bootstrap"]
    result_keys = ("population", "load_control", "N_population", "N_contrast_side0", "N_contrast_side1",
                   "N_common_support", "N_videos_common_support", "N_side0", "N_side1", "N_videos_side0", "N_videos_side1",
                   "N_strata", "N_missing_outcome", "standardized_side0", "standardized_side1",
                   "weighted_numerator_side0", "weighted_numerator_side1", "weighted_denominator", "standardized_gap")
    output = {key: result[key] for key in result_keys}
    output.update(bootstrap_method=b["method"], ci_low=b["ci_low"], ci_high=b["ci_high"],
                  bootstrap_valid=b["repetitions_valid"], bootstrap_requested=b["repetitions_requested"],
                  retention_p05=b["retained_original_support_mass_p05"], support_status=support_status(result),
                  mechanism_status="UNDECIDED", outcome="continuous_tIoU", direction="comp_minus_solo_negative_expected",
                  source_scope=result["source_scope"], readout=result["readout"], design_status="exploratory_after_080")
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def fraction(n, denominator):
    return f"{n:,}/{denominator:,} = {100*n/denominator:.2f}%" if denominator else f"{n}/{denominator}（未定义）"


def f(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def report(control_rows, decomposition, rows):
    lines = ["# 081：连续时间重合与固定读出损失", "", "本轮为080之后的CPU探索。固定原query、原选中实体段和最短5帧规则，无训练、无新模型推理。",
             f"范围：{len(rows):,}条GT、{len({r['video_name'] for r in rows})}视频，正确实体输入的GT27诊断。结果不等同于完整VidOR检测表现。", "",
             "## 1. 另一长谓词并存时，固定输出的时间重合是否下降", "",
             "差值为comp减solo，预期负向。精确控制谓词、0.25秒关系时长档及共存档；负载控制为敏感性。数值为连续tIoU单位，不是命中率百分点。",
             "",
             "|人群|负载控制|bootstrap|solo / comp共同支持GT；视频|solo均值|comp均值|差值与95%区间|保留率p05|支持检查|",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in control_rows:
        lines.append(f"|{r['population']}|{r['load_control']}|{r['bootstrap_method']}|{r['N_side0']}/{r['N_side1']}；{r['N_videos_side0']}/{r['N_videos_side1']}|"
                     f"{f(r['weighted_numerator_side0'])}/{r['weighted_denominator']} = {f(r['standardized_side0'])}|"
                     f"{f(r['weighted_numerator_side1'])}/{r['weighted_denominator']} = {f(r['standardized_side1'])}|"
                     f"{f(r['standardized_gap'])} [{f(r['ci_low'])}, {f(r['ci_high'])}]|{f(r['retention_p05'])}|{r['support_status']}|")
    lines += ["", "加权分子是标准化后的分数总和，不是新增正确GT数。共同支持不等于全体群体；完整筛除数、逐层分子分母和GT成员保存在原始JSON中。",
              "本轮没有预设连续效果阈值，因此不作H5机制成立/证伪判决。负差仅支持一种条件关联；正差或跨零也原样保留。", "",
              "## 2. 固定输出与候选存在性的互斥分解", "",
              "selected_hit是原固定输出命中；oracle_only_hit是原输出未命中、但已存任意query存在可匹配时间；neither是两者都无命中。三者互斥穷尽。oracle_only不是实际恢复。", ""]
    for population in POPULATIONS:
        lines += [f"### {population}", "", "|分组|组规模/该人群|selected_hit|oracle_only_hit|neither|", "|---|---|---|---|---|"]
        for r in decomposition:
            if r["population"] != population or r["view"] not in ("overall", "duration", "repeat", "pair_mix"):
                continue
            label = {"overall": "总体", "duration": r["duration_group"], "repeat": r["repeat_group"], "pair_mix": r["pair_mix_group"]}[r["view"]]
            lines.append("|" + "|".join([label, fraction(r['N'], r['population_N'])] +
                                         [fraction(r[state+'_count'], r['N']) for state in STATES]) + "|")
    lines += ["", "各维度的边际行之间会重叠，不能跨维度相加。完整三维交叉互斥分解见decomposition.csv；每个维度分别加总回各自人群。",
              "重复表示同视频同有序实体对同谓词至少两条标注，未要求时间分离。mixed_short_long表示同pair有短、长标注，不要求同时重叠，不能直接解释为竞争。", "",
              "## 3. 本轮可说与不可说", "",
              "- 连续时间重合可以暴露二值0.5阈值未显示的差异，但若视频数或共同支持不足，不能据此宣布稳定效应。",
              "- oracle_only说明候选集合与固定读出之间存在诊断余量；它混合了query选择、实体段选择、类别分数、重复事件区间复用等环节，不能直接认定是某一模块的原因。",
              "- 正确实体输入固定了实体几何来源，但没有保证所选连续实体段和所选query适合当前GT事件。覆盖GT子群只用于敏感性。",
              "- 这轮不证明长关系压制短关系，不改变073/075/078负结果，也没有训练改善或真实恢复数量。", ""]
    return "\n".join(lines)


def run(out):
    out = Path(out)
    freeze = json.loads((HERE/"protocol_freeze.json").read_text())
    if sha(HERE/"PROTOCOL.md") != freeze["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze")
    source_hash, controlled_hash = sha(SOURCE), sha(CONTROLLED)
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = validate_and_group(list(csv.DictReader(handle)))
    if len(rows) != 2573 or len({r["video_name"] for r in rows}) != 27:
        raise ValueError("Unexpected GT27 source coverage")
    out.mkdir(parents=True, exist_ok=False)
    dependency = importlib.util.spec_from_file_location("controlled080", CONTROLLED)
    controlled = importlib.util.module_from_spec(dependency)
    dependency.loader.exec_module(controlled)
    control_rows = []
    for population in POPULATIONS:
        for load_control in (False, True):
            for method in ("resample", "exp_multiplier"):
                result = controlled.compare(rows, "H5", outcome="tiou", population=population,
                                            definition="overlap_other_pred_ge3", load_control=load_control,
                                            repetitions=2000, seed=80, bootstrap_method=method)
                filename = f"H5_{population}_load{int(load_control)}_{method}.json.gz"
                with gzip.open(out/filename, "wt", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, allow_nan=False)
                flat = flatten(result)
                flat["detail_file"] = filename
                control_rows.append(flat)
                print(json.dumps(flat, ensure_ascii=False), flush=True)
    decomposition = [r for population in POPULATIONS for r in decompose(rows, population)]
    write_csv(out/"summary.csv", control_rows)
    write_csv(out/"decomposition.csv", decomposition)
    fields = ("video_name", "gt_index", "subject_tid", "object_tid", "predicate", "relation_sec", "coexist_sec",
              "duration_group", "repeat_group", "pair_mix_group", "readout_state", "tiou", "oracle_tiou",
              "chosen_pair_covers_gt", "pred_available", "pair_annotation_count", "pair_predicate_annotation_count")
    write_csv(out/"per_gt_partition.csv", [{key:row[key] for key in fields} for row in rows])
    (out/"REPORT.md").write_text(report(control_rows, decomposition, rows), encoding="utf-8")
    if sha(SOURCE) != source_hash or sha(CONTROLLED) != controlled_hash:
        raise ValueError("Source or controlled dependency changed while running")
    manifest = dict(complete=True, date="2026-10-19", design_status="exploratory_after_080_not_independent_preregistration",
                    model_or_training_execution=False, input_path=str(SOURCE), input_sha256=source_hash,
                    controlled_path=str(CONTROLLED), controlled_sha256=controlled_hash,
                    protocol_sha256=freeze["protocol_sha256"], script_sha256=sha(Path(__file__)),
                    tests_sha256=sha(HERE/"test_run.py"), python_version=platform.python_version(),
                    N_GT=len(rows), N_videos=27, sampling_seed=80, bootstrap_repetitions=2000,
                    no_new_query_selection=True, same_input_after_run=True,
                    outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE/"results_v1"))
    args = parser.parse_args()
    run(args.out)
