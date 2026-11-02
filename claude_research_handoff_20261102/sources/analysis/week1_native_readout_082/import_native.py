#!/usr/bin/env python3
"""Validate an existing 082 CPU export and summarize without model execution.

Default acceptance requires the complete frozen 835-video/30,142-GT export.
The optional partial switch accepts only a completed, explicitly bounded smoke
export, never a running/failed export. Diagnostic temporal readout can reuse one
saved prediction for multiple GT events; it is NOT official one-to-one R@K.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_METADATA = HERE.parent/"week1_short_relation_diag_080/results_v1/per_gt_slices.csv"
DEFAULT_PACKAGE = ROOT/"handoff/outgoing/VIDOR-WEEK1-NATIVE-EXPORT-082"
STATS = HERE.parent/"week1_short_relation_diag_080/stats.py"
TOPKS = (50, 100, 200)
DEFINITIONS = ("literal_other_ge3_anywhere", "overlap_other_pred_ge3", "overlap_other_pred_ge3_half", "overlap_other_pred_ge10")
METADATA_FIELDS = ("video_name", "gt_index", "subject_tid", "object_tid", "predicate", "subject_class", "object_class",
                   "begin_fid", "end_fid", "fps", "relation_sec", "coexist_sec", "native_F", "original_group", "VrdONE_hit", "BIG_hit")
NULL_PRED_FIELDS = ("pred_begin_fid", "pred_end_fid", "pred_duration_sec", "duration_ratio", "completely_disjoint",
                    "saved_rank_1based", "pred_score", "prediction_key", "source_sid", "source_oid", "source_pair_idx",
                    "source_candidate_id", "source_query", "selected_input_eligible")
SOURCE_EXPECTED = {
    "mapping": "e1a1d6ada273c9e5162fc90f3e76d0f07fe490771e466b9c4f6b1788630cdbc1",
    "hashes": "3bf9d9360a279920bebf6efe18481aa63f38a5b50583ad5a98b47b8c2a22458b",
    "config": "f339cde89e89ad4309abca3248f8db191a04bd349b222828b8adcfeadf22f582",
    "loader": "329097169bb2772a951b3409952065b26e735fa5cd1f1c5783caf3b458417787",
    "categories": "73b026cc47de7d0d00072a68740236b7483cb4d0edb29264600f0d17d8771ab6",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def boolean(value):
    if value is True or str(value).lower() in ("true", "1"):
        return True
    if value is False or str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Unknown boolean {value!r}")


def number(value, integer=False, nullable=False):
    if value in (None, "", "NA", "None"):
        require(nullable, "Required number is missing")
        return None
    x = float(value)
    require(math.isfinite(x), "Non-finite number")
    require(not integer or x == int(x), "Expected integer")
    return int(x) if integer else x


def close(actual, expected, label, tolerance=1e-8):
    if expected is None:
        require(actual is None, f"{label}: unexpected non-null")
    else:
        require(actual is not None and math.isfinite(float(actual)) and abs(float(actual)-float(expected)) <= tolerance,
                f"{label}: {actual!r} != {expected!r}")


def key(row):
    return str(row["video_name"]), int(row["gt_index"])


def typed_csv(path, fields):
    if Path(path).stat().st_size == 0:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == [f["name"] for f in fields], f"CSV header disagrees with frozen schema: {path}")
        result = []
        for raw in reader:
            require(None not in raw, "Extra CSV cells")
            row = {}
            for field in fields:
                name, kind, nullable = field["name"], field["type"], field["nullable"]
                value = raw[name]
                if value == "NA":
                    require(nullable, f"Nonnullable {name} is NA")
                    row[name] = None
                elif kind == "str":
                    require(value not in (None, ""), f"Empty string {name}; expected explicit NA for missing")
                    row[name] = value
                elif kind == "bool":
                    require(value in ("true", "false"), f"Noncanonical bool {name}")
                    row[name] = value == "true"
                else:
                    require(kind in ("int", "float"), "Unsupported schema type")
                    row[name] = number(value, integer=kind == "int", nullable=False)
            result.append(row)
    return result


def read_metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for original in rows:
        row = dict(original)
        for name in ("gt_index", "subject_tid", "object_tid", "begin_fid", "end_fid"):
            row[name] = number(row[name], integer=True)
        for name in ("fps", "relation_sec", "coexist_sec"):
            row[name] = number(row[name], nullable=name == "coexist_sec")
        for name in ("native_F", "VrdONE_hit", "BIG_hit"):
            row[name] = boolean(row[name])
        for name in ("slices", "competition_flags"):
            row[name] = json.loads(row[name])
        require(key(row) not in result, "Duplicate local metadata GT")
        require(set(row["slices"]) == set(DEFINITIONS), "Metadata slice definitions changed")
        require(row["fps"] > 0 and row["end_fid"] > row["begin_fid"], "Invalid GT interval")
        close(row["relation_sec"], (row["end_fid"]-row["begin_fid"])/row["fps"], "metadata relation duration")
        expected = {(True, True):"A", (True, False):"B", (False, False):"C", (False, True):"D"}[(row["VrdONE_hit"], row["BIG_hit"])]
        require(row["original_group"] == expected, "ABCD metadata mismatch")
        result[key(row)] = row
    return result


def artifact_contract(export_dir, metadata_path, package_root, allow_partial):
    export_dir, package_root = Path(export_dir).resolve(), Path(package_root).resolve()
    manifest, status = load(export_dir/"manifest.json"), load(export_dir/"status.json")
    full = manifest.get("complete") is True
    expected_status = "COMPLETE_FULLVAL_CPU_EXPORT" if full else "COMPLETE_PARTIAL_SMOKE"
    require(full or allow_partial, "Partial export rejected; full 835-video export required")
    for obj in (manifest, status):
        require(obj.get("status") == expected_status and obj.get("complete") is full and obj.get("complete_selected_scope") is True,
                "Incomplete, failed, running, or inconsistent export status")
    require(manifest.get("model_forwards") == 0 and manifest.get("checkpoint_loaded") is False
            and manifest.get("cuda_initialized") is False and manifest.get("models_imported") is False,
            "Export is not the frozen CPU/no-model procedure")
    required = {"schema.json", "config.json", "input_receipt.json", "per_gt_temporal.csv", "proposal_binding.csv",
                "actual_pairs.csv", "saved_candidates.csv", "per_video_receipts.jsonl", "summary.json"}
    require(required <= set(manifest.get("outputs", {})), "Required independent verification artifact missing")
    for name, record in manifest["outputs"].items():
        require(Path(name).name == name and name not in (".", ".."), "Unsafe output filename")
        path = export_dir/name
        require(path.resolve().is_relative_to(export_dir) and path.is_file(), "Output path escaped export directory")
        require(sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], f"Output SHA/size mismatch: {name}")
    for filename in ("export_native.py", "native_geometry.py"):
        require(manifest["code"].get(filename) == sha(package_root/filename), f"Unexpected exporter code: {filename}")
    require(manifest.get("protocol",{}).get("sha256")==sha(package_root/"PROTOCOL.md")
            and manifest.get("protocol",{}).get("bytes")== (package_root/"PROTOCOL.md").stat().st_size,
            "Exporter protocol differs from trusted package")
    require(sha(export_dir/"schema.json") == sha(package_root/"schema.json"), "Schema differs from trusted package")
    schema = load(export_dir/"schema.json")
    require(schema["version"] == 1 and schema["missing_token"] == "NA", "Unsupported schema")
    receipt = load(export_dir/"input_receipt.json")
    require(receipt == load(package_root/"input_receipt.json"), "Input provenance receipt changed")
    require(receipt["metadata_source_sha256"] == sha(metadata_path), "Local 080 metadata SHA mismatch")
    require(receipt["metadata_csv_sha256"] == sha(package_root/"metadata.csv"), "Package metadata SHA mismatch")
    inventory = load(package_root/"inventory.json")
    require(receipt["inventory_sha256"] == sha(package_root/"inventory.json"), "Inventory SHA mismatch")
    require(len(inventory["video_names"]) == len(set(inventory["video_names"])) == receipt["N_inventory_videos"], "Frozen inventory duplicates/count mismatch")
    require(receipt["baseline_all_pass"] is True and receipt["maximum_score_diff"] == 0, "Historical baseline consistency absent")
    config = load(export_dir/"config.json")
    require(config["topks"] == list(TOPKS), "TopK protocol changed")
    require(config["mapping_rule"] == dict(mean_min=.5, coverage_min=.8, margin_min=.05), "Geometry mapping thresholds changed")
    require(config["loader_attributes"] == dict(with_clip_feature=False,feat_stride=4,proposal_min_frames=5,random_stride=False,stride_offset=0)
            and config["viou_threshold"] == "unmodified loader default", "Native input loader settings changed")
    require(config["relation_time_used_for_identity_or_prediction_selection"] is False
            and config["official_recall"] is False and config["inference_forward"] is False,
            "Readout protocol changed")
    videos = manifest["video_names"]
    require(videos == manifest["videos_completed"] == status["videos_completed"], "Video completion lists mismatch")
    require(len(videos) == len(set(videos)) == manifest["N_videos"] == status["N_videos"], "Duplicate/missing completed videos")
    expected_names = inventory["video_names"]
    limit = config.get("limit_videos")
    require(videos == (expected_names if full else expected_names[:limit]), "Completed videos differ from frozen inventory/prefix")
    require(full and limit is None or not full and isinstance(limit, int) and limit > 0, "Partial scope must be explicitly bounded")
    if full:
        require(len(videos) == receipt["N_inventory_videos"] == 835 and manifest["N_GT"] == status["N_GT"] == receipt["N_GT"] == 30142,
                "Fullval count contract failed")
        require(receipt["N_GT_bearing_videos"] == 832, "Fullval GT-bearing video contract failed")
    else:
        require(manifest["N_GT"] == status["N_GT"], "Partial GT counts inconsistent")
    expected_rows = {str(k): manifest["N_GT"] for k in TOPKS}
    require(manifest["N_rows_by_topk"] == status["N_rows_by_topk"] == expected_rows, "TopK declared counts mismatch")
    sources = manifest["source_files"]
    for name, expected in SOURCE_EXPECTED.items():
        require(sources[name]["sha256"] == expected, f"Historical source contract mismatch: {name}")
    for name, record in sources.items():
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None and int(record["bytes"]) >= 0, f"Invalid source digest {name}")
    require(sources["metadata"]["sha256"] == receipt["metadata_csv_sha256"], "Remote metadata provenance mismatch")
    require(sources["inventory"]["sha256"] == receipt["inventory_sha256"], "Remote inventory provenance mismatch")
    return manifest, schema, videos, full


def verify_export(export_dir, metadata_path=DEFAULT_METADATA, package_root=DEFAULT_PACKAGE, *, allow_partial=False):
    export_dir = Path(export_dir)
    manifest, schema, videos, full = artifact_contract(export_dir, metadata_path, package_root, allow_partial)
    all_metadata = read_metadata(metadata_path)
    receipt = load(export_dir/"input_receipt.json")
    require(len(all_metadata) == receipt["N_GT"], "Local metadata total differs from frozen receipt")
    require(len({k[0] for k in all_metadata}) == receipt["N_GT_bearing_videos"], "Local metadata video count mismatch")
    selected_videos=set(videos)
    metadata = {k:r for k,r in all_metadata.items() if k[0] in selected_videos}
    require(len(metadata) == manifest["N_GT"], "Export scope does not cover every local GT in completed videos")
    data = {name: typed_csv(export_dir/name, fields) for name, fields in schema["csv"].items()}
    with (export_dir/"per_video_receipts.jsonl").open() as handle:
        receipts = [json.loads(line) for line in handle if line.strip()]
    require([r["video_name"] for r in receipts] == videos, "Video receipts missing, duplicated, or out of inventory order")
    by_receipt = {r["video_name"]:r for r in receipts}
    meta_by_video = collections.defaultdict(list)
    for r in metadata.values():
        meta_by_video[r["video_name"]].append(r)
    gt_entities = {}
    for r in metadata.values():
        for role in ("subject", "object"):
            identity = (r["video_name"], r[role+"_tid"])
            if identity in gt_entities:
                require(gt_entities[identity] == r[role+"_class"], "GT entity category changes")
            gt_entities[identity] = r[role+"_class"]
    binding = {}
    binding_by_video=collections.defaultdict(list)
    for r in data["proposal_binding.csv"]:
        identity = (r["video_name"], r["proposal_id"])
        require(r["video_name"] in by_receipt and identity not in binding, "Duplicate/foreign proposal binding")
        require(r["proposal_id"] >= 0 and r["proposal_end_fid"] > r["proposal_begin_fid"], "Invalid proposal span/id")
        require(r["status"] in ("bound","no_class","not_input_eligible","low_coverage","low_geometry","ambiguous"), "Unknown binding status")
        if r["input_eligible"]:
            require(r["status"] != "not_input_eligible" and r["N_same_class"] is not None and r["N_same_class"] >= 0, "Eligible binding metadata missing")
            if r["N_same_class"] == 0:
                require(r["status"] == "no_class" and all(r[name] is None for name in ("best_tid","mean_iou","coverage","second_mean_iou","margin")), "No-class binding metadata inconsistent")
            else:
                require(all(r[name] is not None for name in ("best_tid","mean_iou","coverage","second_mean_iou","margin")), "Binding geometry metrics missing")
                require(-1e-8 <= r["second_mean_iou"] <= r["mean_iou"]+1e-8 and r["mean_iou"] <= r["coverage"]+1e-8
                        and -1e-8 <= r["coverage"] <= 1+1e-8, "Binding geometry metric bounds")
                close(r["margin"],r["mean_iou"]-r["second_mean_iou"],"binding margin")
                require(r["low_geometry"] == (r["mean_iou"] < .5) and r["low_coverage"] == (r["coverage"] < .8)
                        and r["ambiguous"] == (r["margin"] < .05), "Binding threshold flags inconsistent")
                expected_status = "low_coverage" if r["coverage"] < .8 else "low_geometry" if r["mean_iou"] < .5 else "ambiguous" if r["margin"] < .05 else "bound"
                require(r["status"] == expected_status, "Binding status does not follow frozen decision rule")
        if r["status"] == "bound":
            require(r["input_eligible"] and r["mapped_tid"] is not None and r["mapped_tid"] == r["best_tid"], "Bound identity inconsistent")
            require(r["mean_iou"] is not None and r["mean_iou"] >= .5 and r["coverage"] >= .8 and r["margin"] >= .05,
                    "Bound mapping below frozen thresholds")
            require(not r["low_geometry"] and not r["low_coverage"] and not r["ambiguous"], "Bound mapping has failure flag")
            if (r["video_name"], r["mapped_tid"]) in gt_entities:
                require(gt_entities[(r["video_name"], r["mapped_tid"])] == r["proposal_category"], "Bound entity class mismatch")
        else:
            require(r["mapped_tid"] is None, "Unbound proposal cannot carry a mapped identity")
        require(r["input_eligible"] or r["status"] == "not_input_eligible", "Ineligible proposal has binding status")
        binding[identity] = r
        binding_by_video[r["video_name"]].append(r)
    pairs = {}
    pair_sets = collections.defaultdict(set)
    eligible = collections.defaultdict(set)
    mapped_pairs = collections.defaultdict(set)
    mapped_subjects, mapped_objects = collections.defaultdict(set), collections.defaultdict(set)
    pair_indices_by_video=collections.defaultdict(list)
    for r in data["actual_pairs.csv"]:
        identity = (r["video_name"], r["pair_idx"])
        require(r["video_name"] in by_receipt and identity not in pairs and r["pair_idx"] >= 0, "Duplicate/foreign pair index")
        require(r["sid"] != r["oid"] and (r["sid"],r["oid"]) not in pair_sets[r["video_name"]], "Invalid/duplicate actual ordered pair")
        bs, bo = binding[(r["video_name"],r["sid"])], binding[(r["video_name"],r["oid"])]
        require(bs["input_eligible"] and bo["input_eligible"], "Actual pair contains ineligible proposal")
        pairs[identity] = r
        pair_indices_by_video[r["video_name"]].append(r["pair_idx"])
        pair_sets[r["video_name"]].add((r["sid"],r["oid"]))
        eligible[r["video_name"]].update((r["sid"],r["oid"]))
        if bs["status"] == "bound":
            mapped_subjects[r["video_name"]].add(bs["mapped_tid"])
        if bo["status"] == "bound":
            mapped_objects[r["video_name"]].add(bo["mapped_tid"])
        if bs["status"] == bo["status"] == "bound":
            mapped_pairs[r["video_name"]].add((bs["mapped_tid"], bo["mapped_tid"]))
    candidates = collections.defaultdict(list)
    candidate_ids = collections.defaultdict(set)
    for r in data["saved_candidates.csv"]:
        video, rank = r["video_name"], r["saved_rank_1based"]
        require(video in by_receipt and rank == len(candidates[video])+1 and rank <= 200, "Candidates must preserve contiguous saved rank")
        require(r["prediction_key"] == f"{video}:saved_rank:{rank}", "Prediction key disagrees with original rank")
        require(r["source_candidate_id"] >= 0 and r["source_candidate_id"] not in candidate_ids[video] and r["source_query"] >= 0,
                "Duplicate/invalid candidate or query ID")
        candidate_ids[video].add(r["source_candidate_id"])
        if candidates[video]:
            require(candidates[video][-1]["pred_score"] >= r["pred_score"], "Candidates not in saved descending score order")
        pair = pairs[(video,r["source_pair_idx"])]
        require((pair["sid"],pair["oid"]) == (r["source_sid"],r["source_oid"]), "Candidate pair differs from actual pair")
        require(r["selected_input_eligible"] is True and r["pred_end_fid"]-r["pred_begin_fid"] >= 5, "Invalid candidate eligibility/interval")
        for role, proposal_key in (("subject","source_sid"),("object","source_oid")):
            b = binding[(video,r[proposal_key])]
            require(b["input_eligible"] and b["proposal_category"] == r[role+"_class"], "Candidate class/eligibility mismatch")
            require(b["proposal_begin_fid"] <= r["pred_begin_fid"] < r["pred_end_fid"] <= b["proposal_end_fid"], "Candidate outside full proposal span")
            require(r[role+"_binding_status"] == b["status"] and r["mapped_"+role+"_tid"] == b["mapped_tid"], "Candidate binding fields mismatch")
        candidates[video].append(r)
    for video in videos:
        vr = by_receipt[video]
        bp = binding_by_video[video]
        pi = sorted(pair_indices_by_video[video])
        require(pi == list(range(len(pi))), "Actual pair indices not consecutive")
        require(sorted(r["proposal_id"] for r in bp) == list(range(len(bp))), "Proposal IDs not consecutive")
        require({r["proposal_id"] for r in bp if r["input_eligible"]} == eligible[video], "Eligibility must equal actual pair role union")
        require(vr["N_GT"] == len(meta_by_video[video]) and vr["N_proposals"] == len(bp)
                and vr["N_actual_input_pairs"] == len(pi) and vr["N_eligible_proposals"] == len(eligible[video]), "Per-video denominator receipt mismatch")
        require(vr["N_predictions"] == vr["N_selected_checked"] == len(candidates[video]), "Not all saved Top200 candidates exported")
        require(type(vr.get("empty_input")) is bool,"Per-video empty_input flag missing or not boolean")
        if vr["empty_input"]:
            # One-way implication only: a nonempty loader result may still have
            # no eligible pair. Do not equate 'no pair' with 'empty input'.
            require(not pi and not candidates[video],"Declared empty input contains actual pairs or saved candidates")
        require(vr["N_output_rows_by_topk"] == {str(k):len(meta_by_video[video]) for k in TOPKS}, "Per-video TopK row receipt mismatch")
        require(vr["binding_status_counts"] == dict(collections.Counter(r["status"] for r in bp)), "Binding status receipt mismatch")
        for source in ("cache", "predictions", "selected", "annotation"):
            require(vr["source_files"][source] == manifest["source_files"][source+":"+video], "Per-video source SHA receipt mismatch")
    choices = {}
    for video in videos:
        for k in TOPKS:
            d = {}
            for c in candidates[video][:k]:
                if c["subject_binding_status"] == c["object_binding_status"] == "bound":
                    identity = (c["mapped_subject_tid"],c["predicate"],c["mapped_object_tid"])
                    d.setdefault(identity,c)
            choices[(video,k)] = d
    rows_by_k = {k:{} for k in TOPKS}
    uses = collections.Counter()
    for r in data["per_gt_temporal.csv"]:
        identity, k = key(r), r["topk"]
        require(k in rows_by_k and identity in metadata and identity not in rows_by_k[k], "Duplicate/foreign GT or invalid TopK")
        m = metadata[identity]
        for name in METADATA_FIELDS:
            if name in ("fps", "relation_sec", "coexist_sec"):
                close(r[name],m[name],"metadata "+name,tolerance=1e-7)
            else:
                require(r[name] == m[name], f"GT metadata differs: {identity} {name}")
        video = identity[0]
        mapped = (r["subject_tid"],r["object_tid"]) in mapped_pairs[video]
        require(r["mapped_input_pair"] == mapped and r["subject_input_bound"] == (r["subject_tid"] in mapped_subjects[video])
                and r["object_input_bound"] == (r["object_tid"] in mapped_objects[video]), "Exported mapping does not match actual ordered pairs")
        require(r["geometry_binding_status"] == ("mapped_actual_pair" if mapped else "no_mapped_actual_pair"), "Binding status mismatch")
        require(r["readout"] == "saved_score_identity_predicate_diagnostic", "Unexpected temporal readout")
        selected = choices[(video,k)].get((r["subject_tid"],r["predicate"],r["object_tid"]))
        require(r["pred_available"] == (selected is not None) and r["no_prediction"] == (selected is None), "No-prediction boolean disagrees with independent saved-rank selection")
        if selected is None:
            require(all(r[name] is None for name in NULL_PRED_FIELDS) and r["tiou"] == 0 and r["same_prediction_GT_uses"] == 0,
                    "Missing prediction must retain null interval/ratio/disjoint and zero tIoU/uses")
        else:
            require(mapped and r["selected_input_eligible"] is True, "Selected prediction lacks mapped actual pair")
            for name in ("pred_begin_fid", "pred_end_fid", "saved_rank_1based", "prediction_key", "source_sid", "source_oid",
                         "source_pair_idx", "source_candidate_id", "source_query", "selected_input_eligible"):
                require(r[name] == selected[name], f"Selected prediction differs from independently reconstructed first saved rank: {name}")
            close(r["pred_score"], selected["pred_score"], "selected score")
            require(r["saved_rank_1based"] <= k, "Selected rank outside TopK")
            a,b,c,d = r["pred_begin_fid"],r["pred_end_fid"],r["begin_fid"],r["end_fid"]
            overlap = max(0,min(b,d)-max(a,c))
            close(r["tiou"], overlap/(b-a+d-c-overlap), "time IoU")
            close(r["duration_ratio"], (b-a)/(d-c), "duration ratio")
            close(r["pred_duration_sec"], (b-a)/r["fps"], "pred duration sec")
            require(r["completely_disjoint"] == (overlap == 0), "Disjoint flag mismatch")
            uses[(k,r["prediction_key"])] += 1
        rows_by_k[k][identity] = r
    for k in TOPKS:
        require(set(rows_by_k[k]) == set(metadata), f"Missing GT rows at Top{k}")
        require(len(rows_by_k[k]) == manifest["N_rows_by_topk"][str(k)], "TopK denominator mismatch")
        for r in rows_by_k[k].values():
            if r["pred_available"]:
                require(r["same_prediction_GT_uses"] == uses[(k,r["prediction_key"])], "Prediction reuse count mismatch")
    for identity in metadata:
        previous = None
        for k in TOPKS:
            r = rows_by_k[k][identity]
            if previous is not None:
                require(r["pred_available"] and all(r[field] == previous[field] for field in NULL_PRED_FIELDS), "Cross-K chosen prediction changed despite nested saved-rank selection")
            if r["pred_available"]:
                previous = r
    require(sum(vr["N_selected_checked"] for vr in receipts) == manifest["N_saved_prediction_checks"], "Total saved prediction check count mismatch")
    exported_summary=load(export_dir/"summary.json")
    require(exported_summary.get("official_recall") is False, "Exporter summary mislabeled as official recall")
    summary_keys=set()
    for summary in exported_summary["rows"]:
        k,pop=summary["topk"],summary["population"]
        require(k in TOPKS and pop in ("all_GT","mapped_actual_input_pair") and (k,pop) not in summary_keys,"Unexpected exporter summary row")
        summary_keys.add((k,pop))
        rs=[r for r in rows_by_k[k].values() if pop=="all_GT" or r["mapped_input_pair"]]
        pred=[r for r in rs if r["pred_available"]]
        n,m=len(rs),len(pred)
        require(summary["N"]==n and summary["N_pred"]==m and summary["no_prediction"]==n-m,"Exporter summary denominator mismatch")
        require(summary["N_videos"]==len({r["video_name"] for r in rs}),"Exporter summary video count mismatch")
        close(summary["tiou_sum"],sum(r["tiou"] for r in rs),"exported tIoU sum",tolerance=1e-6)
        close(summary["ratio_sum"],sum(r["duration_ratio"] for r in pred),"exported ratio sum",tolerance=1e-5)
        close(summary["mean_tiou"],sum(r["tiou"] for r in rs)/n if n else None,"exported mean tIoU")
        close(summary["duration_bias"],sum(r["duration_ratio"] for r in pred)/m if m else None,"exported duration bias")
        for tag,threshold in (("03",.3),("05",.5),("07",.7)):
            require(summary["miss_count_tiou"+tag]==sum(r["tiou"]<threshold for r in rs),"Exporter miss count mismatch")
    require(summary_keys=={(k,p) for k in TOPKS for p in ("all_GT","mapped_actual_input_pair")},"Missing exporter summary group")
    # Recheck the loaded snapshot before a caller can publish summaries.
    for name,record in manifest["outputs"].items():
        require(sha(export_dir/name)==record["sha256"],f"Input changed during verification: {name}")
    require(sha(metadata_path)==receipt["metadata_source_sha256"],"Local metadata changed during verification")
    return dict(full=full, manifest=manifest, metadata=metadata, all_metadata=all_metadata, rows_by_k=rows_by_k,
                binding_rows=data["proposal_binding.csv"], candidate_rows=data["saved_candidates.csv"], receipts=receipts,
                export_dir=export_dir, input_manifest_sha256=sha(export_dir/"manifest.json"))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:"NA" if v is None else v for k,v in r.items()})


def canonical(r, meta, k, definition, population, full):
    return dict(video_name=r["video_name"],gt_index=r["gt_index"],predicate=r["predicate"],relation_sec=r["relation_sec"],
                coexist_sec=r["coexist_sec"],slice=meta["slices"][definition],pred_available=r["pred_available"],
                tiou=r["tiou"],duration_ratio=r["duration_ratio"],source_scope="native_fullval_835" if full else "PARTIAL_NATIVE_CPU_SMOKE",
                readout=f"saved_score_identity_predicate_Top{k}_diagnostic",slice_definition=definition,analysis_population=population)


def summarize(verified, out, *, metadata_path, package_root):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    spec = importlib.util.spec_from_file_location("stats080",STATS)
    stats = importlib.util.module_from_spec(spec); spec.loader.exec_module(stats)
    summaries, audits, official = [], [], []
    for k in TOPKS:
        raw = list(verified["rows_by_k"][k].values())
        for definition in DEFINITIONS:
            expanded = []
            grouped = collections.defaultdict(list)
            for r in raw:
                m = verified["metadata"][key(r)]
                populations = ["all"]+(["mapped_actual_pair"] if r["mapped_input_pair"] else [])+(["native_F"] if r["native_F"] else [])
                for population in populations:
                    z = canonical(r,m,k,definition,population,verified["full"])
                    expanded.append(z)
                    grouped[(population,z["slice"])].append(r)
            current=stats.build_summary(expanded,include_all=True)
            for population in ("all","mapped_actual_pair","native_F"):
                if not any(r["analysis_population"]==population for r in current):
                    scope="native_fullval_835" if verified["full"] else "PARTIAL_NATIVE_CPU_SMOKE"
                    current.extend(stats._summarize([],scope,f"saved_score_identity_predicate_Top{k}_diagnostic",definition,population,slice_name)
                                   for slice_name in (*stats.SLICES,"ALL"))
            summaries.extend(current)
            for (population,slice_name), rs in grouped.items():
                n=len(rs);pred=[r for r in rs if r["pred_available"]];mapped=sum(r["mapped_input_pair"] for r in rs)
                disjoint=sum(r["completely_disjoint"] is True for r in pred)
                stretch_disjoint=sum(r["duration_ratio"]>=2 and r["completely_disjoint"] for r in pred)
                stretch_overlap=sum(r["duration_ratio"]>=2 and not r["completely_disjoint"] for r in pred)
                reuse=sum(r["same_prediction_GT_uses"]>1 for r in pred)
                a=dict(topk=k,slice_definition=definition,population=population,slice=slice_name,N=n,N_pred=len(pred),
                       no_prediction_numerator=n-len(pred),no_prediction_denominator=n,
                       no_prediction_rate=(n-len(pred))/n,mapped_pair_numerator=mapped,mapped_pair_denominator=n,mapped_pair_rate=mapped/n,
                       disjoint_numerator=disjoint,disjoint_denominator=len(pred),disjoint_rate=disjoint/len(pred) if pred else None,
                       disjoint_stretch_numerator=stretch_disjoint,disjoint_stretch_denominator=len(pred),disjoint_stretch_rate=stretch_disjoint/len(pred) if pred else None,
                       overlap_stretch_numerator=stretch_overlap,overlap_stretch_denominator=len(pred),overlap_stretch_rate=stretch_overlap/len(pred) if pred else None,
                       reused_prediction_GT_numerator=reuse,reused_prediction_GT_denominator=len(pred),reused_prediction_GT_rate=reuse/len(pred) if pred else None,
                       unique_selected_predictions=len({r['prediction_key'] for r in pred}))
                audits.append(a)
                if k==100:
                    for model in ("VrdONE","BIG"):
                        hits=sum(r[model+"_hit"] for r in rs)
                        official.append(dict(slice_definition=definition,population=population,slice=slice_name,model=model,
                                             hit_numerator=hits,hit_denominator=n,hit_rate=hits/n,miss_numerator=n-hits,miss_denominator=n,miss_rate=(n-hits)/n,
                                             metric="unchanged historical Top100 category+subject/object vIoU>=0.5 one-to-one hit; NOT newly evaluated R@100"))
    write_csv(out/"diag_summary.csv",summaries)
    write_csv(out/"readout_audit.csv",audits)
    write_csv(out/"official_historical_top100.csv",official)
    binding_counts=collections.Counter(r["status"] for r in verified["binding_rows"])
    write_csv(out/"binding_status_audit.csv",[dict(status=status,numerator=count,denominator=len(verified["binding_rows"]),rate=count/len(verified["binding_rows"])) for status,count in sorted(binding_counts.items())])
    scope="FULLVAL_VERIFIED" if verified["full"] else "PARTIAL_DIAGNOSTIC_ONLY"
    (out/"README.md").write_text(f"# 082导入结果：{scope}\n\n核验通过后生成；无模型执行、无训练。\n\n"
        "diag_summary.csv按四种切片定义、Top50/100/200及all/mapped_actual_pair/native_F分别统计。"
        "映射仅是全实体轨迹几何关联诊断，并非完美GT实体输入。选择不使用目标关系时间，一个预测可复用于多个GT事件，时间命中不是官方R@K。\n\n"
        "readout_audit.csv单列无预测、实际输入pair映射、完全错时段伸长和预测复用。official_historical_top100.csv只复制原命中标记的分组统计，不是新模型指标。\n")
    manifest=dict(complete=verified["full"],complete_selected_scope=True,status=scope,
                  N_videos=verified["manifest"]["N_videos"],N_GT=len(verified["metadata"]),N_rows_by_topk={str(k):len(verified["rows_by_k"][k]) for k in TOPKS},
                  training_or_model_execution=False,actual_pairs_and_saved_rank_selection_independently_recomputed=True,
                  output_metrics_are_official_recall=False,export_dir=str(verified["export_dir"].resolve()),
                  export_manifest_sha256=verified["input_manifest_sha256"],metadata_path=str(Path(metadata_path).resolve()),metadata_sha256=sha(metadata_path),
                  package_root=str(Path(package_root).resolve()),code={"import_native.py":sha(Path(__file__)),"stats080.py":sha(STATS)},
                  outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest


def import_export(export_dir, out, *, metadata_path=DEFAULT_METADATA, package_root=DEFAULT_PACKAGE, allow_partial=False):
    require(not Path(out).exists(),"Output already exists; refusing overwrite")
    verified=verify_export(export_dir,metadata_path,package_root,allow_partial=allow_partial)
    return summarize(verified,out,metadata_path=metadata_path,package_root=package_root)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--metadata",type=Path,default=DEFAULT_METADATA)
    p.add_argument("--package-root",type=Path,default=DEFAULT_PACKAGE)
    p.add_argument("--allow-partial-smoke",action="store_true")
    a=p.parse_args()
    receipt=import_export(a.export_dir,a.out,metadata_path=a.metadata,package_root=a.package_root,allow_partial=a.allow_partial_smoke)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
