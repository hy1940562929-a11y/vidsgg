#!/usr/bin/env python3
"""079-C: CE versus CE + expected true-frame interval-IoU risk.

Prepare locally; execute only after transfer to an authorized private server.
No full-video inference is launched. The outputs are fixed matched-query
train/development localization diagnostics, never official detection metrics.
Run --help without torch. --self-test uses synthetic CPU tensors only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from loss_math import (advancement_gate, decoded_frame_interval, endpoint_metrics, interval_iou,
                       interval_objective, summarize_rows)


HERE = Path(__file__).resolve().parent
STEPS, SEED = 500, 79
MANIFEST_SHA = {
    "train": "39c0555bda3d99e928aef89a9f346a9b2c418a4e051205729a3838927a2306c5",
    "adapter_dev": "4a9dc55807723c69b2f8d10d73ad827ca3b9c9f2007f2e6607603f8a16db7edc",
}
ARMS = ("ce", "ce_iou")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


class Paths:
    def __init__(self, config, config_file):
        self.config = config
        self.base = Path(config_file).resolve().parent
        self.paths = config.get("paths", config)
        self.repo = self.get("repo_root")
        self.data = self.get("data_root")
        self.remaps = {
            "/root/rebinding_project/VrdONE": str(self.repo),
            "/root/autodl-tmp/VrdONE-data": str(self.data),
            **config.get("path_remap", {}),
        }

    def get(self, name, default=None):
        value = self.paths.get(name, default)
        if value is None:
            raise ValueError("config.paths." + name + " is required")
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base / p).resolve()

    def remap(self, value):
        p = Path(value)
        for old, new in sorted(self.remaps.items(), key=lambda v: -len(v[0])):
            try:
                suffix = p.relative_to(old)
            except ValueError:
                continue
            return Path(new) / suffix
        return p if p.is_absolute() else (self.base / p).resolve()


def verify_file(path, expected, sources):
    ensure(Path(path).is_file(), "Required source is missing: " + str(path))
    actual = digest(path)
    ensure(actual == expected, "Source SHA256 mismatch: " + str(path))
    sources[str(path)] = actual


def import_vendor(vendor, repo):
    # A standalone process prevents historical generic modules (heads/losses)
    # from silently resolving to a different experiment's implementations.
    sys.path.insert(0, str(vendor))
    sys.path.insert(1, str(repo))
    modules = {}
    for name in ("heads", "readout", "frozen043", "frozen049"):
        module = importlib.import_module(name)
        ensure(Path(module.__file__).resolve().parent == vendor.resolve(),
               "Unexpected imported module location for " + name)
        modules[name] = module
    return modules


def prepare_caches(torch, model, inventory, paths, cohorts, vendor, out, sources, device):
    """Hash-identical 043 caches, fresh fixed matching, uniform numerical gate."""
    cache_root = paths.get("cache_root")
    caches, audit, names = {}, [], {}
    for role in ("train", "adapter_dev"):
        path = cache_root / (role + "_cache.pt")
        verify_file(path, inventory[role]["sha256"], sources)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        ensure(len(raw) == inventory[role]["records"], "Cache record count changed")
        records, videos = [], set()
        for ri, old in enumerate(raw):
            record = {
                k: ([v.to(device).detach() for v in value] if k in ("preds", "targets")
                    else value.to(device).detach() if torch.is_tensor(value) else value)
                for k, value in old.items()
            }
            with torch.no_grad():
                projected = vendor["readout"].project(model.predictor.mask_embed, record)
                projected = projected.masked_fill(~record["valid"][None, ...], -10.)
                difference = (projected - record["masks"]).abs()
                ensure(bool(torch.isfinite(difference).all()), "Nonfinite cache projection")
                ensure(float(difference.max()) <= 4e-6,
                       f"Uniform projection tolerance exceeded: {role}/{ri}")
                binary_equal = torch.equal(projected.sigmoid() > .5,
                                           record["masks"].sigmoid() > .5)
                ensure(binary_equal, f"Native binary masks changed: {role}/{ri}")
                old_match = vendor["frozen049"].matched(model, record)
                new_match = vendor["frozen049"].matched(model, dict(record, masks=projected))
                matches_equal = all(
                    torch.equal(a["b"], b["b"]) and torch.equal(a["q"], b["q"]) and
                    len(a["indices"]) == len(b["indices"]) and
                    all(torch.equal(ai, bi) and torch.equal(aj, bj)
                        for (ai, aj), (bi, bj) in zip(a["indices"], b["indices"]))
                    for a, b in zip(old_match, new_match)
                )
                ensure(matches_equal, f"Hungarian assignments changed: {role}/{ri}")
                record["match"] = old_match
            records.append(record)
            videos.update(item["video_id"] for item in record["metadata"])
            audit.append(dict(role=role, batch=ri, N_pairs=len(record["targets"]),
                N_GT=sum(len(x) for x in record["targets"]),
                exact_projection=torch.equal(projected, record["masks"]),
                max_abs_projection_difference=float(difference.max()),
                binary_equal=binary_equal, all_layer_assignments_equal=matches_equal,
                matching_source="unchanged original043 cached native outputs"))
        ensure(sum(len(r["targets"]) for r in records) == inventory[role]["N_pairs"],
               "Cache pair count changed")
        ensure(sum(len(t) for r in records for t in r["targets"]) == inventory[role]["N_GT"],
               "Cache GT count changed")
        caches[role], names[role] = records, videos
        del raw
    validation = set()
    for cohort in cohorts.values():
        manifest = paths.remap(cohort["export"]) / "manifest.json"
        verify_file(manifest, cohort["manifest_sha256"], sources)
        validation.update(load_json(manifest)["video_names"])
    ensure(not names["train"] & names["adapter_dev"], "Training/development video overlap")
    ensure(not validation & (names["train"] | names["adapter_dev"]),
           "Training/development overlaps exposed validation cohorts")
    write_json(out / "split_audit.json", dict(
        cache_projection_absolute_tolerance=4e-6, role_specific_exceptions=False,
        train_video_names=sorted(names["train"]), adapter_dev_video_names=sorted(names["adapter_dev"]),
        exposed_validation_video_names=sorted(validation),
        roles_are_disjoint=True, validation_inference_run=False,
        development_status="Previously exposed adapter development split; not independent generalization"))
    write_json(out / "cache_audit.json", audit)
    return caches


def bind_annotations(torch, caches, paths, sources, out):
    """Bind every retained target to one original annotation before diagnostics.

    Source exports and annotations remain read-only, including censored pairs.
    Metadata derives from their actual FPS, frame origin and stride. No GT
    interval is substituted into a model prediction.
    """
    from dataloaders.category import vidor_pred_name_to_id
    export_root = paths.get("train_export_root")
    receipts = []
    for role, records in caches.items():
        manifest_path = export_root / role / "manifest.json"
        verify_file(manifest_path, MANIFEST_SHA[role], sources)
        manifest = load_json(manifest_path)
        ensure(len(records) == len(manifest["records"]), "Source export record count changed")
        annotations = {}
        for ri, (record, reference) in enumerate(zip(records, manifest["records"])):
            path = paths.remap(reference["path"])
            verify_file(path, reference["sha256"], sources)
            old = torch.load(path, map_location="cpu", weights_only=False)
            ensure(record["metadata"] == old["metadata"]["pair_metadata"], "Pair metadata changed")
            record["gt_metadata"] = []
            for b, pm in enumerate(record["metadata"]):
                ensure(torch.equal(record["preds"][b].cpu(), old["preds_list"][b]), "Predicate targets changed")
                ensure(torch.equal(record["targets"][b].cpu(), old["masks_list"][b]), "Mask targets changed")
                video = pm["video_id"]
                if video not in annotations:
                    ann_ref = manifest["inputs"][video]
                    ann_path = paths.remap(ann_ref["annotation"])
                    verify_file(ann_path, ann_ref["annotation_sha256"], sources)
                    annotations[video] = load_json(ann_path)
                ann = annotations[video]
                fps = float(ann["fps"])
                ensure(fps > 0, "Invalid annotated FPS")
                tids = sorted({int(v["tid"]) for frame in ann["trajectories"] for v in frame})
                sid, oid = (tids[pm["pair_key"][i]] for i in (0, 1))
                origin, stride = int(pm["sample_frame_origin"]), int(pm["feat_stride"])
                pair0, pair1 = pm["pair_frame_interval"]
                length = record["targets"][b].shape[-1]
                n_valid = int(record["valid"][b, 0].sum())
                ensure(bool(record["valid"][b, 0, :n_valid].all()) and
                       not bool(record["valid"][b, 0, n_valid:].any()),
                       "079-C requires prefix-contiguous native validity")
                ensure(stride > 0 and n_valid >= 2, "Unsupported temporal support")
                candidates, all_episodes = [], []
                for gi, gt in enumerate(ann["relation_instances"]):
                    if gt["subject_tid"] != sid or gt["object_tid"] != oid:
                        continue
                    first = max(0, math.ceil((gt["begin_fid"] - origin) / stride))
                    last = min(length, math.ceil((gt["end_fid"] - origin) / stride))
                    mask = torch.zeros(length, dtype=torch.bool, device=record["valid"].device)
                    if last > first:
                        mask[first:last] = True
                    entry = (gi, gt, vidor_pred_name_to_id[gt["predicate"]], mask)
                    all_episodes.append(entry)
                    if pair0 <= gt["begin_fid"] < gt["end_fid"] <= pair1:
                        candidates.append(entry)
                metadata = []
                for j, (pid, y) in enumerate(zip(record["preds"][b], record["targets"][b])):
                    found = [(gi, gt) for gi, gt, p, mask in candidates
                             if p == int(pid) and torch.equal(mask, y.bool())]
                    ensure(len(found) == 1,
                           f"Nonunique original annotation alignment: {role}/{ri}/{b}/{j}")
                    gi, gt = found[0]
                    union = torch.stack([mask for _, _, p, mask in all_episodes if p == int(pid)]).any(0)
                    kept = (record["targets"][b][record["preds"][b] == pid] > 0).any(0)
                    ensure(bool((kept <= union).all()), "Retained mask exceeds annotation union")
                    missing = int(((union & ~kept) & record["valid"][b, 0]).sum())
                    censored = int(pm["censored_positive_cells"]) > 0
                    ensure(censored or missing == 0, "Unexplained missing annotated positive cells")
                    metadata.append(dict(video_name=video, gt_index=gi, predicate=gt["predicate"],
                        subject_tid=sid, object_tid=oid, fps=fps, origin=origin, stride=stride,
                        N_valid=n_valid, original_interval=[int(gt["begin_fid"]), int(gt["end_fid"])],
                        relation_sec=(gt["end_fid"] - gt["begin_fid"]) / fps,
                        pair_support_sec=(pair1 - pair0) / fps, censored_pair=censored,
                        annotation_union_missing_retained_valid_cells=missing))
                record["gt_metadata"].append(metadata)
                receipts.append(dict(role=role, batch=ri, pair_row=b, annotations=metadata))
            del old
    write_json(out / "annotation_alignment.json.gz", receipts)


def selected_metadata(record):
    match = record["match"][-1]
    ids = [(b, int(j)) for b, (_, targets) in enumerate(match["indices"]) for j in targets.tolist()]
    ensure([b for b, _ in ids] == match["b"].tolist(), "Matching target order changed")
    metadata = [record["gt_metadata"][b][j] for b, j in ids]
    return match, ids, metadata


def objective(head, record, weight):
    match, _, meta = selected_metadata(record)
    scores = head.scores(record, match["b"], match["q"])
    loss, parts = interval_objective(scores, match["first"], match["last"],
        [v["origin"] for v in meta], [v["stride"] for v in meta],
        [v["original_interval"] for v in meta], risk_weight=weight)
    return loss, parts, scores


def decode_scores(scores):
    size = scores.shape[-1]
    flat = scores.flatten(1)
    index = flat.argmax(-1)
    valid = flat.amax(-1) > -5e8
    return (index // size).masked_fill(~valid, -1), (index % size).masked_fill(~valid, -1)


def snapshot(torch, heads, zero, caches, vendor, out, stage):
    rows, summaries = [], {}
    with torch.no_grad():
        for role, records in caches.items():
            role_rows, totals = [], {a: dict(ce=0., risk=0., eligible=0) for a in ARMS}
            for ri, record in enumerate(records):
                match, ids, meta = selected_metadata(record)
                b, q = match["b"], match["q"]
                native = vendor["frozen049"].masks_intervals(record["masks"][-1, b, q], record["valid"][b, 0])
                anchors = zero.anchor(record, b, q)
                zero_decoded = zero.decode(record, b, q)
                ensure(all(torch.equal(a, d) for a, d in zip(anchors[:2], zero_decoded)),
                       "Zero-update span head no longer decodes its native/fallback anchor")
                native_valid = native[0] < native[1]
                ensure(all(torch.equal(a[native_valid], d[native_valid])
                           for a, d in zip(native, zero_decoded)),
                       "Zero-update span head changes an already valid native interval")
                decoded = dict(native=native, zero_update=zero_decoded)
                for arm, head in heads.items():
                    _, parts, scores = objective(head, record, float(arm == "ce_iou"))
                    decoded[arm] = decode_scores(scores)
                    for key in ("ce", "risk"):
                        totals[arm][key] += float(parts[key]) * parts["eligible"]
                    totals[arm]["eligible"] += parts["eligible"]
                for k, (bb, j) in enumerate(ids):
                    first, last = int(match["first"][k]), int(match["last"][k])
                    item = meta[k]
                    row = dict(role=role, batch=ri, pair_row=bb, target_index=j,
                        query=int(q[k]), sampled_GT_interval=[first, last],
                        singleton_GT=first == last, loss_eligible=0 <= first < last < item["N_valid"],
                        zero_update_uses_fallback=bool(anchors[2][k]), **item, arms={})
                    for arm, (starts, ends) in decoded.items():
                        pred = decoded_frame_interval(int(starts[k]), int(ends[k]), item["origin"],
                                                      item["stride"], item["N_valid"])
                        row["arms"][arm] = dict(
                            sampled_interval=[int(starts[k]), int(ends[k])], frame_interval=pred,
                            true_frame_tIoU=interval_iou(pred, item["original_interval"]),
                            **endpoint_metrics(pred, item["original_interval"], item["fps"]))
                    role_rows.append(row)
            summaries[role] = dict(groups=summarize_rows(role_rows), objective={
                arm: dict(eligible_GT=t["eligible"], mean_CE=t["ce"] / t["eligible"] if t["eligible"] else None,
                          mean_expected_iou_risk=t["risk"] / t["eligible"] if t["eligible"] else None)
                for arm, t in totals.items()})
            summaries[role]["by_predicate"] = {
                predicate: summarize_rows([r for r in role_rows if r["predicate"] == predicate])
                for predicate in sorted({r["predicate"] for r in role_rows})}
            summaries[role]["long_input_pair_support_short"] = dict(
                definition="relation<3s and source pair_frame_interval>=10s; training crop may be shorter",
                groups=summarize_rows([r for r in role_rows if r["relation_sec"] < 3 and r["pair_support_sec"] >= 10]))
            rows.extend(role_rows)
    write_json(out / ("diagnostics_" + stage + ".json.gz"), rows)
    write_json(out / ("diagnostic_summary_" + stage + ".json"), summaries)
    # One stable row per GT/arm is convenient for follow-up audit without reruns.
    fields = ("role", "batch", "pair_row", "target_index", "video_name", "gt_index", "predicate",
              "query", "relation_sec", "singleton_GT", "loss_eligible", "arm", "true_frame_tIoU",
              "start_abs_frames", "end_abs_frames", "endpoint_abs_seconds", "endpoint_relative",
              "prediction_begin", "prediction_end", "GT_begin", "GT_end")
    with (out / ("per_gt_" + stage + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for arm, value in row["arms"].items():
                data = {key: row[key] for key in fields if key in row}
                data.update({key: value[key] for key in fields if key in value})
                pred = value["frame_interval"] or (None, None)
                data.update(arm=arm, prediction_begin=pred[0], prediction_end=pred[1],
                            GT_begin=row["original_interval"][0], GT_end=row["original_interval"][1])
                writer.writerow(data)
    return summaries


def run(config_file, output):
    start = time.monotonic()
    out = Path(output).expanduser().resolve()
    ensure(not out.exists(), "Output directory already exists; choose a fresh run directory")
    out.mkdir(parents=True)
    phase = "CONFIG"
    try:
        config = load_json(config_file)
        options = config.get("loss_probe", {})
        steps = int(options.get("steps", STEPS))
        seed = int(options.get("seed", SEED))
        learning_rate = float(options.get("learning_rate", 3e-4))
        lambda_iou = float(options.get("lambda_iou", 1.))
        wall_seconds = int(options.get("wall_seconds", 1800))
        ensure((steps, seed, learning_rate, lambda_iou) == (500, 79, 3e-4, 1.),
               "079-C fixes 500 steps, seed79, lr3e-4, lambda1. Revise the protocol before tuning these config values")
        ensure(wall_seconds > 0, "loss_probe.wall_seconds must be positive")
        def timeout_handler(*_):
            raise TimeoutError(f"079-C exceeded configured {wall_seconds}-second wall budget")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(wall_seconds)
        paths = Paths(config, config_file)
        vendor_path = paths.get("vendor078", str(HERE / "vendor078"))
        ensure(vendor_path.is_dir(), "Vendored078 directory is missing")
        files = config.get("files", {})
        def source_file(key, default):
            p = Path(files.get(key, str(vendor_path / default)))
            return p if p.is_absolute() else (Path(config_file).resolve().parent / p).resolve()
        contract_path = source_file("source_contract043", "source_contract043.json")
        inventory_path = source_file("cache_inventory", "CACHE_INVENTORY.json")
        cohorts_path = source_file("cohorts", "cohorts.json")
        contract, inventory, cohorts = map(load_json, (contract_path, inventory_path, cohorts_path))
        sources = {str(p): digest(p) for p in (Path(config_file).resolve(), contract_path, inventory_path,
                                              cohorts_path, HERE / "loss_probe.py", HERE / "loss_math.py")}
        for path in sorted(vendor_path.glob("*.py")):
            sources[str(path)] = digest(path)
        freeze = vendor_path / "runtime_manifest.json"
        ensure(freeze.is_file(), "079 vendored runtime manifest is missing")
        for filename, expected in load_json(freeze)["files"].items():
            verify_file(vendor_path / filename, expected, sources)
        sources[str(freeze)] = digest(freeze)
        phase = "SOURCE_CONTRACT"
        for relative, expected in contract["repo_sources"].items():
            verify_file(paths.repo / relative, expected, sources)
        checkpoint = paths.get("checkpoint")
        verify_file(checkpoint, contract["checkpoint_sha256"], sources)
        cfg_export = paths.remap(cohorts["047"]["export"])
        cfg_manifest = cfg_export / "manifest.json"
        verify_file(cfg_manifest, cohorts["047"]["manifest_sha256"], sources)
        manifest = load_json(cfg_manifest)
        cfgpath = cfg_export / "effective_config.json"
        verify_file(cfgpath, manifest["outputs"]["effective_config.json"]["sha256"], sources)
        cfg = load_json(cfgpath)
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        import numpy as np
        import torch
        device = str(config.get("device", "cuda:0"))
        ensure(device.startswith("cuda") and torch.cuda.is_available(),
               "Real079-C execution requires an available authorized CUDA device; use --self-test for CPU")
        torch.cuda.set_device(torch.device(device))
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        vendor = import_vendor(vendor_path, paths.repo)
        # Modules are only used for their read-only helpers. Bind their historical
        # locations explicitly; this workflow never calls their main() functions.
        for module in vendor.values():
            if hasattr(module, "REPO"):
                module.REPO = paths.repo
        from models.maskvrd import MaskVRD
        phase = "FROZEN_MODEL"
        model = MaskVRD(cfg["model_config"], device).eval()
        model._config_eval(cfg["inference_config"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)[contract["checkpoint_key"]]
        model.load_state_dict({key.removeprefix("module."): value for key, value in payload.items()}, strict=True)
        del payload
        model.to(device).requires_grad_(False)
        model_before = vendor["frozen043"].tree_sha(model.state_dict())
        phase = "CACHE_AND_ANNOTATION_AUDIT"
        caches = prepare_caches(torch, model, inventory, paths, cohorts, vendor, out, sources, device)
        bind_annotations(torch, caches, paths, sources, out)
        initial_head = vendor["heads"].SpanHead(mode="joint").to(device)
        zero = copy.deepcopy(initial_head).requires_grad_(False).eval()
        heads = {arm: copy.deepcopy(initial_head).train() for arm in ARMS}
        initial_hashes = {arm: vendor["frozen043"].tree_sha(head.state_dict()) for arm, head in heads.items()}
        ensure(len(set(initial_hashes.values())) == 1, "Intervention heads do not share initialization")
        optimizer = {arm: torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.05)
                     for arm, head in heads.items()}
        generator, order = np.random.default_rng(seed), []
        while len(order) < steps:
            order.extend(generator.permutation(len(caches["train"])).tolist())
        order = order[:steps]
        protocol = dict(experiment="079-C", seed=seed, steps=steps, learning_rate=learning_rate,
            weight_decay=.05, clip_norm=1., risk_weight=dict(ce=0., ce_iou=lambda_iou),
            wall_seconds=wall_seconds,
            risk_definition="Expected 1-tIoU over every legal interval in original frame coordinates",
            fixed_query_matching="Native original043 cached Hungarian assignment; diagnostics only",
            supervised_layers="final layer only", duration_weighting="none; event-mean within each cache batch",
            normalization_from043=False, hard_negatives=False, development_checkpoint_selection=False,
            inference_on_full128=False, schedule=order, initial_head_sha256=initial_hashes,
            trainable_parameters={a: sum(p.numel() for p in h.parameters()) for a, h in heads.items()},
            exclusions="Singleton/invalid target spans excluded from both losses; retained in total diagnostics",
            scope="Matched-query localization diagnostic; no mAP, official recall, or C-class recovery claim")
        write_json(out / "effective_protocol.json", protocol)
        write_json(out / "effective_config.json", config)
        write_json(out / "source_sha256_before.json", sources)
        torch.save(dict(state=initial_head.state_dict(), seed=seed, mode="joint"), out / "joint_initial.pt")
        phase = "INITIAL_DIAGNOSTICS"
        initial = snapshot(torch, heads, zero, caches, vendor, out, "initial")
        for role in initial:
            ensure(initial[role]["groups"]["all"]["arms"]["ce"] == initial[role]["groups"]["all"]["arms"]["ce_iou"],
                   "Identical heads have different initial diagnostics")
        history = []
        phase = "TRAIN_LOSS_ONLY"
        for step, batch in enumerate(order, 1):
            record = caches["train"][batch]
            for arm, head in heads.items():
                optimizer[arm].zero_grad(set_to_none=True)
                loss, parts, _ = objective(head, record, lambda_iou if arm == "ce_iou" else 0.)
                ensure(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                ensure(bool(torch.isfinite(norm)), "Nonfinite gradient")
                optimizer[arm].step()
                history.append(dict(step=step, batch=batch, arm=arm, loss=float(loss.detach()),
                    CE=float(parts["ce"].detach()), expected_iou_risk=float(parts["risk"].detach()),
                    eligible=parts["eligible"], gradient_norm_before_clip=float(norm)))
            if step % 50 == 0:
                write_json(out / "training_history.json", history)
                print("LOSS079", step, "elapsed", round(time.monotonic()-start, 1),
                      {v["arm"]: round(v["loss"], 5) for v in history[-2:]}, flush=True)
        for arm, head in heads.items():
            torch.save(dict(state=head.state_dict(), seed=seed, arm=arm, steps=steps,
                            mode="joint", protocol=protocol), out / (arm + f"_step{steps}.pt"))
        phase = "FINAL_DIAGNOSTICS"
        final = snapshot(torch, heads, zero, caches, vendor, out, "final")
        ensure(vendor["frozen043"].tree_sha(model.state_dict()) == model_before, "Frozen model changed")
        ensure(digest(checkpoint) == contract["checkpoint_sha256"], "Original checkpoint changed")
        # Rehash every measured input, including source caches and annotations.
        for path, expected in sources.items():
            ensure(digest(path) == expected, "Measured input changed during experiment: " + path)
        dev = final["adapter_dev"]["groups"]
        short = dev["short"]["arms"]["ce_iou"]["comparisons"]["ce"]
        all_change = dev["all"]["arms"]["ce_iou"]["comparisons"]["ce"]
        long_change = dev["long"]["arms"]["ce_iou"]["comparisons"]["ce"]
        train_short = final["train"]["groups"]["short"]["arms"]["ce_iou"]["comparisons"]["ce"]
        # A single-seed, exposed-development gate is a triage result, not a
        # significance test. It never launches a follow-on experiment.
        gate = advancement_gate(final)
        decision = gate["decision"]
        write_json(out / "gate_decision.json", gate)
        gradient_audit = {}
        for arm in ARMS:
            norms = [x["gradient_norm_before_clip"] for x in history if x["arm"] == arm]
            gradient_audit[arm] = dict(steps=len(norms), clipped_steps=sum(v > 1. for v in norms),
                first_gradient_norm=norms[0], mean_gradient_norm=sum(norms)/len(norms),
                max_gradient_norm=max(norms))
        write_json(out / "gradient_audit.json", gradient_audit)
        result = dict(complete=True, decision=decision, experiment="079-C", initial=initial, final=final,
            gate=gate,
            gradient_audit=gradient_audit,
            elapsed_seconds=time.monotonic()-start, frozen_model_state_unchanged=True,
            original_checkpoint_unchanged=True, measured_inputs_unchanged=True,
            native_fallback_warning="zero_update includes deterministic fallback for invalid native hulls; compare against both anchors",
            C_class_recovery=None, official_mAP=None, official_recall=None, full128_inference_run=False,
            inference_selection_uses_GT_matching=True,
            scope="Diagnostic only: fixed GT-matched queries on retained train/adapter_dev caches",
            source_sha256=sources, independent_review="PENDING")
        write_json(out / "summary.json", result)
        write_json(out / "status.json", dict(complete=True, decision=decision))
        print("COMPLETE LOSS079", decision, "short CE+IoU vs CE", short,
              "all CE+IoU vs CE", all_change, flush=True)
    except Exception as error:
        write_json(out / "status.json", dict(complete=False, phase=phase, error=repr(error),
                                             elapsed_seconds=time.monotonic()-start))
        raise
    finally:
        signal.alarm(0)


def self_test():
    import unittest
    try:
        import torch  # Explicitly required here, never during --help.
    except ImportError as error:
        raise SystemExit("SKIP/NOT VERIFIED: torch unavailable; synthetic gradient gate did not run. "
                         "Run --self-test in the existing server PyTorch environment.") from error
    print("079-C synthetic CPU tests; no real dataset, training run or evidence", flush=True)
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_loss_math.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON containing explicit private source paths")
    parser.add_argument("--out", help="Fresh private output directory; existing directories are rejected")
    parser.add_argument("--self-test", action="store_true", help="Synthetic CPU math/gradient tests only; requires torch")
    args = parser.parse_args()
    if args.self_test:
        ensure(not args.config and not args.out, "--self-test must not be combined with a real run")
        self_test()
    else:
        if not args.config or not args.out:
            parser.error("--config and --out are required unless --self-test is selected")
        run(args.config, args.out)


if __name__ == "__main__":
    main()
