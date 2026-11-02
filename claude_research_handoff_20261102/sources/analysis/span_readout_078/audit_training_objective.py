"""Posthoc078 diagnostic: fixed-support initial/final losses, no training.

Loads only frozen feature caches, the already saved assignment audit and the
four completed checkpoints. It never creates predictions for new videos and
never modifies model/cache/checkpoint files. Output is a NEW JSON (--out).
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"

import torch
from torch.nn import functional as F


ARMS = ("independent", "joint", "joint_hard", "joint_random")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def import_heads(path):
    spec = importlib.util.spec_from_file_location("frozen_078_heads", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def inspect_scores(scores, first, last, invalid_score):
    n, t, _ = scores.shape
    flat = scores.reshape(n, t * t)
    target = first * t + last
    eligible = (first >= 0) & (first < last) & (last < t)
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if rows.numel():
        legal = flat[rows, target[rows]] > invalid_score / 2
        rows = rows[legal]
    eligible[:] = False
    eligible[rows] = True
    indices = flat.argmax(-1)
    start, end = indices // t, indices % t
    inter = (torch.minimum(end, last) - torch.maximum(start, first) + 1).clamp_min(0)
    union = (end - start + 1) + (last - first + 1) - inter
    iou = inter.float() / union.clamp_min(1)
    valid_prediction = (start < end) & (flat.amax(-1) > invalid_score / 2)
    iou = torch.where(valid_prediction, iou, 0)
    nll, margins = torch.full((n,), float("nan"), device=scores.device), torch.full((n,), float("nan"), device=scores.device)
    if rows.numel():
        selected = flat[rows]
        gt_score = selected.gather(1, target[rows, None]).squeeze(1)
        nll[rows] = F.cross_entropy(selected, target[rows], reduction="none")
        top = selected.topk(2, dim=1)
        best_other = torch.where(top.indices[:, 0] == target[rows], top.values[:, 1], top.values[:, 0])
        has_other = best_other > invalid_score / 2
        margins[rows[has_other]] = (gt_score - best_other)[has_other]
    return dict(eligible=eligible, first=start, last=end, iou=iou,
                hit=(iou >= .5) & valid_prediction, nll=nll, margin=margins,
                margin_eligible=torch.isfinite(margins))


def average(values):
    return sum(values) / len(values) if values else None


def reduce_role(rows, batch_rows, arms):
    eligible = [r for r in rows if r["eligible"]]
    result = dict(N_all=len(rows), N_eligible=len(eligible),
                  N_excluded=len(rows) - len(eligible),
                  N_singleton=sum(r["gt_first"] == r["gt_last"] for r in rows),
                  N_native_anchor_fallback=sum(r["native_anchor_fallback"] for r in rows),
                  N_native_anchor_fallback_eligible=sum(r["native_anchor_fallback"] for r in eligible),
                  arms={})
    for arm in arms:
        a = [r["arms"][arm] for r in eligible]
        batches = [b["arms"][arm] for b in batch_rows if b["N_eligible"]]
        result["arms"][arm] = dict(
            CE_event_weighted=average([x["CE"] for x in a]),
            CE_event_sum=sum(x["CE"] for x in a), CE_event_denominator=len(a),
            CE_batch_weighted=average([x["CE_mean"] for x in batches]),
            CE_batch_denominator=len(batches),
            decoded_mean_sample_tIoU=average([x["sample_tIoU"] for x in a]),
            decoded_tIoU50_hits=sum(x["hit_tIoU50"] for x in a),
            decoded_exact_endpoint_hits=sum(x["exact_gt_endpoints"] for x in a),
            gt_vs_best_other_score_margin_mean=average([x["gt_vs_best_other_score_margin"] for x in a if x["gt_vs_best_other_score_margin"] is not None]),
            gt_vs_best_other_score_margin_denominator=sum(x["gt_vs_best_other_score_margin"] is not None for x in a),
            added_vs_initial=sum(r["arms"][arm]["hit_tIoU50"] and not r["arms"]["initial_anchor_prior"]["hit_tIoU50"] for r in eligible),
            lost_vs_initial=sum(r["arms"]["initial_anchor_prior"]["hit_tIoU50"] and not r["arms"][arm]["hit_tIoU50"] for r in eligible),
        )
    base = result["arms"]["initial_anchor_prior"]
    for arm in arms[1:]:
        value = result["arms"][arm]
        value["delta_CE_event_weighted_vs_initial"] = value["CE_event_weighted"] - base["CE_event_weighted"]
        value["delta_CE_batch_weighted_vs_initial"] = value["CE_batch_weighted"] - base["CE_batch_weighted"]
        value["delta_mean_sample_tIoU_vs_initial"] = value["decoded_mean_sample_tIoU"] - base["decoded_mean_sample_tIoU"]
        value["net_tIoU50_hits_vs_initial"] = value["added_vs_initial"] - value["lost_vs_initial"]
        assert value["net_tIoU50_hits_vs_initial"] == value["decoded_tIoU50_hits"] - base["decoded_tIoU50_hits"]
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--code", required=True, type=Path, help="Frozen code_v1 containing heads.py and CACHE_INVENTORY.json")
    p.add_argument("--cache", type=Path, default=Path("/root/autodl-tmp/VrdONE-data/commander_mask_training_043/results_v1"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Refuse to overwrite existing diagnostic: " + str(args.out))
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError("Posthoc078 diagnostic exceeded 120 seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(78)
    mod = import_heads(args.code / "heads.py")
    sources = {str(args.code / "heads.py"): sha(args.code / "heads.py")}
    source_audit_path = args.results / "training_cache_audit.json"
    source_audit = load_json(source_audit_path)
    sources[str(source_audit_path)] = sha(source_audit_path)
    inventory = load_json(args.code / "CACHE_INVENTORY.json")
    sources[str(args.code / "CACHE_INVENTORY.json")] = sha(args.code / "CACHE_INVENTORY.json")
    matching = {(r["role"], r["batch"]): r["layer_matches"][-1] for r in source_audit["records"]}
    zero = mod.SpanHead(mode="joint").to(args.device).eval().requires_grad_(False)
    assert float(zero.gate) == 0 and torch.count_nonzero(zero.endpoint.weight) == 0
    assert torch.count_nonzero(zero.endpoint.bias) == 0
    heads = {"initial_anchor_prior": zero}
    gates = {}
    for arm in ARMS:
        cp = args.results / (arm + "_step500.pt")
        saved = torch.load(cp, map_location="cpu", weights_only=False)
        assert saved["arm"] == arm and saved["step"] == 500 and saved["seed"] == 78
        head = mod.SpanHead(mode="independent" if arm == "independent" else "joint")
        head.load_state_dict(saved["state"], strict=True)
        heads[arm] = head.to(args.device).eval().requires_grad_(False)
        gates[arm] = float(head.gate)
        sources[str(cp)] = sha(cp)
    all_rows, all_batches, summaries = [], [], {}
    for role, expected_all, expected_eligible in (("train", 563, 561), ("adapter_dev", 169, 168)):
        cache_path = args.cache / (role + "_cache.pt")
        cache_sha = sha(cache_path)
        assert cache_sha == inventory[role]["sha256"]
        sources[str(cache_path)] = cache_sha
        records = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert len(records) == inventory[role]["records"]
        rows, batches = [], []
        for ri, r in enumerate(records):
            match = matching[role, ri]
            b = torch.tensor(match["b"], device=args.device)
            q = torch.tensor(match["q"], device=args.device)
            first = torch.tensor(match["first"], device=args.device)
            last = torch.tensor(match["last"], device=args.device)
            _, _, fallback = zero.anchor(r, b, q)
            metrics = {}
            for arm, head in heads.items():
                scores = head.scores(r, b, q)
                metrics[arm] = inspect_scores(scores, first, last, mod.INVALID_SCORE)
                ce, count = mod.target_span_loss(scores, first, last)
                assert count == int(metrics[arm]["eligible"].sum())
                if count:
                    assert torch.allclose(ce, metrics[arm]["nll"][metrics[arm]["eligible"]].mean(), atol=1e-6, rtol=1e-6)
                del scores
            base = metrics["initial_anchor_prior"]
            for value in metrics.values():
                assert torch.equal(value["eligible"], base["eligible"])
            batches.append(dict(role=role, batch=ri, N_all=len(first), N_eligible=int(base["eligible"].sum()),
                                arms={a:dict(CE_mean=float(m["nll"][m["eligible"]].mean()) if m["eligible"].any() else None)
                                      for a, m in metrics.items()}))
            cpu = {a: {k: v.cpu().tolist() for k, v in m.items()} for a, m in metrics.items()}
            fallback = fallback.cpu().tolist()
            for k in range(len(first)):
                eligible = cpu["initial_anchor_prior"]["eligible"][k]
                f, e = match["first"][k], match["last"][k]
                arms = {}
                for arm, m in cpu.items():
                    arms[arm] = dict(decoded=[m["first"][k], m["last"][k]], sample_tIoU=m["iou"][k],
                                     hit_tIoU50=m["hit"][k], exact_gt_endpoints=m["first"][k] == f and m["last"][k] == e,
                                     CE=m["nll"][k] if eligible else None,
                                     gt_vs_best_other_score_margin=m["margin"][k] if m["margin_eligible"][k] else None)
                rows.append(dict(role=role, batch=ri, matched_index=k, pair_row=match["b"][k], query=match["q"][k],
                                 gt_first=f, gt_last=e, eligible=eligible, native_anchor_fallback=fallback[k], arms=arms))
            del metrics, cpu
        summaries[role] = reduce_role(rows, batches, tuple(heads))
        assert summaries[role]["N_all"] == expected_all
        assert summaries[role]["N_eligible"] == expected_eligible
        all_rows.extend(rows)
        all_batches.extend(batches)
        del records
        print(role, json.dumps(summaries[role], ensure_ascii=False), flush=True)
    result = dict(complete=True, posthoc=True, training_performed=False, checkpoint_modified=False,
                  scope="Same fixed final-layer matched training/dev queries; sampled tIoU, not official video-level recall or mAP",
                  initial_definition="Analytic native-anchor displacement prior with exactly zero endpoint residual and zero gate; fallback retained and flagged",
                  averaging="Event CE: sum eligible per-event CE / eligible events. Batch CE: arithmetic mean of per-batch eligible CE means. Singletons excluded in both.",
                  objective_scope="Recomputes shared span cross-entropy only; the extra hard/random contrast penalty is not included. No comparison of different batches' logged losses.",
                  support_reused="Saved training_cache_audit final-layer assignments; not rematched",
                  gates=gates, summaries=summaries, per_batch=all_batches, per_event=all_rows,
                  sources_sha256=sources, elapsed_seconds=time.monotonic() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    signal.alarm(0)
    print("POSTHOC078_COMPLETE", str(args.out), round(result["elapsed_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
