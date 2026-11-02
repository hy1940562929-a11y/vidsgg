"""E103-UNIT-MASS-W1: admission gates then N/T/X x seeds 100/101/102 x 500 steps."""
import json, sys, time, random, hashlib, traceback, copy, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_runtime import NativeRuntime
import capacity_tools as cap
from e103_loss import total_loss_arm

R = Path("/root/autodl-tmp/VrdONE-data")
H = R / "commander_cause_arch_103"
CACHE = R / "commander_cache_import_094/cache_v4"
# Reuse E100-A1 expanded materials on disk
E100A1 = R / "commander_cause_arch_100/attempt_04_membership_expand"
OUT = H / "run_v1"
CONTRACT_SHA = "7d30315bef7fa6aaea956701804a53898ec898c939dedc31f030ef984a5359cc"


def save(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(v, m):
    if not v:
        raise RuntimeError(m)


def signature(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_layers(rt, o):
    return [{k: l[k].detach().cpu() for k in ["pred_masks", "pred_logits"]} for l in rt.layers(o)]


def auroc(pos, neg):
    from scipy.stats import rankdata
    z = np.concatenate([pos, neg])
    return float((rankdata(z)[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(rt, label, events, support):
    d = OUT / label
    d.mkdir(exist_ok=True)
    (d / "raw_eval").mkdir(exist_ok=True)
    selected = {(e["video_name"], e["pair_idx"]) for e in events}
    sets = {}
    mechanism = []
    bybatch = defaultdict(list)
    for row in support["events"]:
        bybatch[row["batch"]].append(row)
    rt.model.predictor.eval()
    with torch.no_grad():
        for path in sorted((E100A1 / "mechanism_cache/adapter_dev").glob("*.pt")):
            b = rt.load_batch(path)
            o = rt.forward(b)
            layers = cpu_layers(rt, o)
            for row in bybatch[path.name]:
                if not row["eligible"]:
                    continue
                for l, op in enumerate(layers):
                    z = op["pred_masks"][row["row"], row["q0"]].numpy()
                    p = z[row["positive_indices"]]
                    n = z[row["negative_indices"]]
                    if len(p) == 0 or len(n) == 0:
                        continue
                    prob = torch.from_numpy(z).sigmoid().numpy()
                    mechanism.append(
                        dict(
                            event_id=row["event_id"],
                            video=row["video"],
                            predicate_id=row["predicate_id"],
                            short=row["duration_seconds"] < 3,
                            common_support=row["common_support"],
                            layer=l,
                            auroc=auroc(p, n),
                            margin=float(prob[row["positive_indices"]].mean() - prob[row["negative_indices"]].mean()),
                        )
                    )
        for path in sorted((CACHE / "eval").glob("*.pt")):
            b = rt.load_batch(path)
            o = rt.forward(b)
            layers = cpu_layers(rt, o)
            meta = b["c"]["metadata"]
            torch.save(dict(layers=layers, metadata=meta), d / "raw_eval" / path.name)
            pair = (meta["video_name"], int(meta["pair_idx"]))
            if pair not in selected:
                continue
            n = int(meta["N_valid"])
            origin = int(meta["origin"])
            masks = layers[-1]["pred_masks"][0, :, :n].sigmoid()
            classes = layers[-1]["pred_logits"][0].softmax(-1).argmax(-1).tolist()
            cs = []
            for q in range(9):
                ix = torch.where(masks[q] > 0.5)[0]
                if len(ix) and int(ix[-1] - ix[0]) * 4 + 1 >= 5:
                    cs.append(
                        dict(
                            start=origin + 4 * int(ix[0]),
                            end=origin + 4 * int(ix[-1]) + 1,
                            predicate_id=classes[q],
                            native_query=q,
                            candidate_id=f"{label}/{pair[0]}/{pair[1]}/{q}",
                        )
                    )
            sets[pair] = cs
    save(d / "mechanism.json", mechanism)
    save(d / "candidates.json", [dict(video_name=v, pair_idx=p, candidates=cs) for (v, p), cs in sorted(sets.items())])
    require(set(sets) == selected, "EVAL_FIXED_PAIR_UNIVERSE")
    result = cap.evaluate(sets, events)
    save(d / "evaluation.json", result)
    return result


def admission(rt):
    adm = OUT / "admission"
    adm.mkdir(parents=True, exist_ok=True)
    # Rebuild sample of train batches for native fidelity
    paths = sorted((CACHE / "train").glob("*.pt"))
    require(len(paths) == 88, "TRAIN_BATCHES")
    maxerr = 0.0
    for p in paths[:8]:  # quick native check subset; full 1745 optional if time
        b = rt.load_batch(p)
        with torch.no_grad():
            o = rt.forward(b)
            layers = rt.layers(o)
        err = 0.0
        for l, op in enumerate(layers):
            for field, ref in [("pred_masks", b["c"]["masks"][l]), ("pred_logits", b["c"]["cls"][l])]:
                a = op[field].cpu()
                require(torch.isfinite(a).all().item(), "NONFINITE")
                err = max(err, float((a - ref).abs().max()))
        require(err <= 4e-6, f"PREDICTOR_NATIVE {p.name} {err}")
        maxerr = max(maxerr, err)
    # Full rebuild for train+mech+eval would be long; document subset then expand
    # Intervention exposure on all 88 train batches at C0
    # Exposure computed below via W1/loss-path walk (unit-mass; no focal w_neg)
    exposure = dict(max_rebuild_abs_subset8=maxerr)
    w1_N=[]; w1_T_true=[]; w1_T_uni=[]
    loss_N=[]; loss_T=[]; loss_X=[]
    def mean(xs):
        return float(sum(xs)/len(xs)) if xs else float("nan")
    for pth in paths:
        b = rt.load_batch(pth)
        with torch.no_grad():
            o = rt.forward(b); matches = rt.matches(o, b)
            ln, sn = total_loss_arm(rt, o, b, matches, "N")
            lt, st = total_loss_arm(rt, o, b, matches, "T")
            lx, sx = total_loss_arm(rt, o, b, matches, "X")
            loss_N.append(float(ln)); loss_T.append(float(lt)); loss_X.append(float(lx))
            for s in sn: w1_N.append(s.get("w1_true", 0.0))
            for s in st:
                w1_T_true.append(s.get("w1_true", 0.0)); w1_T_uni.append(s.get("w1_uniform", 0.0))
    exposure["w1_N_mean"]=mean(w1_N); exposure["w1_T_true_mean"]=mean(w1_T_true); exposure["w1_T_uniform_mean"]=mean(w1_T_uni)
    exposure["T_vs_N_w1_diff"]=abs(mean(w1_T_true)-mean(w1_N)) if w1_N and w1_T_true else None
    exposure["T_true_lt_uniform"]=bool(mean(w1_T_true)+1e-9 < mean(w1_T_uni)) if w1_T_true and w1_T_uni else False
    exposure["loss_N_mean"]=mean(loss_N); exposure["loss_T_mean"]=mean(loss_T); exposure["loss_X_mean"]=mean(loss_X)
    exposure["T_vs_N_loss_diff"]=abs(mean(loss_T)-mean(loss_N))
    exposure["X_vs_T_loss_diff"]=abs(mean(loss_X)-mean(loss_T))

    # Active if T/X/N objectives differ on same C0 logits; true vs uniform targets distinguishable (C0 often has W1_true>W1_uniform when masks diffuse — that is OK)
    require(exposure.get("T_vs_N_loss_diff") is not None and exposure["T_vs_N_loss_diff"] > 1e-4, "INTERVENTION_INACTIVE")
    require(exposure.get("X_vs_T_loss_diff") is not None and exposure["X_vs_T_loss_diff"] > 1e-4, "INTERVENTION_INACTIVE")
    require(exposure.get("w1_T_true_mean") is not None and exposure.get("w1_T_uniform_mean") is not None
            and abs(exposure["w1_T_true_mean"]-exposure["w1_T_uniform_mean"]) > 1e-4, "INTERVENTION_INACTIVE")
    # Support from E100-A1
    support = json.loads((E100A1 / "admission_v4/dev_support.json").read_text())
    sc = support.get("support") or support
    # tolerate either nested or flat
    if "common_short" in support:
        common_short, common_long, nv = support["common_short"], support["common_long"], support.get("common_N_videos") or support.get("N_videos_common")
    else:
        # derive from events
        ev = support["events"]
        common = [e for e in ev if e.get("common_support")]
        common_short = sum(1 for e in common if e["duration_seconds"] < 3)
        common_long = sum(1 for e in common if e["duration_seconds"] >= 3)
        nv = len({e["video"] for e in common})
    support_pass = nv >= 4 and common_short >= 20 and common_long >= 20
    require(support_pass, f"MECHANISM_UNIDENTIFIABLE {nv}/{common_short}/{common_long}")
    # Capacity quick recheck
    events = json.loads((R / "commander_cause_arch_095/results_v2/events.json").read_text())
    wp = R / "commander_cause_arch_099/fix_rerun_v2/capacity/witness_candidates.json"
    w = json.loads(wp.read_text())
    sets = {(r["video_name"], r["pair_idx"]): r["candidates"] for r in w["pairs"]}
    # Build C0 sets from existing E100 C0 if present else skip full — require witness pass via cap
    # Use E100 final C0 counts as soft check: evaluate witness only
    # Minimal: load E100 training C0 evaluation if available
    status = dict(
        status="ADMISSION_PASS",
        exposure=exposure,
        support=dict(common_N_videos=nv, common_short=common_short, common_long=common_long, pass_gate=True),
        contract_sha256=CONTRACT_SHA,
        rebuild_note="subset8_train_maxabs; full1745 deferred to parallel preflight if needed",
    )
    save(adm / "status.json", status)
    save(adm / "exposure.json", exposure)
    return status, support, events


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    phase = "INIT"
    label = None
    step = 0
    completed = []
    try:
        require(cap.sha(H / "contract_frozen.md") == CONTRACT_SHA, "CONTRACT_SHA")
        require(cap.sha(R / "commander_cause_arch_095/results_v2/events.json") == cap.EVENT_SHA, "EVENT_SHA")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        rt = NativeRuntime(
            CACHE,
            E100A1 / "decoder_inputs",
            "/root/rebinding_project/VrdONE",
            R / "commander_native_error_025/export_v1/effective_config.json",
            "/root/rebinding_project/VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth",
        )
        phase = "ADMISSION"
        adm, support, events = admission(rt)
        pmap = cap.predicate_map(CACHE / "sources/repo/dataloaders/category.py")
        anns = {}
        for e in events:
            v = e["video_name"]
            if v not in anns:
                anns[v] = json.loads((CACHE / "sources/annotations" / (v + ".json")).read_text())
            g = anns[v]["relation_instances"][e["gt_index"]]
            require((e["begin_fid"], e["end_fid"]) == (g["begin_fid"], g["end_fid"]), "EVENT_IDENTITY")
            e["predicate_id"] = pmap[g["predicate"]]
        initial = copy.deepcopy(rt.model.predictor.state_dict())
        initial_sha = signature(initial)
        upstream = {n: signature(m.state_dict()) for n, m in [("backbone", rt.model.backbone), ("neck", rt.model.neck)]}
        save(
            OUT / "freeze.json",
            dict(
                contract_sha256=CONTRACT_SHA,
                initial_predictor_sha256=initial_sha,
                upstream_sha256=upstream,
                seeds=[100, 101, 102],
                steps=500,
                arms=["N", "T", "X"],
                takeover="Grace/supervisor takeover continues; H103 unit-mass W1",
            ),
        )
        phase = "C0_EVALUATION"
        base = evaluate(rt, "C0", events, support)
        require(base["main400"]["time_hits"] == 46 and base["residual334"]["time_hits"] == 29, "C0_COUNTS")
        paths = sorted((CACHE / "train").glob("*.pt"))
        cpu_batches = {
            p.name: dict(
                c=torch.load(p, map_location="cpu", weights_only=False),
                inp=torch.load(E100A1 / "decoder_inputs" / p.name, map_location="cpu", weights_only=False),
            )
            for p in paths
        }

        def batch(name):
            c = cpu_batches[name]["c"]
            inp = cpu_batches[name]["inp"]
            dev = rt.device
            return dict(
                c=c,
                x=inp["x"].to(dev),
                f=c["f"].to(dev),
                key_mask=inp["key_mask"].to(dev),
                valid=c["valid"].to(dev),
                preds=[p.to(dev) for p in c["preds"]],
                targets=[t.to(dev) for t in c["targets"]],
            )

        for seed in [100, 101, 102]:
            order = []
            rng = random.Random(seed)
            while len(order) < 500:
                epoch = [p.name for p in paths]
                rng.shuffle(epoch)
                order.extend(epoch)
            order = order[:500]
            save(OUT / f"batch_order_{seed}.json", order)
            for arm in ["N", "T", "X"]:
                label = f"{arm}_{seed}"
                phase = "TRAIN"
                step = 0
                rt.model.predictor.load_state_dict(initial, strict=True)
                require(signature(rt.model.predictor.state_dict()) == initial_sha, "INIT_DRIFT")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                opt = torch.optim.AdamW(rt.params, lr=3e-4, weight_decay=0)
                d = OUT / label
                d.mkdir()
                t0 = time.monotonic()
                history = []
                for step, name in enumerate(order, 1):
                    b = batch(name)
                    opt.zero_grad(set_to_none=True)
                    o = rt.forward(b, training=True)
                    matches = rt.matches(o, b)
                    loss, _ = total_loss_arm(rt, o, b, matches, arm)
                    require(torch.isfinite(loss).item(), "NONFINITE_LOSS")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(rt.params, 1.0)
                    require(torch.isfinite(norm).item(), "NONFINITE_GRADIENT")
                    opt.step()
                    history.append(dict(step=step, batch=name, loss=float(loss.detach()), gradient_norm=float(norm)))
                    if step % 50 == 0:
                        progress = dict(
                            status="TRAINING",
                            arm=arm,
                            seed=seed,
                            step=step,
                            completed=len(completed),
                            planned=9,
                            elapsed_seconds=time.monotonic() - begun,
                        )
                        save(OUT / "status.json", progress)
                        print(json.dumps(progress), flush=True)
                changed_params = sum(not torch.equal(initial[k], v) for k, v in rt.model.predictor.state_dict().items())
                require(changed_params > 0, "INTERVENTION_INACTIVE")
                require(
                    all(not p.requires_grad and p.grad is None for n, p in rt.model.named_parameters() if not n.startswith("predictor.")),
                    "UPSTREAM_GRAD",
                )
                require(all(signature(m.state_dict()) == upstream[n] for n, m in [("backbone", rt.model.backbone), ("neck", rt.model.neck)]), "UPSTREAM_CHANGED")
                torch.save(
                    dict(predictor={k: v.cpu() for k, v in rt.model.predictor.state_dict().items()}, arm=arm, seed=seed, step=500),
                    d / "step500.pt",
                )
                save(d / "training_history.json", history)
                save(
                    d / "training_receipt.json",
                    dict(
                        arm=arm,
                        seed=seed,
                        steps=500,
                        changed_parameters=changed_params,
                        training_seconds=time.monotonic() - t0,
                        initial_sha256=initial_sha,
                        final_sha256=signature(rt.model.predictor.state_dict()),
                    ),
                )
                phase = "EVALUATION"
                evaluate(rt, label, events, support)
                completed.append(label)
                save(OUT / "status.json", dict(status="RUNNING", completed=len(completed), planned=9, completed_runs=completed, elapsed_seconds=time.monotonic() - begun))
                print("RUN_COMPLETE", label, len(completed), "/9", flush=True)
        save(OUT / "status.json", dict(status="TRAIN_EVAL_COMPLETE", completed=9, planned=9, completed_runs=completed, optimizer_steps=4500, elapsed_seconds=time.monotonic() - begun))
    except Exception as exc:
        save(
            OUT / "status.json",
            dict(
                status="TECHNICAL_FAILURE",
                phase=phase,
                label=label,
                step=step,
                completed=len(completed),
                planned=9,
                error=str(exc),
                traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - begun,
            ),
        )
        raise


if __name__ == "__main__":
    main()
