"""E105: train native N x 3 seeds; eval hull (N) / soft quantile (T) / reverse-soft (X)."""
import json, sys, time, random, hashlib, traceback, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_runtime import NativeRuntime
import capacity_tools as cap
from e105_decode import native_hull_candidates, soft_quantile_candidates

R = Path("/root/autodl-tmp/VrdONE-data")
H = R / "commander_cause_arch_105"
CACHE = R / "commander_cache_import_094/cache_v4"
E100A1 = R / "commander_cause_arch_100/attempt_04_membership_expand"
OUT = H / "run_v1"
CONTRACT_SHA = "14c83eb5e1248158abcd41e62bb33ecb08e4136836cd2a707c999941478f630b"


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


def native_loss(rt, o, b):
    matches = rt.matches(o, b)
    ce = []
    mask = []
    num = o["pred_masks"].new_tensor(float(max(sum(len(x) for x in b["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(o), matches):
        args = (layer["pred_logits"], layer["pred_masks"], b["preds"], b["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        mask.append(sum(rt.model.loss_masks(*args).values()))
    return sum(ce) + sum(mask)


def decode_sets(rt, events, decode="hull", label="C0"):
    selected = {(e["video_name"], e["pair_idx"]) for e in events}
    sets = {}
    with torch.no_grad():
        for path in sorted((CACHE / "eval").glob("*.pt")):
            b = rt.load_batch(path)
            o = rt.forward(b)
            last = rt.layers(o)[-1]
            meta = b["c"]["metadata"]
            pair = (meta["video_name"], int(meta["pair_idx"]))
            if pair not in selected:
                continue
            n = int(meta["N_valid"])
            origin = int(meta["origin"])
            logits = last["pred_masks"][0]
            classes = last["pred_logits"][0].softmax(-1).argmax(-1).tolist()
            pref = f"{label}/{pair[0]}/{pair[1]}"
            if decode == "hull":
                cs = native_hull_candidates(logits.sigmoid(), classes, n, origin, pref)
            elif decode == "soft":
                cs = soft_quantile_candidates(logits, classes, n, origin, pref, reverse=False)
            elif decode == "revsoft":
                cs = soft_quantile_candidates(logits, classes, n, origin, pref, reverse=True)
            else:
                raise RuntimeError(decode)
            sets[pair] = cs
    require(set(sets) == selected, "EVAL_FIXED_PAIR_UNIVERSE")
    return sets


def evaluate_decode(rt, events, decode, label):
    d = OUT / label
    d.mkdir(exist_ok=True)
    sets = decode_sets(rt, events, decode=decode, label=label)
    save(d / "candidates.json", [dict(video_name=v, pair_idx=p, candidates=cs) for (v, p), cs in sorted(sets.items())])
    result = cap.evaluate(sets, events)
    save(d / "evaluation.json", result)
    return result


def admission(rt, events):
    adm = OUT / "admission"
    adm.mkdir(parents=True, exist_ok=True)
    paths = sorted((CACHE / "train").glob("*.pt"))
    require(len(paths) == 88, "TRAIN_BATCHES")
    maxerr = 0.0
    for p in paths[:8]:
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
    # C0 decode divergence on residual-relevant eval pairs sample
    base_h = evaluate_decode(rt, events, "hull", "C0_hull")
    base_s = evaluate_decode(rt, events, "soft", "C0_soft")
    base_r = evaluate_decode(rt, events, "revsoft", "C0_revsoft")
    require(base_h["main400"]["time_hits"] == 46 and base_h["residual334"]["time_hits"] == 29, "C0_HULL_COUNTS")
    diff_hs = abs(base_s["residual334"]["time_hits"] - base_h["residual334"]["time_hits"])
    diff_sr = abs(base_s["residual334"]["time_hits"] - base_r["residual334"]["time_hits"])
    exposure = dict(
        max_rebuild_abs_subset8=maxerr,
        C0_hull_334=base_h["residual334"]["time_hits"],
        C0_soft_334=base_s["residual334"]["time_hits"],
        C0_revsoft_334=base_r["residual334"]["time_hits"],
        hull_soft_334_absdiff=diff_hs,
        soft_rev_334_absdiff=diff_sr,
    )
    require(diff_hs > 0 or diff_sr > 0, "INTERVENTION_INACTIVE")
    # support from E100-A1
    support = json.loads((E100A1 / "admission_v4/dev_support.json").read_text())
    ev = support["events"]
    common = [e for e in ev if e.get("common_support")]
    common_short = sum(1 for e in common if e["duration_seconds"] < 3)
    common_long = sum(1 for e in common if e["duration_seconds"] >= 3)
    nv = len({e["video"] for e in common})
    require(nv >= 4 and common_short >= 20 and common_long >= 20, f"MECHANISM_UNIDENTIFIABLE {nv}/{common_short}/{common_long}")
    status = dict(
        status="ADMISSION_PASS",
        exposure=exposure,
        support=dict(common_N_videos=nv, common_short=common_short, common_long=common_long, pass_gate=True),
        contract_sha256=CONTRACT_SHA,
    )
    save(adm / "status.json", status)
    return status, support, base_h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    phase = "INIT"
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
        events = json.loads((R / "commander_cause_arch_095/results_v2/events.json").read_text())
        pmap = cap.predicate_map(CACHE / "sources/repo/dataloaders/category.py")
        anns = {}
        for e in events:
            v = e["video_name"]
            if v not in anns:
                anns[v] = json.loads((CACHE / "sources/annotations" / (v + ".json")).read_text())
            g = anns[v]["relation_instances"][e["gt_index"]]
            require((e["begin_fid"], e["end_fid"]) == (g["begin_fid"], g["end_fid"]), "EVENT_IDENTITY")
            e["predicate_id"] = pmap[g["predicate"]]
        phase = "ADMISSION"
        adm, support, c0hull = admission(rt, events)
        initial = copy.deepcopy(rt.model.predictor.state_dict())
        initial_sha = signature(initial)
        upstream = {n: signature(m.state_dict()) for n, m in [("backbone", rt.model.backbone), ("neck", rt.model.neck)]}
        save(OUT / "freeze.json", dict(contract_sha256=CONTRACT_SHA, initial_predictor_sha256=initial_sha, upstream_sha256=upstream, seeds=[100, 101, 102], steps=500, note="shared weights; decode arms N/T/X", takeover="Grace after residual synthesis"))
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
            phase = "TRAIN"
            rt.model.predictor.load_state_dict(initial, strict=True)
            require(signature(rt.model.predictor.state_dict()) == initial_sha, "INIT_DRIFT")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            opt = torch.optim.AdamW(rt.params, lr=3e-4, weight_decay=0)
            d = OUT / f"train_{seed}"
            d.mkdir(exist_ok=True)
            history = []
            for step, name in enumerate(order, 1):
                b = batch(name)
                opt.zero_grad(set_to_none=True)
                o = rt.forward(b, training=True)
                loss = native_loss(rt, o, b)
                require(torch.isfinite(loss).item(), "NONFINITE_LOSS")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(rt.params, 1.0)
                require(torch.isfinite(norm).item(), "NONFINITE_GRADIENT")
                opt.step()
                history.append(dict(step=step, batch=name, loss=float(loss.detach()), gradient_norm=float(norm)))
                if step % 50 == 0:
                    progress = dict(status="TRAINING", seed=seed, step=step, completed=len(completed), planned=3, elapsed_seconds=time.monotonic() - begun)
                    save(OUT / "status.json", progress)
                    print(json.dumps(progress), flush=True)
            changed = sum(not torch.equal(initial[k], v) for k, v in rt.model.predictor.state_dict().items())
            require(changed > 0, "INTERVENTION_INACTIVE")
            torch.save(dict(predictor={k: v.cpu() for k, v in rt.model.predictor.state_dict().items()}, seed=seed, step=500), d / "step500.pt")
            save(d / "training_history.json", history)
            # Eval three decodes on this checkpoint
            phase = "EVALUATION"
            for decode, tag in [("hull", "N"), ("soft", "T"), ("revsoft", "X")]:
                label = f"{tag}_{seed}"
                evaluate_decode(rt, events, decode, label)
                completed.append(label)
            save(OUT / "status.json", dict(status="RUNNING", completed_trains=seed, eval_labels=completed, planned_eval=9, elapsed_seconds=time.monotonic() - begun))
            print("SEED_COMPLETE", seed, flush=True)
        save(OUT / "status.json", dict(status="TRAIN_EVAL_COMPLETE", completed_trains=3, completed_evals=len(completed), planned_eval=9, optimizer_steps=1500, elapsed_seconds=time.monotonic() - begun, note="3 native trains x 3 decode evals"))
    except Exception as exc:
        save(OUT / "status.json", dict(status="TECHNICAL_FAILURE", phase=phase, error=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - begun))
        raise


if __name__ == "__main__":
    main()
