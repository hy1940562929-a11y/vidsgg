"""E117 centerness-weighted mask loss: N native / T gaussian / X shuffled-cxw gaussian."""
import json, sys, time, random, hashlib, traceback, copy
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_runtime import NativeRuntime
import capacity_tools as cap
from e117_centerness import losses_arm, weight_disagreement

R = Path("/root/autodl-tmp/VrdONE-data")
H = R / "commander_cause_arch_117"
CACHE = R / "commander_cache_import_094/cache_v4"
E100A1 = R / "commander_cause_arch_100/attempt_04_membership_expand"
OUT = H / "run_v1"
CONTRACT_SHA = "26130c76f308753d55a639e3ce930d57c53084475ffcf33ec2e3494cb06645c1"


def save(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(v, m):
    if not v:
        raise RuntimeError(m)


def signature(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(rt, events, label):
    sets = {}
    selected = {(e["video_name"], e["pair_idx"]) for e in events}
    with torch.no_grad():
        for path in sorted((CACHE / "eval").glob("*.pt")):
            b = rt.load_batch(path)
            meta = b["c"]["metadata"]
            pair = (meta["video_name"], int(meta["pair_idx"]))
            if pair not in selected:
                continue
            rt.model.predictor.eval()
            o = rt.forward(b, training=False)
            last = rt.layers(o)[-1]
            n = int(meta["N_valid"]); origin = int(meta["origin"])
            masks = last["pred_masks"][0, :, :n].sigmoid()
            classes = last["pred_logits"][0].softmax(-1).argmax(-1).tolist()
            cs = []
            for q in range(9):
                ix = torch.where(masks[q] > 0.5)[0]
                if len(ix) and int(ix[-1] - ix[0]) * 4 + 1 >= 5:
                    cs.append(dict(start=origin + 4 * int(ix[0]), end=origin + 4 * int(ix[-1]) + 1,
                                   predicate_id=classes[q], native_query=q,
                                   candidate_id="%s/%s/%s/%s" % (label, pair[0], pair[1], q)))
            sets[pair] = cs
    d = OUT / label; d.mkdir(exist_ok=True)
    save(d / "candidates.json", [dict(video_name=v, pair_idx=p, candidates=cs) for (v, p), cs in sorted(sets.items())])
    result = cap.evaluate(sets, events)
    save(d / "evaluation.json", result)
    return result


def admission(rt, events):
    adm = OUT / "admission"; adm.mkdir(parents=True, exist_ok=True)
    paths = sorted((CACHE / "train").glob("*.pt"))
    require(len(paths) == 88, "TRAIN_BATCHES")
    maxerr = 0.0
    disagrees = []
    for p in paths[:8]:
        b = rt.load_batch(p)
        with torch.no_grad():
            o = rt.forward(b); layers = rt.layers(o)
        err = 0.0
        for l, op in enumerate(layers):
            for field, ref in [("pred_masks", b["c"]["masks"][l]), ("pred_logits", b["c"]["cls"][l])]:
                a = op[field].cpu(); require(torch.isfinite(a).all().item(), "NONFINITE")
                err = max(err, float((a - ref).abs().max()))
        require(err <= 4e-6, f"PREDICTOR_NATIVE {p.name} {err}")
        maxerr = max(maxerr, err)
        dmean, nshort = weight_disagreement(rt, o, b)
        if nshort:
            disagrees.append(dmean)
    support = json.loads((E100A1 / "admission_v4/dev_support.json").read_text())
    common = [e for e in support["events"] if e.get("common_support")]
    cs = sum(1 for e in common if e["duration_seconds"] < 3)
    cl = sum(1 for e in common if e["duration_seconds"] >= 3)
    nv = len({e["video"] for e in common})
    require(nv >= 4 and cs >= 20 and cl >= 20, f"MECHANISM_UNIDENTIFIABLE {nv}/{cs}/{cl}")
    c0 = evaluate(rt, events, "C0")
    require(c0["main400"]["time_hits"] == 46 and c0["residual334"]["time_hits"] == 29, "C0_COUNTS")
    mean_dis = sum(disagrees) / len(disagrees) if disagrees else 0.0
    require(mean_dis > 0 or not disagrees, "T_WEIGHT_INACTIVE")
    status = dict(status="ADMISSION_PASS",
                  exposure=dict(max_rebuild_abs_subset8=maxerr, centerness_disagree_T_vs_N=mean_dis),
                  support=dict(common_N_videos=nv, common_short=cs, common_long=cl, pass_gate=True),
                  contract_sha256=CONTRACT_SHA)
    save(adm / "status.json", status)
    return status


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic(); phase = "INIT"; completed = []
    try:
        require(cap.sha(H / "contract_frozen.md") == CONTRACT_SHA, "CONTRACT_SHA")
        require(cap.sha(R / "commander_cause_arch_095/results_v2/events.json") == cap.EVENT_SHA, "EVENT_SHA")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        rt = NativeRuntime(
            CACHE, E100A1 / "decoder_inputs", "/root/rebinding_project/VrdONE",
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
        admission(rt, events)
        initial_pred = copy.deepcopy(rt.model.predictor.state_dict())
        pred_sha = signature(initial_pred)
        save(OUT / "freeze.json", dict(contract_sha256=CONTRACT_SHA, predictor_sha256=pred_sha,
                                       seeds=[100, 101, 102], steps=500,
                                       takeover="Grace R15 H117 centerness after H114-SCALE"))
        paths = sorted((CACHE / "train").glob("*.pt"))
        for seed in [100, 101, 102]:
            order = []
            rng = random.Random(seed)
            while len(order) < 500:
                epoch = [p.name for p in paths]; rng.shuffle(epoch); order.extend(epoch)
            order = order[:500]
            save(OUT / f"batch_order_{seed}.json", order)
            for arm in ["N", "T", "X"]:
                label = f"{arm}_{seed}"; phase = "TRAIN"
                rt.model.predictor.load_state_dict(initial_pred, strict=True)
                require(signature(rt.model.predictor.state_dict()) == pred_sha, "INIT_DRIFT")
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                params = list(rt.params)
                opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0)
                d = OUT / label; d.mkdir(exist_ok=True)
                history = []; rt.model.predictor.train()
                for step, name in enumerate(order, 1):
                    b = rt.load_batch(CACHE / "train" / name)
                    opt.zero_grad(set_to_none=True)
                    o = rt.forward(b)
                    loss, matches, _ = losses_arm(rt, o, b, arm, seed_key=(seed * 1000003 + step))
                    require(torch.isfinite(loss).item(), "NONFINITE_LOSS")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    require(torch.isfinite(norm).item(), "NONFINITE_GRADIENT")
                    opt.step()
                    history.append(dict(step=step, batch=name, loss=float(loss.detach()), gradient_norm=float(norm)))
                    if step % 50 == 0:
                        progress = dict(status="TRAINING", arm=arm, seed=seed, step=step,
                                        completed=len(completed), planned=9,
                                        elapsed_seconds=time.monotonic() - begun)
                        save(OUT / "status.json", progress); print(json.dumps(progress), flush=True)
                changed = sum(not torch.equal(initial_pred[k], v) for k, v in rt.model.predictor.state_dict().items())
                require(changed > 0, "INTERVENTION_INACTIVE")
                torch.save(dict(predictor={k: v.cpu() for k, v in rt.model.predictor.state_dict().items()},
                                arm=arm, seed=seed), d / "step500.pt")
                save(d / "training_history.json", history)
                phase = "EVALUATION"
                evaluate(rt, events, label)
                completed.append(label)
                save(OUT / "status.json", dict(status="RUNNING", completed=len(completed), planned=9,
                                               completed_runs=completed, elapsed_seconds=time.monotonic() - begun))
                print("RUN_COMPLETE", label, len(completed), "/9", flush=True)
        save(OUT / "status.json", dict(status="TRAIN_EVAL_COMPLETE", completed=9, planned=9,
                                       completed_runs=completed, optimizer_steps=4500,
                                       elapsed_seconds=time.monotonic() - begun))
    except Exception as exc:
        save(OUT / "status.json", dict(status="TECHNICAL_FAILURE", phase=phase, completed=len(completed),
                                       error=str(exc), traceback=traceback.format_exc(),
                                       elapsed_seconds=time.monotonic() - begun))
        raise


if __name__ == "__main__":
    main()
