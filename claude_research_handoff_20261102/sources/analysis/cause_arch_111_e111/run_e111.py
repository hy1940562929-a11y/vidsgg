"""E111 short-query bank: all arms Q=12; N no prior / T short prior / X long prior."""
import json, sys, time, random, hashlib, traceback, copy
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_runtime import NativeRuntime
import capacity_tools as cap
from e111_queries import expand_query_embed, SpanPrior, apply_prior_to_extra, forward_with_query_weight

R = Path("/root/autodl-tmp/VrdONE-data")
H = R / "commander_cause_arch_111"
CACHE = R / "commander_cache_import_094/cache_v4"
E100A1 = R / "commander_cause_arch_100/attempt_04_membership_expand"
OUT = H / "run_v1"
CONTRACT_SHA = "8d9736b4252b45a3d494e2b2cfafd97c877e3d5e35d5741599ba756698636f88"
EXTRA = 3
Q_TOTAL = 9 + EXTRA


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
    ce, mask = [], []
    num = o["pred_masks"].new_tensor(float(max(sum(len(x) for x in b["preds"]), 1)))
    for layer, (indices, lm) in zip(rt.layers(o), matches):
        args = (layer["pred_logits"], layer["pred_masks"], b["preds"], b["targets"], None, indices, num, lm)
        ce.append(rt.model.loss_labels(*args)["loss_class"])
        mask.append(sum(rt.model.loss_masks(*args).values()))
    return sum(ce) + sum(mask)


def arm_kind(arm):
    return {"N": "none", "T": "short", "X": "long"}[arm]


def forward_arm(rt, b, prior, arm):
    w = apply_prior_to_extra(rt.model.predictor, prior, arm_kind(arm))
    return forward_with_query_weight(rt.model.predictor, b, w)


def evaluate(rt, events, prior, arm, label):
    d = OUT / label
    d.mkdir(exist_ok=True)
    selected = {(e["video_name"], e["pair_idx"]) for e in events}
    sets = {}
    rt.model.predictor.eval()
    prior.eval()
    nq = rt.model.predictor.num_queries
    with torch.no_grad():
        for path in sorted((CACHE / "eval").glob("*.pt")):
            b = rt.load_batch(path)
            if arm == "C0":
                # C0 uses expanded bank without prior (same as N init path)
                o = forward_arm(rt, b, prior, "N")
            else:
                o = forward_arm(rt, b, prior, arm)
            last = rt.layers(o)[-1]
            meta = b["c"]["metadata"]
            pair = (meta["video_name"], int(meta["pair_idx"]))
            if pair not in selected:
                continue
            n = int(meta["N_valid"])
            origin = int(meta["origin"])
            masks = last["pred_masks"][0, :, :n].sigmoid()
            classes = last["pred_logits"][0].softmax(-1).argmax(-1).tolist()
            cs = []
            for q in range(nq):
                ix = torch.where(masks[q] > 0.5)[0]
                if len(ix) and int(ix[-1] - ix[0]) * 4 + 1 >= 5:
                    cs.append(dict(start=origin + 4 * int(ix[0]), end=origin + 4 * int(ix[-1]) + 1,
                                   predicate_id=classes[q], native_query=q,
                                   candidate_id=f"{label}/{pair[0]}/{pair[1]}/{q}"))
            sets[pair] = cs
    require(set(sets) == selected, "EVAL_FIXED_PAIR_UNIVERSE")
    result = cap.evaluate(sets, events)
    save(d / "evaluation.json", result)
    save(d / "candidates.json", [dict(video_name=v, pair_idx=p, candidates=cs) for (v, p), cs in sorted(sets.items())])
    return result


def admission(rt, events, prior):
    adm = OUT / "admission"
    adm.mkdir(parents=True, exist_ok=True)
    require(rt.model.predictor.num_queries == Q_TOTAL, "QUERY_EXPAND")
    paths = sorted((CACHE / "train").glob("*.pt"))
    # smoke forward
    diffs_tn, diffs_tx = [], []
    prior.eval()
    with torch.no_grad():
        for p in paths[:12]:
            b = rt.load_batch(p)
            o_n = forward_arm(rt, b, prior, "N")
            o_t = forward_arm(rt, b, prior, "T")
            o_x = forward_arm(rt, b, prior, "X")
            # zero-init prior ⇒ T≈N≈X at init
            diffs_tn.append(float((o_t["pred_masks"] - o_n["pred_masks"]).abs().mean()))
            diffs_tx.append(float((o_t["pred_masks"] - o_x["pred_masks"]).abs().mean()))
    def mean(xs):
        return float(sum(xs) / len(xs))
    exposure = dict(init_mask_L1_T_vs_N=mean(diffs_tn), init_mask_L1_T_vs_X=mean(diffs_tx),
                    num_queries=rt.model.predictor.num_queries, extra=EXTRA)
    require(exposure["init_mask_L1_T_vs_N"] < 1e-3, "PRIOR_NOT_NEAR_IDENTITY")
    # break zero-init symmetry for T vs X check by probing non-zero prior direction
    with torch.no_grad():
        # nonzero weight ⇒ short vs long scalar inputs yield different biases
        prior.proj.weight.fill_(0.05)
        b = rt.load_batch(paths[0])
        o_t = forward_arm(rt, b, prior, "T")
        o_x = forward_arm(rt, b, prior, "X")
        probe = float((o_t["pred_masks"] - o_x["pred_masks"]).abs().mean())
        prior.proj.weight.zero_()
        prior.proj.bias.zero_()
    require(probe > 1e-8, "PRIOR_TX_INACTIVE")
    exposure["probe_T_vs_X_after_bias_nudge"] = probe
    support = json.loads((E100A1 / "admission_v4/dev_support.json").read_text())
    common = [e for e in support["events"] if e.get("common_support")]
    cs = sum(1 for e in common if e["duration_seconds"] < 3)
    cl = sum(1 for e in common if e["duration_seconds"] >= 3)
    nv = len({e["video"] for e in common})
    require(nv >= 4 and cs >= 20 and cl >= 20, f"MECHANISM_UNIDENTIFIABLE {nv}/{cs}/{cl}")
    c0 = evaluate(rt, events, prior, "C0", "C0")
    # C0 with Q=12 may differ from classic 29/46 — record but require finite
    exposure["C0_residual334"] = c0["residual334"]["time_hits"]
    exposure["C0_main400"] = c0["main400"]["time_hits"]
    status = dict(status="ADMISSION_PASS", exposure=exposure,
                  support=dict(common_N_videos=nv, common_short=cs, common_long=cl, pass_gate=True),
                  contract_sha256=CONTRACT_SHA,
                  note="C0 uses Q=12 expanded bank; classic 29/46 was Q=9")
    save(adm / "status.json", status)
    return status


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
            CACHE, E100A1 / "decoder_inputs", "/root/rebinding_project/VrdONE",
            R / "commander_native_error_025/export_v1/effective_config.json",
            "/root/rebinding_project/VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth",
        )
        expand_query_embed(rt.model.predictor, extra=EXTRA)
        # re-collect trainable params after expand
        rt.model.predictor.requires_grad_(True)
        rt.named = list(rt.model.predictor.named_parameters())
        rt.params = [p for n, p in rt.named]
        prior = SpanPrior(n_embd=256, extra=EXTRA).to(rt.device)
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
        admission(rt, events, prior)
        initial_pred = copy.deepcopy(rt.model.predictor.state_dict())
        initial_prior = copy.deepcopy(prior.state_dict())
        pred_sha = signature(initial_pred)
        prior_sha = signature(initial_prior)
        save(OUT / "freeze.json", dict(contract_sha256=CONTRACT_SHA, predictor_sha256=pred_sha,
                                       prior_sha256=prior_sha, num_queries=Q_TOTAL, extra=EXTRA,
                                       seeds=[100, 101, 102], steps=500, takeover="Grace R7 after H110"))
        paths = sorted((CACHE / "train").glob("*.pt"))
        cpu_batches = {
            p.name: dict(
                c=torch.load(p, map_location="cpu", weights_only=False),
                inp=torch.load(E100A1 / "decoder_inputs" / p.name, map_location="cpu", weights_only=False),
            ) for p in paths
        }

        def batch(name):
            c = cpu_batches[name]["c"]; inp = cpu_batches[name]["inp"]; dev = rt.device
            return dict(c=c, x=inp["x"].to(dev), f=c["f"].to(dev), key_mask=inp["key_mask"].to(dev),
                        valid=c["valid"].to(dev), preds=[p.to(dev) for p in c["preds"]],
                        targets=[t.to(dev) for t in c["targets"]])

        for seed in [100, 101, 102]:
            order = []
            rng = random.Random(seed)
            while len(order) < 500:
                epoch = [p.name for p in paths]; rng.shuffle(epoch); order.extend(epoch)
            order = order[:500]
            save(OUT / f"batch_order_{seed}.json", order)
            for arm in ["N", "T", "X"]:
                label = f"{arm}_{seed}"
                phase = "TRAIN"
                rt.model.predictor.load_state_dict(initial_pred, strict=True)
                prior.load_state_dict(initial_prior, strict=True)
                require(signature(rt.model.predictor.state_dict()) == pred_sha, "INIT_DRIFT")
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                params = list(rt.model.predictor.parameters()) + list(prior.parameters())
                opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0)
                d = OUT / label; d.mkdir(exist_ok=True)
                history = []
                prior.train()
                for step, name in enumerate(order, 1):
                    b = batch(name)
                    opt.zero_grad(set_to_none=True)
                    o = forward_arm(rt, b, prior, arm)
                    loss = native_loss(rt, o, b)
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
                        save(OUT / "status.json", progress)
                        print(json.dumps(progress), flush=True)
                changed = sum(not torch.equal(initial_pred[k], v) for k, v in rt.model.predictor.state_dict().items())
                require(changed > 0, "INTERVENTION_INACTIVE")
                torch.save(dict(predictor={k: v.cpu() for k, v in rt.model.predictor.state_dict().items()},
                                prior={k: v.cpu() for k, v in prior.state_dict().items()},
                                arm=arm, seed=seed), d / "step500.pt")
                save(d / "training_history.json", history)
                phase = "EVALUATION"
                evaluate(rt, events, prior, arm, label)
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
