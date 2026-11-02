"""E099 native gradient/virtual-step measurements. No candidate training trajectory.

--mode support is CPU/stdlib only. --mode state measures exactly one fixed
state (C0 or an O250/O500 snapshot); it never selects a checkpoint.
"""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def support_audit(cache, out=None):
    source = Path(cache) / "event_support.csv"
    rows = list(csv.DictReader(source.open()))
    dev = [r for r in rows if r["split"] == "adapter_dev"]
    require(len(dev) == 169 and len({r["event_id"] for r in dev}) == 169, "DEV_SUPPORT_IDENTITY")
    by_vp = defaultdict(list)
    for r in dev:
        by_vp[(r["video_id"], int(r["predicate_id"]))].append(r)
    common = {k: v for k, v in by_vp.items()
              if any(float(r["duration_seconds"]) < 3 for r in v)
              and any(float(r["duration_seconds"]) >= 3 for r in v)}
    short = [r for v in common.values() for r in v if float(r["duration_seconds"]) < 3]
    long = [r for v in common.values() for r in v if float(r["duration_seconds"]) >= 3]
    nvideos = len({v for v, p in common})
    result = dict(
        source=str(source), sha256=sha(source), N_dev=169,
        N_videos=len({r["video_id"] for r in dev}), common_N_videos=nvideos,
        common_short=len(short), common_long=len(long),
        support_pass=nvideos >= 4 and len(short) >= 20 and len(long) >= 20,
        common_groups=[dict(video=v, predicate=p, short=[r["event_id"] for r in rs if float(r["duration_seconds"]) < 3],
                            long=[r["event_id"] for r in rs if float(r["duration_seconds"]) >= 3])
                       for (v, p), rs in sorted(common.items())],
        all_dev_events=dev,
        interpretation="support only; REJECT requires technical and joint-capacity gates to pass first")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        write(Path(out)/"support_audit.json", result)
    return result


def dot64(a, b):
    import torch
    return sum((x.double()*y.double()).sum() for x, y in zip(a, b))


def route(c, m):
    """Returns O/P/W shared gradients and exact measured algebra receipts."""
    import torch
    cc, mm, cm = dot64(c, c), dot64(m, m), dot64(c, m)
    beta = torch.minimum(cm, cm.new_zeros(()))/mm if float(mm) > 0 else cm.new_zeros(())
    cp = [x-beta.to(x.dtype)*y for x, y in zip(c, m)]
    pp = dot64(cp, cp)
    alpha = (pp/cc).sqrt() if float(cc) > 0 else cc.new_zeros(())
    cw = [alpha.to(x.dtype)*x for x in c]
    ww = dot64(cw, cw)
    denominator = max(float(pp.sqrt()), float(ww.sqrt()))
    relative = abs(float(pp.sqrt())-float(ww.sqrt()))/denominator if denominator else 0.
    projection_residual = float(dot64(cp, m))
    require(relative <= 1e-6, "P_W_CLASSIFICATION_NORM")
    if float(cm) >= 0:
        require(all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(c, cp, cw)),
                "NONCONFLICT_PATHS_DIFFER")
    if float(cm) < 0 and float(mm) > 0:
        normalized = abs(projection_residual)/max(float(pp.sqrt()*mm.sqrt()), 1e-300)
        # Collinear removal can leave a zero vector; test relative to original c,m.
        algebra = abs(projection_residual)/max(float(cc.sqrt()*mm.sqrt()), 1e-300)
        require(algebra <= 1e-6, "PROJECTION_ALGEBRA")
    else:
        normalized, algebra = None, 0.
    return {name: [x+y for x, y in zip(m, cls)] for name, cls in (("O", c), ("P", cp), ("W", cw))}, dict(
        c_norm=float(cc.sqrt()), m_norm=float(mm.sqrt()), c_dot_m=float(cm),
        cosine=float(cm/(cc*mm).sqrt()) if float(cc*mm) > 0 else None,
        zero_c=float(cc) == 0, zero_m=float(mm) == 0, conflict=float(cm) < 0,
        beta=float(beta), alpha=float(alpha), projected_c_norm=float(pp.sqrt()),
        weighted_c_norm=float(ww.sqrt()), relative_class_norm_error=relative,
        projected_dot_m=projection_residual, projection_relative_error=algebra)


def rng_capture():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def rng_restore(state):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


class NativeRuntime:
    """Shared reusable native predictor/criterion adapter for E099 only."""
    def __init__(self, cache, decoder_inputs, repo, config, checkpoint, device="cuda:0"):
        import torch
        self.torch, self.cache, self.inputs = torch, Path(cache), Path(decoder_inputs)
        sys.path.insert(0, str(repo))
        from models.maskvrd import MaskVRD
        from models.losses import masked_sigmoid_focal_loss, masked_dice_loss
        self.focal, self.dice = masked_sigmoid_focal_loss, masked_dice_loss
        cfg = json.loads(Path(config).read_text())
        self.model = MaskVRD(cfg["model_config"], device).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict_ema"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        self.model.requires_grad_(False)
        self.model.predictor.requires_grad_(True)
        self.model.eval()
        require(all(not p.requires_grad for n, p in self.model.named_parameters()
                    if not n.startswith("predictor.")), "UPSTREAM_NOT_FROZEN")
        require(not self.model.with_fuzzy, "FUZZY_UNSUPPORTED")
        require(self.model.loss_factor["loss_class"] == 2 and self.model.loss_factor["loss_mask"] == 2
                and self.model.loss_factor["loss_dice"] == 5 and self.model.loss_factor["eos_coef"] == .1,
                "NATIVE_LOSS_COEFFICIENTS")
        self.device = device
        self.named = list(self.model.predictor.named_parameters())
        self.params = [p for n, p in self.named]
        self.shared_indices = [i for i, (n, p) in enumerate(self.named)
                               if not n.startswith(("class_embed.", "mask_embed."))]
        self.private_indices = [i for i in range(len(self.params)) if i not in self.shared_indices]
        require(bool(self.shared_indices) and bool(self.private_indices), "PARAMETER_PARTITION")
        self.optimizer = torch.optim.AdamW(self.params, lr=3e-4, weight_decay=0)
        self.batches = {}
        for role, count in (("train", 88), ("adapter_dev", 23)):
            paths = sorted((self.cache/role).glob("*.pt"))
            require(len(paths) == count, "BATCH_SUPPORT")
            self.batches[role] = paths

    def load_batch(self, path):
        t = self.torch
        c = t.load(path, map_location="cpu", weights_only=False)
        inp = t.load(self.inputs/Path(path).name, map_location="cpu", weights_only=False)
        return dict(c=c, x=inp["x"].to(self.device), f=c["f"].to(self.device),
                    key_mask=inp["key_mask"].to(self.device), valid=c["valid"].to(self.device),
                    preds=[v.to(self.device) for v in c.get("preds", [])],
                    targets=[v.to(self.device) for v in c.get("targets", [])])

    def forward(self, batch, training=False):
        self.model.predictor.train(training)
        return self.model.predictor(batch["x"], batch["f"], batch["key_mask"], output_mask=batch["valid"])

    def layers(self, output):
        layers = list(output.get("aux_outputs", [])) + [output]
        require(len(layers) == 4, "AUX_LAYER_CONTRACT")
        return layers

    def matches(self, output, batch):
        return [self.model.bipartite_match(
            layer["pred_logits"], batch["preds"], layer["pred_masks"], batch["targets"],
            None, batch["valid"]) for layer in self.layers(output)]

    def losses(self, output, batch, matches=None):
        """Native all-layer CE and event mask terms. Frozen matches when supplied."""
        t = self.torch
        if matches is None:
            matches = self.matches(output, batch)
        events = {}
        for metadata in batch["c"]["metadata"]:
            for e in metadata["events"]:
                events[e["event_id"]] = dict(meta=e, terms=[])
        ce = []
        native_mask = []
        # Tensor argument required by the original scripted loss interface.
        # Its value, event normalization, device and float32 precision stay fixed.
        num = output["pred_masks"].new_tensor(float(max(sum(len(p) for p in batch["preds"]), 1)))
        for layer, (indices, lm) in zip(self.layers(output), matches):
            ce.append(self.model.loss_labels(layer["pred_logits"], layer["pred_masks"], batch["preds"],
                                             batch["targets"], None, indices, num, lm)["loss_class"])
            original = self.model.loss_masks(layer["pred_logits"], layer["pred_masks"], batch["preds"],
                                             batch["targets"], None, indices, num, lm)
            native_mask.append(sum(original.values()))
            for b, (queries, targets) in enumerate(indices):
                for q, target in zip(queries.tolist(), targets.tolist()):
                    e = batch["c"]["metadata"][b]["events"][target]
                    z, y, valid = layer["pred_masks"][b, q:q+1], batch["targets"][b][target:target+1], batch["valid"][b]
                    one = z.new_tensor(1.)
                    value = 2*self.focal(z, y, one, valid)+5*self.dice(z, y, one, valid)
                    events[e["event_id"]]["terms"].append(value)
        values = {eid: sum(record["terms"]) for eid, record in events.items() if len(record["terms"]) == 4}
        missing = [eid for eid in events if eid not in values]
        total_mask = sum(native_mask)
        if not missing:
            reconstructed = sum(values.values())/num
            require(t.allclose(reconstructed.detach(), total_mask.detach(), atol=4e-6, rtol=1e-6),
                    "EVENT_LOSS_RECONSTRUCTION")
        short = sum((value for eid, value in values.items() if events[eid]["meta"]["duration_seconds"] < 3),
                    total_mask*0)/num
        long = sum((value for eid, value in values.items() if events[eid]["meta"]["duration_seconds"] >= 3),
                   total_mask*0)/num
        return dict(Lc=sum(ce), Lm=total_mask, short=short, long=long, events=values,
                    missing=missing, matches=matches,
                    N_short=sum(e["meta"]["duration_seconds"] < 3 for e in events.values()),
                    N_long=sum(e["meta"]["duration_seconds"] >= 3 for e in events.values()))

    def gradients(self, loss, parameters=None, retain_graph=True):
        t = self.torch
        parameters = self.params if parameters is None else parameters
        grads = t.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
        return [t.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads)]

    def snapshot(self):
        return dict(predictor=copy.deepcopy(self.model.predictor.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), rng=rng_capture())

    def restore(self, snapshot):
        self.model.predictor.load_state_dict(snapshot["predictor"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)
        rng_restore(snapshot["rng"])

    def apply(self, shared, private, clip):
        for i, g in zip(self.shared_indices, shared):
            self.params[i].grad = g.clone()*clip
        for i, g in zip(self.private_indices, private):
            self.params[i].grad = g.clone()*clip
        self.optimizer.step()


def serialize_matches(matches):
    return [[dict(query=q.tolist(), target=t.tolist()) for q, t in indices]
            for indices, mask in matches]


def summarize_finite(rows, support, out):
    """88 directions first per event; then predicate/video hierarchy, bootstrap99."""
    import numpy as np
    require(len(rows) == 88*169, "FINITE_STEP_ROW_SUPPORT")
    meta = {r["event_id"]: r for r in support["all_dev_events"]}
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)
    require(set(by_event) == set(meta) and all(len(v) == 88 for v in by_event.values()), "EVENT_DIRECTION_SUPPORT")
    means = {}
    for eid, records in by_event.items():
        means[eid] = {name: float(np.mean([r[name] for r in records]))
                      for name in ("d_OP", "d_OW", "d_WP", "delta_O", "delta_P", "delta_W")}
    videos = sorted({g["video"] for g in support["common_groups"]})
    per_video = {}
    for video in videos:
        groups = [g for g in support["common_groups"] if g["video"] == video]
        metrics = {}
        for metric in ("d_OP", "d_OW", "d_WP"):
            short = [np.mean([means[e][metric] for e in g["short"]]) for g in groups]
            long = [np.mean([means[e][metric] for e in g["long"]]) for g in groups]
            metrics[metric+"_short"] = float(np.mean(short))
            metrics[metric+"_long"] = float(np.mean(long))
            metrics[metric+"_short_minus_long"] = float(np.mean(short)-np.mean(long))
        per_video[video] = metrics
    draws = np.random.default_rng(99).integers(0, len(videos), size=(10000, len(videos)))
    result = {}
    for metric in next(iter(per_video.values())):
        values = np.array([per_video[v][metric] for v in videos])
        result[metric] = dict(point=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist())
    all_videos = sorted({r["video_id"] for r in meta.values()})
    all_result = {}
    all_draws = np.random.default_rng(99).integers(0, len(all_videos), size=(10000, len(all_videos)))
    for metric in next(iter(means.values())):
        values = np.array([np.mean([means[e][metric] for e in meta if meta[e]["video_id"] == v])
                           for v in all_videos])
        all_result[metric] = dict(point=float(values.mean()),
                                 ci95=np.quantile(values[all_draws].mean(1), [.025, .975]).tolist())
    np.savez_compressed(Path(out)/"bootstrap_indices.npz", common=draws, all_dev=all_draws)
    result = dict(common=result, all_dev=all_result, per_video=per_video, per_event=means,
                  common_video_order=videos, all_video_order=all_videos,
                  bootstrap_seed=99, repetitions=10000, reduction="88 directions/event, predicate within video, videos equally")
    write(Path(out)/"finite_summary.json", result)
    return result


def measure_state(args):
    import numpy as np
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    support = support_audit(args.cache, out)
    if not support["support_pass"]:
        write(out/"status.json", dict(status="SUPPORT_INSUFFICIENT", candidate_rejection="SHORT_SPECIFICITY_UNIDENTIFIABLE",
                                      requires_commander_technical_capacity_confirmation=True))
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(99)
    torch.cuda.manual_seed_all(99)
    runtime = NativeRuntime(args.cache, args.decoder_inputs, args.repo, args.config, args.checkpoint)
    if args.state != "C0":
        require(args.state in ("O250", "O500") and args.state_file, "STATE_ARGUMENT")
        saved = torch.load(args.state_file, map_location=runtime.device, weights_only=False)
        require(saved["arm"] == "O" and saved["step"] == int(args.state[1:]), "O_STATE_IDENTITY")
        runtime.model.predictor.load_state_dict(saved["predictor"], strict=True)
        runtime.optimizer.load_state_dict(saved["optimizer"])
        rng_restore(saved["rng"])
    start = time.monotonic()
    baseline_state = runtime.snapshot()
    direction_rng = copy.deepcopy(baseline_state["rng"])
    dev = [runtime.load_batch(p) for p in runtime.batches["adapter_dev"]]
    dev_matches, baseline = [], {}
    with torch.no_grad():
        for batch in dev:
            output = runtime.forward(batch, training=False)
            loss = runtime.losses(output, batch)
            require(not loss["missing"], "DEV_UNMATCHED_C0_SUPPORT")
            dev_matches.append(loss["matches"])
            baseline.update({eid: float(v) for eid, v in loss["events"].items()})
    require(len(baseline) == 169, "DEV_BASELINE")
    write(out/"dev_frozen_matching.json", [serialize_matches(m) for m in dev_matches])
    write(out/"dev_baseline_loss.json", baseline)
    shared = [runtime.params[i] for i in runtime.shared_indices]
    mean_c = [torch.zeros_like(p, dtype=torch.float64) for p in shared]
    rows, ledger = [], []
    for direction, path in enumerate(runtime.batches["train"]):
        runtime.restore(baseline_state)
        # Advance the state's native RNG stream once per direction. All three
        # virtual updates reuse this single stochastic forward and matching.
        rng_restore(direction_rng)
        batch = runtime.load_batch(path)
        output = runtime.forward(batch, training=True)
        direction_rng = rng_capture()
        loss = runtime.losses(output, batch)
        require(not loss["missing"], "TRAIN_TARGET_MATCHING_MISSING")
        cg, mg = runtime.gradients(loss["Lc"]), runtime.gradients(loss["Lm"])
        sg = runtime.gradients(loss["short"], shared)
        lg = runtime.gradients(loss["long"], shared, retain_graph=False)
        c, m = [cg[i] for i in runtime.shared_indices], [mg[i] for i in runtime.shared_indices]
        directions, algebra = route(c, m)
        for accumulator, g in zip(mean_c, c):
            accumulator.add_(g.double()/88)
        private = [cg[i]+mg[i] for i in runtime.private_indices]
        original = [cg[i]+mg[i] for i in range(len(cg))]
        total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in original]), 2)
        clip = torch.clamp(1./(total_norm+1e-6), max=1.)
        record = dict(batch=path.name, direction=direction, Lc=float(loss["Lc"].detach()),
                      Lm=float(loss["Lm"].detach()), N_short=loss["N_short"], N_long=loss["N_long"],
                      short_norm=float(dot64(sg, sg).sqrt()), long_norm=float(dot64(lg, lg).sqrt()),
                      c_dot_short=float(dot64(c, sg)), c_dot_long=float(dot64(c, lg)),
                      original_total_norm=float(total_norm), original_clip=float(clip),
                      matching=serialize_matches(loss["matches"]), **algebra)
        record["short_gradient_norm_fraction_of_total"] = (record["short_norm"]/record["m_norm"]
                                                           if record["m_norm"] else None)
        record["long_gradient_norm_fraction_of_total"] = (record["long_norm"]/record["m_norm"]
                                                          if record["m_norm"] else None)
        record["short_zero_gradient"] = record["short_norm"] == 0
        record["long_zero_gradient"] = record["long_norm"] == 0
        with (out/"gradient_batches.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        ledger.append(record)
        variants, private_delta, parameter_updates = {}, {}, {}
        for arm in ("O", "P", "W"):
            runtime.restore(baseline_state)
            runtime.apply(directions[arm], private, clip)
            delta = {runtime.named[i][0]: (runtime.params[i].detach()-baseline_state["predictor"][runtime.named[i][0]]).clone()
                     for i in runtime.private_indices}
            if arm == "O":
                private_delta = delta
            else:
                require(all(torch.equal(value, private_delta[name]) for name, value in delta.items()),
                        "PRIVATE_HEAD_DELTAS_DIFFER")
            shared_delta = [runtime.params[i].detach()-baseline_state["predictor"][runtime.named[i][0]]
                            for i in runtime.shared_indices]
            record[arm+"_shared_delta_norm"] = float(dot64(shared_delta, shared_delta).sqrt())
            record[arm+"_private_delta_norm"] = float(dot64(list(delta.values()), list(delta.values())).sqrt())
            parameter_updates[arm] = dict(
                shared={runtime.named[i][0]: float(d.double().norm()) for i, d in zip(runtime.shared_indices, shared_delta)},
                private={name: dict(norm=float(d.double().norm()),
                                   max_abs_difference_from_O=float((d-private_delta[name]).abs().max()))
                         for name, d in delta.items()})
            measured = {}
            with torch.no_grad():
                for db, matching in zip(dev, dev_matches):
                    dl = runtime.losses(runtime.forward(db, training=False), db, matches=matching)
                    require(not dl["missing"], "FIXED_DEV_MATCHING_MISSING")
                    measured.update({eid: float(value) for eid, value in dl["events"].items()})
            require(set(measured) == set(baseline), "DEV_VIRTUAL_SUPPORT_CHANGED")
            variants[arm] = measured
        for eid, base in baseline.items():
            o, p, w = (variants[arm][eid] for arm in ("O", "P", "W"))
            row = dict(state=args.state, direction=direction, train_batch=path.name, event_id=eid,
                       before=base, O=o, P=p, W=w, d_OP=(o-p)/max(base, 1e-8),
                       d_OW=(o-w)/max(base, 1e-8), d_WP=(w-p)/max(base, 1e-8),
                       delta_O=o-base, delta_P=p-base, delta_W=w-base)
            rows.append(row)
            with (out/"finite_events.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
        with (out/"virtual_steps.jsonl").open("a") as f:
            f.write(json.dumps(dict(batch=path.name, original_clip=float(clip), private_deltas_exact=True,
                                   parameter_update_norms=parameter_updates,
                                   **{k: v for k, v in record.items() if "delta_norm" in k}))+"\n")
        del output, loss, cg, mg, sg, lg, c, m, directions, private, original, batch
        print(json.dumps(dict(state=args.state, train_directions=direction+1, total=88,
                              elapsed=time.monotonic()-start)), flush=True)
    require(any(r["c_norm"] > 0 for r in ledger) and any(r["m_norm"] > 0 for r in ledger),
            "SHARED_GRADIENT_PATH_NOT_PRESENT")
    runtime.restore(baseline_state)
    dev_dot = {}
    for batch, matching in zip(dev, dev_matches):
        values = runtime.losses(runtime.forward(batch, training=False), batch, matches=matching)["events"]
        for j, (eid, value) in enumerate(values.items()):
            g = runtime.gradients(value, shared, retain_graph=j+1 < len(values))
            dev_dot[eid] = float(dot64(mean_c, g))
    write(out/"dev_class_gradient_dot.json", dict(values=dev_dot,
          reduction="dot(mean of 88 train classification gradients, dev event gradient); exactly linear equivalent to 88 dots then average"))
    summary = summarize_finite(rows, support, out)
    by_video = defaultdict(list)
    for group in support["common_groups"]:
        by_video[group["video"]].append(float(np.mean([dev_dot[e] for e in group["short"]])))
    short_dot = float(np.mean([np.mean(v) for v in by_video.values()]))
    gates = dict(short_OP=summary["common"]["d_OP_short"]["ci95"][0] > 0,
                 short_WP=summary["common"]["d_WP_short"]["ci95"][0] > 0,
                 short_specificity=summary["common"]["d_OP_short_minus_long"]["ci95"][0] > 0,
                 short_class_dot_negative=short_dot < 0)
    status = ("C0_MECHANISM_GATE_PASS" if all(gates.values()) else "MECHANISM_GATE_NOT_PASSED") \
        if args.state == "C0" else "POSTTRAIN_STATE_MEASUREMENT_COMPLETE"
    write(out/"status.json", dict(status=status,
          state=args.state, gates=gates, short_class_dot=short_dot, train_directions=88,
          dev_events=169, elapsed_seconds=time.monotonic()-start,
          conflict_batches=sum(r["conflict"] for r in ledger),
          virtual_steps_absorbed_into_training=False,
          interpretation="C0 alone is the native-mechanism gate; O250/O500 are descriptive only"))


def rebuild_predictor(args):
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    runtime = NativeRuntime(args.cache, args.decoder_inputs, args.repo, args.config, args.checkpoint)
    files = [p for role in ("train", "adapter_dev", "eval")
             for p in sorted((Path(args.cache)/role).glob("*.pt"))]
    require(len(files) == 1725, "DIRECT_REBUILD_BATCH_COUNT")
    maximum = 0.
    start = time.monotonic()
    with torch.no_grad():
        for j, path in enumerate(files):
            batch = runtime.load_batch(path)
            output = runtime.forward(batch, training=False)
            errors, equal = [], True
            for li, layer in enumerate(runtime.layers(output)):
                for name, old in (("pred_logits", batch["c"]["cls"][li]),
                                  ("pred_masks", batch["c"]["masks"][li])):
                    new = layer[name].detach().cpu()
                    require(new.shape == old.shape and new.dtype == old.dtype, "DIRECT_SHAPE_DTYPE")
                    require(bool(torch.isfinite(new).all() and torch.isfinite(old).all()), "DIRECT_NONFINITE")
                    errors.append(float((new-old).abs().max()))
                    if name == "pred_logits":
                        equal &= torch.equal(new.argmax(-1), old.argmax(-1))
                        equal &= torch.equal(new.softmax(-1)[..., 1:].topk(6, -1).indices,
                                             old.softmax(-1)[..., 1:].topk(6, -1).indices)
                    else:
                        equal &= torch.equal(new.sigmoid() > .5, old.sigmoid() > .5)
            maximum = max(maximum, max(errors))
            receipt = dict(batch=str(path), max_abs=max(errors), discrete_equal=equal,
                           cached_input_sha256=sha(Path(args.decoder_inputs)/path.name))
            with (out/"direct_rebuild_rows.jsonl").open("a") as f:
                f.write(json.dumps(receipt)+"\n")
            require(max(errors) <= 4e-6 and equal, "DIRECT_PREDICTOR_NUMERICAL_FAILURE")
            if j % 100 == 0:
                print(json.dumps(dict(checked=j+1, total=1725, maximum=maximum)), flush=True)
    write(out/"status.json", dict(status="DIRECT_PREDICTOR_REBUILD_PASS", batches=1725,
                                  maximum_abs_error=maximum, elapsed_seconds=time.monotonic()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("support", "state", "rebuild"), default="support")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decoder-inputs")
    parser.add_argument("--repo", default="/root/rebinding_project/VrdONE")
    parser.add_argument("--config", default="/root/autodl-tmp/VrdONE-data/commander_native_error_025/export_v1/effective_config.json")
    parser.add_argument("--checkpoint", default="/root/rebinding_project/VrdONE/experiments/vrdone_vidor/ckpt_vidor.pth")
    parser.add_argument("--state", choices=("C0", "O250", "O500"), default="C0")
    parser.add_argument("--state-file")
    args = parser.parse_args()
    if args.mode == "support":
        result = support_audit(args.cache, args.out)
        print(json.dumps({k: result[k] for k in ("common_N_videos", "common_short", "common_long", "support_pass")}))
    else:
        try:
            (measure_state if args.mode == "state" else rebuild_predictor)(args)
        except Exception as exc:
            import traceback
            Path(args.out).mkdir(parents=True, exist_ok=True)
            write(Path(args.out)/"status.json", dict(status="TECHNICAL_FAILURE", error=str(exc), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
