"""E099 original-path material/rebuild gate. No optimizer, no model edits."""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/VrdONE-data")
REPO = Path("/root/rebinding_project/VrdONE")
CACHE = ROOT / "commander_cache_import_094/cache_v4"
RAW = ROOT / "commander_mech_minifalsify_087/results_v1/fresh_raw"
OUT = ROOT / "commander_cause_arch_099/preflight_v1"
CHECKPOINT = REPO / "experiments/vrdone_vidor/ckpt_vidor.pth"
CONFIG = ROOT / "commander_native_error_025/export_v1/effective_config.json"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def flatten(x, prefix=""):
    if isinstance(x, dict):
        return {k: v for key, value in x.items()
                for k, v in flatten(value, prefix + "/" + str(key)).items()}
    if isinstance(x, list):
        return {k: v for i, value in enumerate(x)
                for k, v in flatten(value, prefix + "/" + str(i)).items()}
    return {prefix: x.detach().cpu()}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "decoder_inputs").mkdir()
    begin = time.monotonic()
    phase = "MATERIAL"
    current = None
    rows = []
    try:
        import torch
        require(torch.cuda.is_available(), "GPU_UNAVAILABLE")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        expected_sources = json.loads((CACHE / "source_sha256.json").read_text())
        require(sha(CHECKPOINT) == "4dcbc1091faf71330694fce138c3a4ff84e43ca3fb988b66fdacf26745bce627", "CHECKPOINT_SHA")
        cfg = json.loads(CONFIG.read_text())
        old_manifest = json.loads((CONFIG.parent / "manifest.json").read_text())
        require(sha(CONFIG) == old_manifest["outputs"]["effective_config.json"]["sha256"], "CONFIG_SHA")
        repo_hashes = {}
        for p in sorted((REPO / "models").glob("*.py")):
            repo_hashes[str(p)] = sha(p)
            if str(p) in expected_sources:
                require(repo_hashes[str(p)] == expected_sources[str(p)], "SOURCE_CHANGED " + str(p))
        files = [p for role in ("train", "adapter_dev", "eval") for p in sorted((CACHE / role).glob("*.pt"))]
        require(len(files) == 1725, "CACHE_COUNT")
        raw_paths = [RAW / p.name for p in files]
        require(all(p.exists() for p in raw_paths), "MISSING_RAW_INPUT")
        dump("material_receipt.json", {
            "checkpoint_sha256": sha(CHECKPOINT), "config_sha256": sha(CONFIG),
            "manifest_sha256": sha(CACHE / "manifest.json"), "models": repo_hashes,
            "native_batches": len(files), "raw_inputs_present": len(raw_paths),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "settings": {"cuda_matmul_tf32": False, "cudnn_tf32": True, "benchmark": False, "threads": 2},
            "model_config": cfg["model_config"],
        })
        sys.path.insert(0, str(REPO))
        from models.maskvrd import MaskVRD
        model = MaskVRD(cfg["model_config"], "cuda:0").eval()
        model._config_eval(cfg["inference_config"])
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict_ema"]
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        del state
        model.cuda().requires_grad_(False)
        saved = {}

        def predictor_input(module, args, kwargs):
            saved["x"] = args[0].detach().cpu()
            saved["key_mask"] = args[2].detach().cpu()
            saved["f"] = args[1].detach().cpu()
            saved["output_mask"] = kwargs["output_mask"].detach().cpu()

        def attention_input(module, args):
            saved["attention"] = args[0].detach().cpu()

        def cross_input(module, args, kwargs):
            require(kwargs.get("_attn_mask") is None, "QUERY_SPECIFIC_HARD_MASK")
            saved["cross_key_mask"] = kwargs["_kv_mask"].detach().cpu()

        layers = model.predictor.transformer.decoder.layers
        require(len(layers) == 4 and model.predictor.num_queries == 9, "DECODER_ARCH")
        attn = layers[-1].multihead_attn
        require(type(attn).__name__ == "MaskedMHCA_QKV" and attn.n_kv_stride == 1, "ATTENTION_INTERFACE")
        model.predictor.register_forward_pre_hook(predictor_input, with_kwargs=True)
        attn.register_forward_pre_hook(cross_input, with_kwargs=True)
        attn.attn_drop.register_forward_pre_hook(attention_input)
        dump("attention_interface.json", {
            "class": type(attn).__name__, "layers": len(layers), "heads": attn.n_head,
            "n_kv_stride": attn.n_kv_stride, "key_kernel": 3,
            "backbone_time_scale": cfg["model_config"]["scale_factor"] ** cfg["model_config"]["backbone_arch"][-1],
            "captured": "last decoder cross-attention all heads, before dropout",
            "training_updates": 0,
        })
        phase = "NATIVE_REBUILD"
        for i, (p, raw_path) in enumerate(zip(files, raw_paths)):
            current = str(p)
            saved.clear()
            c = torch.load(p, map_location="cpu", weights_only=False)
            raw_sha = sha(raw_path)
            require(str(raw_path) in expected_sources and expected_sources[str(raw_path)] == raw_sha, "RAW_SHA " + str(raw_path))
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            require("inputs" in raw and "valid" in raw and "native" in raw, "RAW_FIELDS")
            with torch.no_grad():
                output = model._mask_vrd(raw["inputs"].cuda(), raw["valid"].cuda())
            now = flatten(output)
            require(set(now) == set(raw["native"]), "OUTPUT_KEYS")
            errors = {}
            finite = True
            discrete = True
            for name, actual in now.items():
                reference = raw["native"][name]
                require(actual.shape == reference.shape and actual.dtype == reference.dtype, "SHAPE_DTYPE " + name)
                if actual.is_floating_point():
                    finite &= bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
                    errors[name] = float((actual - reference).abs().max())
                    if name.endswith("pred_masks"):
                        discrete &= torch.equal(actual.sigmoid() > .5, reference.sigmoid() > .5)
                    if name.endswith("pred_logits"):
                        discrete &= torch.equal(actual.argmax(-1), reference.argmax(-1))
                        discrete &= torch.equal(actual.softmax(-1)[..., 1:].topk(6, -1).indices, reference.softmax(-1)[..., 1:].topk(6, -1).indices)
                else:
                    discrete &= torch.equal(actual, reference)
            receipt = {"batch": current, "raw": str(raw_path), "raw_sha256": raw_sha,
                       "max_abs": max(errors.values()), "errors": errors, "finite": finite,
                       "discrete_equal": discrete, "elapsed_seconds": time.monotonic() - begin}
            rows.append(receipt)
            with (OUT / "rebuild_batches.jsonl").open("a") as f:
                f.write(json.dumps(receipt) + "\n")
            require(finite and receipt["max_abs"] <= 4e-6 and discrete, "NATIVE_REBUILD " + json.dumps(receipt))
            require(torch.equal(saved["f"], c["f"]) and torch.equal(saved["output_mask"], c["valid"]), "BRANCH_INPUT_CHANGED")
            require(torch.equal(saved["key_mask"], saved["cross_key_mask"]), "KEY_COORDINATE_MASK")
            if p.parent.name in ("train", "adapter_dev"):
                require("preds" in c and "targets" in c and "metadata" in c, "FULL_TARGETS_MISSING")
            torch.save({k: v for k, v in saved.items() if k != "f"},
                       OUT / "decoder_inputs" / p.name)
            if (i + 1) % 100 == 0 or i == 0:
                print(json.dumps({"checked": i + 1, "total": len(files),
                                  "max_error": max(r["max_abs"] for r in rows),
                                  "elapsed_seconds": time.monotonic() - begin}), flush=True)
        dump("status.json", {"status": "NATIVE_REBUILD_PASS", "checked": len(rows),
                            "max_abs": max(r["max_abs"] for r in rows),
                            "training_runs": 0, "elapsed_seconds": time.monotonic() - begin})
    except Exception as exc:
        dump("status.json", {"status": "TECHNICAL_FAILURE", "phase": phase,
                            "batch": current, "error": str(exc), "traceback": traceback.format_exc(),
                            "checked": len(rows), "training_runs": 0,
                            "elapsed_seconds": time.monotonic() - begin})
        print((OUT / "status.json").read_text(), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
