"""Read-only numerical audit of B's final development batch; no training."""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from loss_probe import Paths, load_json, import_vendor, write_json

config_file = Path("config.json").resolve()
paths = Paths(load_json(config_file), config_file)
vendor_path = paths.get("vendor078")
contract = load_json(vendor_path / "source_contract043.json")
cohorts = load_json(vendor_path / "cohorts.json")
vendor = import_vendor(vendor_path, paths.repo)
from models.maskvrd import MaskVRD
from dataloaders.vidor import VidOR

torch.set_num_threads(2)
mc = load_json(paths.remap(cohorts["047"]["export"]) / "effective_config.json")
model = MaskVRD(mc["model_config"], "cuda:0").eval()
model._config_eval(mc["inference_config"])
state = torch.load(paths.get("checkpoint"), map_location="cpu",
                   weights_only=False)[contract["checkpoint_key"]]
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
model.cuda().requires_grad_(False)
raw = torch.load(paths.get("cache_root") / "adapter_dev_cache.pt",
                 map_location="cpu", weights_only=False)
record = raw[-1]
manifest = load_json(paths.get("train_export_root") / "adapter_dev/manifest.json")
ref = manifest["records"][-1]
old = torch.load(paths.remap(ref["path"]), map_location="cpu", weights_only=False)
dc = manifest["effective_dataset_config"]
ds = VidOR.__new__(VidOR)
for key in ("with_clip_feature", "feat_stride", "max_seq_len",
            "cut_max_preds", "proposal_max_preds"):
    setattr(ds, key, dc[key])
loader = paths.remap(dc["cache_dir"]) / (dc["cache_tag"] + "_VidOR_training")
xs = []
for row, pm in enumerate(record["metadata"]):
    with (loader / (pm["video_id"] + ".pkl")).open("rb") as f:
        data = pickle.load(f)
    index = [list(k) for k in data["relation_keys"]].index(pm["pair_key"])
    random.seed(pm["sampling_seed"])
    np.random.seed(pm["sampling_seed"])
    torch.manual_seed(pm["sampling_seed"])
    item = ds._train_getitem(data, (index, index + 1))
    assert torch.equal(item["preds_list"][0], old["preds_list"][row])
    assert torch.equal(item["masks_list"][0], old["masks_list"][row])
    xs.append(item["so_features_list"][0])
padded = torch.zeros(len(xs), 2069, 512, device="cuda")
for row, x in enumerate(xs):
    padded[row, :, :x.shape[-1]] = x.cuda()
valid = record["valid"].cuda()
results = []
for tf32, benchmark, deterministic in [
    (True, False, False), (False, False, False),
    (True, True, False), (True, False, True),
]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    with torch.no_grad(), vendor["frozen043"].Capture(model) as cap:
        model._mask_vrd(padded, valid)
    fresh = cap.records[0]
    differences = {k: float((fresh[k].cpu() - record[k]).abs().max())
                   for k in ("f", "cls", "masks")}
    result = dict(cudnn_tf32=tf32, benchmark=benchmark, deterministic=deterministic,
                  differences=differences,
                  binary_changes=int(((fresh["masks"].cpu().sigmoid() > .5)
                                      != (record["masks"].sigmoid() > .5)).sum()),
                  predicate_top6_equal=torch.equal(
                      fresh["cls"].cpu().softmax(-1)[..., 1:].topk(6, -1).indices,
                      record["cls"].softmax(-1)[..., 1:].topk(6, -1).indices))
    results.append(result)
    print(json.dumps(result), flush=True)
write_json(Path("../B_last_batch_numerics_20261020.json"),
           dict(diagnostic_only=True, training=False, batch=22,
                batch_size=len(xs), results=results))
