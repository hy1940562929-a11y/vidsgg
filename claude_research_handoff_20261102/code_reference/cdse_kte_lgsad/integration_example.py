"""Reusable construction helper; does not launch training or remote execution."""
from pathlib import Path
import importlib
import sys

from relation_enhancement import EnhancementConfig, load_text_anchors


def build_model(
    vrdone_root, model_config, native_state_dict, anchor_path, predicate_names,
    seconds_per_step, device="cpu", assume_exhaustive_annotations=False,
    enhancement_overrides=None, enabled=True,
):
    """Use your existing dataloader, optimizer, training loop and evaluator.

    native_state_dict is the ORIGINAL model's state_dict, not a whole checkpoint.
    Caller must use actual foreground category names in native ID order 1..P.
    This function never picks a GPU or downloads/installs anything.
    """
    root = Path(vrdone_root).resolve()
    if not (root / "models/maskvrd.py").is_file():
        raise ValueError("vrdone_root must contain the original models/maskvrd.py")
    sys.path.insert(0, str(root))
    native_module = importlib.import_module("models.maskvrd")
    if root not in Path(native_module.__file__).resolve().parents:
        raise RuntimeError("A different 'models' package is already imported; use a fresh process")
    from vrdone_adapter import EnhancedMaskVRD
    text = load_text_anchors(anchor_path, predicate_names)
    dimensions = dict(pair_dim=model_config["embd_dim"], roi_dim=model_config["visual_dim"],
                      geometry_dim=model_config["bbox_so_dim"], text_dim=text.shape[1])
    overrides = dict(enhancement_overrides or {})
    if set(dimensions) & set(overrides):
        raise ValueError("Native/text dimensions are derived from the model and cache")
    cfg = EnhancementConfig(**dimensions, **overrides)
    return EnhancedMaskVRD(model_config, device, cfg, text, seconds_per_step,
                          native_state_dict=native_state_dict,
                          assume_exhaustive_annotations=assume_exhaustive_annotations,
                          enabled=enabled).to(device)
