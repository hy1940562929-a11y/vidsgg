"""CPU-only software smoke check. No real data, training run or research metrics."""
import json
import torch

from relation_enhancement import (
    EnhancementConfig, RelationEnhancement, EnhancementCriterion, build_dense_targets,
)


def run():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = EnhancementConfig(pair_dim=32, roi_dim=24, geometry_dim=5,
                            text_dim=16, hidden_dim=16, predicate_chunk_size=2)
    # Synthetic anchors are ONLY software fixtures, not CLIP or empirical evidence.
    model = RelationEnhancement(cfg, torch.randn(3, 16))
    b, t = 2, 32
    valid = torch.arange(t)[None] < torch.tensor([32, 23])[:, None]
    inputs = (torch.randn(b, t, 32), torch.randn(b, t, 24),
              torch.randn(b, t, 24), torch.randn(b, t, 5), valid)
    # Force predicted-background selection solely to execute/test the CDSE path.
    with torch.no_grad():
        model.coarse_head[-1].weight.zero_()
        model.coarse_head[-1].bias.fill_(-8)
        model.motion_strength.fill_(0.1)
        model.context_strength.fill_(0.1)
    masks0 = torch.zeros(3, 32)
    masks0[0, 5:8] = 1
    masks0[1, 17:20] = 1
    masks0[2, 0:32] = 1
    masks1 = torch.zeros(1, 23)
    masks1[0, 9:12] = 1
    targets = build_dense_targets([torch.tensor([1, 1, 2]), torch.tensor([3])],
                                 [masks0, masks1], valid, 3,
                                 assume_exhaustive_annotations=True)
    output = model(*inputs, seconds_per_step=4 / 30, context_enabled=True)
    criterion = EnhancementCriterion()
    losses = criterion(output, targets)
    total = torch.stack(list(losses.values())).sum()
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(total) and all(torch.isfinite(g).all() for g in gradients))
    assert finite
    assert any(g.abs().sum() > 0 for g in gradients)
    return {"status": "SOFTWARE_SMOKE_PASS", "device": "cpu", "synthetic": True,
            "research_result": False, "torch_version": torch.__version__,
            "finite_loss_and_gradients": finite,
            "relation_shape": list(output["relation_logits"].shape),
            "high_resolution_shape": list(output["high_res_features"].shape),
            "cdse_selection_exercised": bool(output["context_available"].any())}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
