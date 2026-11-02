"""Synthetic-only structural tests. Run: python test_heads.py [--device cuda]."""
import argparse
import copy
import unittest

import torch

from heads import INVALID_SCORE, SpanHead, target_span_loss


DEVICE = "cpu"


def record(t=12, batch=2, dim=256, layers=3, queries=9):
    generator = torch.Generator().manual_seed(278)
    return {
        "f": torch.randn(batch, dim, t, generator=generator).to(DEVICE),
        "hs": torch.randn(layers * batch, dim, queries, generator=generator).to(DEVICE),
        "cls": torch.randn(layers, batch, queries, 51, generator=generator).to(DEVICE),
        "masks": torch.full((layers, batch, queries, t), -3., device=DEVICE),
        "valid": torch.ones(batch, 1, t, dtype=torch.bool, device=DEVICE),
    }


class ReadoutTests(unittest.TestCase):
    def test_native_hull_and_disjoint_are_exact(self):
        r = record()
        r["masks"][-1, 0, 0, 2:5] = 3
        r["masks"][-1, 0, 1, [1, 2, 7, 8]] = 3
        r["masks"][-1, 1, 2, 0:12] = 3
        for mode in ("independent", "joint"):
            head = SpanHead(mode).to(DEVICE)
            first, last, fallback = head.anchor(r, [0, 0, 1], [0, 1, 2])
            self.assertEqual(first.tolist(), [2, 1, 0])
            self.assertEqual(last.tolist(), [4, 8, 11])
            self.assertFalse(fallback.any())
            s, e = head.decode(r, [0, 0, 1], [0, 1, 2])
            self.assertTrue(torch.equal(first, s) and torch.equal(last, e))

    def test_same_initial_state_count_and_scores(self):
        r = record()
        r["masks"][-1, :, :, 3:8] = 1
        a = SpanHead("independent").to(DEVICE)
        b = SpanHead("joint").to(DEVICE)
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        self.assertTrue(torch.equal(a.scores(r, 0, [0, 1, 7]),
                                    b.scores(r, 0, [0, 1, 7])))
        self.assertTrue(torch.equal(a.decode(r, 0, [0, 7])[0],
                                    b.decode(r, 0, [0, 7])[0]))

    def test_padding_holes_and_fallback(self):
        r = record(t=9)
        r["valid"][0, 0, 6:] = False
        r["valid"][0, 0, 2] = False
        r["masks"][-1, 0, 0, 3] = 4
        r["masks"][-1, 0, 1, 8] = 8  # Padded peak must be ignored.
        h = SpanHead("joint").to(DEVICE)
        s, e, flag = h.anchor(r, 0, [0, 1])
        self.assertEqual(s.tolist(), [3, 0])
        self.assertEqual(e.tolist(), [4, 1])
        self.assertEqual(flag.tolist(), [True, True])
        z = h.scores(r, 0, [0, 1])
        self.assertTrue((z[:, 6:] == INVALID_SCORE).all())
        self.assertTrue((z[:, :, 6:] == INVALID_SCORE).all())
        self.assertTrue((z[:, 2] == INVALID_SCORE).all())
        self.assertTrue((z[:, :, 2] == INVALID_SCORE).all())
        self.assertTrue((z.diagonal(dim1=1, dim2=2) == INVALID_SCORE).all())
        ds, de = h.decode(r, 0, [0, 1])
        self.assertTrue(torch.equal(ds, s) and torch.equal(de, e))
        r["valid"][1] = False
        r["valid"][1, 0, 4] = True
        self.assertEqual(h.decode(r, 1, 0)[0].tolist(), [-1])
        self.assertEqual(h.decode(r, 1, 0)[1].tolist(), [-1])

    def test_variable_shapes_and_empty_selection(self):
        h = SpanHead("joint", dim=16, hidden=7, rank=3).to(DEVICE)
        for t, n in ((2, 1), (7, 4), (19, 9)):
            r = record(t=t, dim=16)
            z = h.scores(r, 0, list(range(n)))
            self.assertEqual(tuple(z.shape), (n, t, t))
            s, e = h.decode(r, 0, list(range(n)))
            self.assertTrue((s < e).all())
        z = h.scores(r, [], [])
        self.assertEqual(tuple(z.shape), (0, 19, 19))
        self.assertEqual(h.decode(r, [], [])[0].numel(), 0)

    def test_loss_accounting_and_gate_gradient(self):
        torch.manual_seed(149)
        r = record(t=12)
        r["valid"][1, 0, 7:] = False
        r["masks"][-1, :, :, 2:6] = 3
        h = SpanHead("joint").to(DEVICE)
        score = h.scores(r, [0, 0, 1, 1], [0, 1, 2, 3])
        loss, eligible = target_span_loss(score, [1, 3, 5, 6], [8, 3, 9, 2])
        self.assertEqual(eligible, 1)  # Singleton, padding, reversed excluded.
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(h.gate.grad)
        self.assertGreater(abs(float(h.gate.grad)), 1e-9)
        for parameter in h.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(h.endpoint.weight.grad.abs().sum()), 0)

    def test_all_ineligible_has_finite_differentiable_zero(self):
        r = record(t=3)
        h = SpanHead().to(DEVICE)
        score = h.scores(r, 0, [0, 1])
        loss, eligible = target_span_loss(score, [1, -1], [1, 2])
        self.assertEqual(eligible, 0)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(h.gate.grad))

    def test_only_final_layer_is_used_and_no_gt_dependency(self):
        r = record()
        r["masks"][-1, :, :, 1:5] = 3
        h = SpanHead("joint").to(DEVICE)
        with torch.no_grad():
            h.gate.fill_(.3)
            h.endpoint.weight.normal_(std=.1)
        before = h.scores(r, [0, 1], [3, 2]).detach()
        r["cls"][:-1] = 100
        r["masks"][:-1] = 100
        r["hs"][:-r["cls"].shape[1]] = 100
        r["targets"] = "This must never be inspected by inference"
        self.assertTrue(torch.equal(before, h.scores(r, [0, 1], [3, 2])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args, rest = parser.parse_known_args()
    DEVICE = args.device
    unittest.main(argv=[__file__] + rest)
