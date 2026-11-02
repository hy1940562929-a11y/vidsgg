"""Synthetic math checks; no VidOR experiment."""
import tempfile
import unittest
import importlib.util
import os

from gradient_gate import summarize_finite

try:
    import torch
    from gradient_gate import route
except ImportError:
    torch = None


class HierarchyTests(unittest.TestCase):
    def test_directions_not_independent_bootstrap_units(self):
        events = []
        for v in ("v0", "v1", "v2", "v3"):
            for group, duration in (("s", 1), ("l", 5)):
                events.append(dict(event_id=f"{v}/{group}", video_id=v, predicate_id=1,
                                   duration_seconds=duration))
        support = dict(all_dev_events=events,
                       common_groups=[dict(video=v, predicate=1, short=[f"{v}/s"], long=[f"{v}/l"])
                                      for v in ("v0", "v1", "v2", "v3")])
        # The real routine requires169 events: use complete169 records with4
        # matched groups, remaining events contribute only to the all-dev table.
        for i in range(161):
            events.append(dict(event_id=f"extra/{i}", video_id="extra", predicate_id=2,
                               duration_seconds=1))
        rows = [dict(event_id=e["event_id"], d_OP=1 if e["event_id"].endswith("/s") else 0,
                     d_OW=0, d_WP=1 if e["event_id"].endswith("/s") else 0,
                     delta_O=0, delta_P=-1, delta_W=0)
                for direction in range(88) for e in events]
        with tempfile.TemporaryDirectory() as out:
            result = summarize_finite(rows, support, out)
        self.assertEqual(result["common"]["d_OP_short"]["point"], 1)
        self.assertEqual(result["common"]["d_OP_short_minus_long"]["ci95"], [1, 1])
        self.assertEqual(len(result["common_video_order"]), 4)
        self.assertEqual(len(result["per_event"]), 169)


@unittest.skipIf(torch is None, "torch not installed in this CPU interpreter")
class ProjectionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("E099_NATIVE_LOSSES"), "native loss source not supplied")
    def test_native_tensor_denominators_and_event_reduction(self):
        spec = importlib.util.spec_from_file_location("e099_native_loss_test", os.environ["E099_NATIVE_LOSSES"])
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        z = torch.tensor([[.2, -.7, .5, -.2], [.3, .8, -.4, .1]], requires_grad=True)
        y = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 0.]])
        valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        n = z.new_tensor(2.)
        batch_loss = 2*native.masked_sigmoid_focal_loss(z, y, n, valid) + 5*native.masked_dice_loss(z, y, n, valid)
        events = [2*native.masked_sigmoid_focal_loss(z[i:i+1], y[i:i+1], z.new_tensor(1.), valid[i:i+1])
                  + 5*native.masked_dice_loss(z[i:i+1], y[i:i+1], z.new_tensor(1.), valid[i:i+1])
                  for i in range(2)]
        self.assertTrue(torch.allclose(sum(events)/n, batch_loss, atol=4e-6, rtol=1e-6))
        batch_loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_conflict_projection_and_same_class_norm(self):
        c = [torch.tensor([-2., 3.])]
        m = [torch.tensor([1., 0.])]
        directions, receipt = route(c, m)
        self.assertTrue(torch.equal(directions["P"][0], torch.tensor([1., 3.])))
        self.assertEqual(receipt["projected_dot_m"], 0)
        self.assertLessEqual(receipt["relative_class_norm_error"], 1e-6)
        self.assertTrue(receipt["conflict"])

    def test_positive_dot_paths_exact(self):
        directions, receipt = route([torch.tensor([2., 3.])], [torch.tensor([1., 0.])])
        self.assertTrue(torch.equal(directions["O"][0], directions["P"][0]))
        self.assertTrue(torch.equal(directions["O"][0], directions["W"][0]))
        self.assertFalse(receipt["conflict"])

    def test_zero_norm_and_collinear(self):
        for c, m in (([0., 0.], [1., 0.]), ([1., 2.], [0., 0.]), ([-1., 0.], [1., 0.])):
            directions, receipt = route([torch.tensor(c)], [torch.tensor(m)])
            self.assertTrue(all(torch.isfinite(x).all() for v in directions.values() for x in v))
            self.assertLessEqual(receipt["relative_class_norm_error"], 1e-6)

    def test_same_clip_private_adam_step_identical(self):
        p0 = torch.tensor([1., 2.])
        values = []
        for arm in "OPW":
            p = torch.nn.Parameter(p0.clone())
            optimizer = torch.optim.AdamW([p], lr=3e-4, weight_decay=0)
            p.grad = torch.tensor([.2, -.4])*.25
            optimizer.step()
            values.append(p.detach())
        self.assertTrue(torch.equal(values[0], values[1]) and torch.equal(values[0], values[2]))


if __name__ == "__main__":
    unittest.main()
