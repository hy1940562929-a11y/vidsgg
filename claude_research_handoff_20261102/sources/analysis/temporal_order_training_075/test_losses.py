"""Small synthetic mathematical invariants. Run on CPU; no data or training."""
import unittest
import torch
from losses import focal_rows, rank_domain, rank_rows, compose_focal


class LossTests(unittest.TestCase):
    def test_padding_and_separate_means(self):
        z = torch.zeros((1, 8), requires_grad=True)
        y = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]])
        v = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        a, b = focal_rows(z, y, v)
        self.assertTrue(torch.allclose(b, 2 * a))
        b.sum().backward()
        self.assertEqual(z.grad[0, 4:].abs().sum().item(), 0.)

    def test_other_episode_and_boundary_guard(self):
        y = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0.])
        union = y.bool(); union[8:11] = True
        d = rank_domain(y, union, torch.ones(12, dtype=torch.bool))
        self.assertEqual(d['positive'].tolist(), [2])
        self.assertEqual(d['negative'].tolist(), [5, 6])

    def test_censor_singleton_and_all_positive_fallback(self):
        for y, censored in [(torch.ones(8), False), (torch.tensor([0., 1., 0., 0.]), False), (torch.tensor([0., 1., 1., 1., 0., 0.]), True)]:
            d = rank_domain(y, y.bool(), torch.ones_like(y, dtype=torch.bool), censored)
            rank, ok = rank_rows(torch.zeros((1, len(y))), [d])
            self.assertFalse(ok.item())
            self.assertEqual(rank.item(), 0.)
            got = compose_focal(torch.tensor([1.]), torch.tensor([2.]), rank, ok, 'ordered', 3., 7.)
            self.assertEqual(got.item(), 6.)

    def test_rank_gradient_direction(self):
        z = torch.zeros((1, 8), requires_grad=True)
        y = torch.tensor([0., 1., 1., 1., 0., 0., 0., 0.])
        d = rank_domain(y, y.bool(), torch.ones(8, dtype=torch.bool))
        r, ok = rank_rows(z, [d]); r.sum().backward()
        self.assertTrue(ok.item())
        self.assertLess(z.grad[0, 2].item(), 0.)
        self.assertTrue((z.grad[0, d['negative']] > 0).all())
        self.assertEqual(z.grad[0, 1].item(), 0.)

    def test_initial_mean_matching_and_empty_focal_side(self):
        a = torch.tensor([1., 3., 2.]); b = torch.tensor([2., 8., 4.]); r = torch.tensor([4., 0., 2.]); e = torch.tensor([True, False, True])
        kb = a.mean()/b.mean(); kr = (kb*b[e]).mean()/r[e].mean()
        c = compose_focal(a,b,r,e,'ordered',kb,kr)
        self.assertTrue(torch.allclose(c.mean(), a.mean()))
        x = torch.zeros((1,4)); y = torch.ones_like(x)
        original, balanced = focal_rows(x,y,torch.ones_like(x,dtype=torch.bool))
        self.assertTrue(torch.allclose(original, balanced))

    def test_sampling_limit_and_mixed_fallback(self):
        y=torch.zeros(128);y[10:60]=1
        v=torch.zeros(128,dtype=torch.bool);v[:110]=True
        d=rank_domain(y,y.bool(),v)
        self.assertEqual(len(d['positive']),16);self.assertEqual(len(d['negative']),16)
        self.assertTrue((d['negative']<110).all())
        empty=rank_domain(y,y.bool(),v,True)
        z=torch.zeros((2,128),requires_grad=True)
        r,e=rank_rows(z,[d,empty]);a,b=focal_rows(z,y.repeat(2,1),v.repeat(2,1))
        c=compose_focal(a,b,r,e,'ordered',2.,3.)
        self.assertEqual(c[1].item(),(2*b[1]).item())
        c.sum().backward();self.assertEqual(z.grad[:,110:].abs().sum().item(),0.)


if __name__ == '__main__':
    unittest.main()
