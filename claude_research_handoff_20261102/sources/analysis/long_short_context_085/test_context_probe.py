import unittest
import numpy as np
from context_probe import choose, overlap_cells, spans, metrics

class ContextTest(unittest.TestCase):
    def test_selection_and_protection(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(2069,150)).astype('float32');tt=np.arange(150)*4
        c,reason=choose(x,tt,(240,270),(170,400),[(170,400)],[(240,270)],30)
        self.assertIsNone(reason);self.assertIsNotNone(c)
        self.assertNotIn('N',c)
        merged=dict(**dict(N=150),**c);self.assertEqual(merged['N'],150)
        L,N,D=[c[k] for k in ('L','N_start','donor')]
        for a,b in ((L,N),(L,D),(N,D)):self.assertGreaterEqual(abs(a-b),8)
        protect=overlap_cells(tt,225,285)
        for k in (L,N,D):self.assertFalse(protect[k:k+8].any())
        for k in (N,D):self.assertFalse(overlap_cells(tt,170,400)[k:k+8].any())
        self.assertLessEqual(c['distance_difference_seconds'],2)
        self.assertTrue(.5<=c['RMS_ratio_L_over_N']<=2)
        y=x.copy();y[:2048,L:L+8]=x[:2048,D:D+8]
        np.testing.assert_array_equal(y[2048:],x[2048:]);np.testing.assert_array_equal(y[:,protect],x[:,protect])
    def test_no_feasible_no_relaxation(self):
        x=np.ones((2069,30),dtype='float32');tt=np.arange(30)*4
        c,reason=choose(x,tt,(48,64),(0,120),[(0,120)],[(48,64)],30)
        self.assertIsNone(c);self.assertIsNotNone(reason)
    def test_native_decoder(self):
        self.assertEqual(spans(np.array([[False,True,False,True],[True,False,False,False]]),np.arange(4)*4),[[4,13],None])
    def test_locked_query_margin(self):
        r=dict(N=100,origin=0,target=[160,200],protected_indices=list(range(36,54)),predicate_id=1,long_predicate_id=2,L=5)
        m=np.zeros((2,100));m[0,40:50]=2;m[1,40:50]=10;c=np.array([[0.,3.,0.],[0.,0.,3.]])
        a=metrics(m,c,m>0,r,0,1);m[0,5:13]=-100;b=metrics(m,c,m>0,r,0,1)
        self.assertEqual(a['query'],0);self.assertEqual(a['target_vs_outside_margin'],2);self.assertEqual(a['target_vs_outside_margin'],b['target_vs_outside_margin'])

if __name__=='__main__':unittest.main()
