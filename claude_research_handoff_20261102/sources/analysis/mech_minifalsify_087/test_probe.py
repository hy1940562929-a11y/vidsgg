import csv,importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
from probe_math import position_full,local_points,auroc,aggregate,decision
from run_probe import compare
HERE=Path(__file__).resolve().parent

class Tensor:
    def __init__(self,x):self.x=np.asarray(x);self.shape=self.x.shape;self.dtype=self.x.dtype
    def is_floating_point(self):return np.issubdtype(self.dtype,np.floating)
    def numel(self):return self.x.size
    def __sub__(self,b):return Tensor(self.x-b.x)
    def abs(self):return Tensor(np.abs(self.x))
    def max(self):return self.x.max()
class TorchStub:
    @staticmethod
    def isfinite(a):return np.isfinite(a.x)
    @staticmethod
    def equal(a,b):return np.array_equal(a.x,b.x)

class TestProbe(unittest.TestCase):
    def test_positions_extend_exactly(self):
        # Reference is the unchanged formula read from079 layer_math.py; local
        # iCloud-placeholder source cannot be imported reliably in this test.
        for n in (1,2,50,511,512):
            u=np.linspace(-1,1,n,dtype=np.float32)
            expected=np.zeros((256,n),np.float32)
            expected[:9]=np.stack([np.ones(n,np.float32),u,u*u,np.sin(np.pi*u),np.cos(np.pi*u),np.sin(2*np.pi*u),np.cos(2*np.pi*u),np.sin(3*np.pi*u),np.cos(3*np.pi*u)])
            np.testing.assert_array_equal(position_full(n),expected)
        self.assertEqual(position_full(2049).shape,(256,2049));self.assertEqual(position_full(1)[1,0],-1)
    def test_exact_ties(self):
        s=np.array([1,2,2,0,2]);p=np.array([1,1,0,0,0],bool);n=~p
        expected=sum(float(a>b)+.5*float(a==b) for a in s[p] for b in s[n])/(p.sum()*n.sum())
        self.assertEqual(auroc(s,p,n),expected)
    def test_local_union(self):
        f=np.arange(0,80,4);p,n=local_points(f,20,32,[(20,32),(8,16),(36,44)])
        self.assertEqual(f[p].tolist(),[20,24,28]);self.assertEqual(f[n].tolist(),[16,32])
        self.assertIsNone(auroc(np.zeros(len(f)),p,np.zeros(len(f),bool)))
    def test_bootstrap_video_equal(self):
        r=[dict(video_name='a',eligible=True,auroc=dict(fine=1.,fused=0.,position=0.))]*100
        r+=[dict(video_name='b',eligible=True,auroc=dict(fine=0.,fused=1.,position=0.))]
        a=aggregate(r);self.assertEqual(a['comparisons']['fine_minus_fused']['mean'],0)
        self.assertEqual(decision(a),'INCONCLUSIVE_INSUFFICIENT_SUPPORT')
        self.assertEqual(a,aggregate(r))
    def test_support_decision_and_kill(self):
        r=[dict(video_name=str(i//5),eligible=True,auroc=dict(fine=.9,fused=.7,position=.5)) for i in range(100)]
        self.assertEqual(decision(aggregate(r)),'SUPPORTED_FIXED_RECIPE_READABILITY_DIFFERENCE')
        for x in r:x['auroc']=dict(fine=.7,fused=.7,position=.5)
        self.assertEqual(decision(aggregate(r)),'STOP_FIXED_RECIPE_FUSION_DAMAGE_EXPLANATION')
    def test_gate_tolerance_and_discrete(self):
        a={'m':Tensor(np.array([1.],np.float32))};b={'m':Tensor(np.array([1.+3e-6],np.float32))}
        compare(TorchStub,a,b,'pass')
        with self.assertRaises(RuntimeError):compare(TorchStub,a,{'m':Tensor(np.array([1.+5e-6],np.float32))},'fail')
        with self.assertRaises(RuntimeError):compare(TorchStub,{'m':Tensor([True])},{'m':Tensor([False])},'bits')
        with self.assertRaises(RuntimeError):compare(TorchStub,a,{'m':Tensor([np.nan])},'nan')
    def test_locked_members_coordinates_and_points(self):
        m=json.loads((HERE/'members.json').read_text());coords=json.loads((HERE/'eval_coordinates.json').read_text())
        with (HERE/'eval_locked.csv').open() as f:rr=list(csv.DictReader(f))
        bykey={(r['video_name'],int(r['gt_index'])):r for r in rr};self.assertEqual(len(bykey),2573)
        for group,count in [('long_coexist_short_covered',400),('long_coexist_subsecond_covered',163)]:
            self.assertEqual(len(m[group]),count)
            for video,gi in m[group]:
                r=bykey[(video,int(gi))];c=coords[f"{video}/{r['pair_idx']}"];frames=int(c['origin'])+4*np.arange(int(c['N_valid']))
                intervals=[(int(x['begin_fid']),int(x['end_fid'])) for x in rr if (x['video_name'],x['subject_tid'],x['object_tid'],x['predicate'])==(video,r['subject_tid'],r['object_tid'],r['predicate'])]
                p,n=local_points(frames,int(r['begin_fid']),int(r['end_fid']),intervals);self.assertTrue(p.any() and n.any())

if __name__=='__main__':unittest.main()
