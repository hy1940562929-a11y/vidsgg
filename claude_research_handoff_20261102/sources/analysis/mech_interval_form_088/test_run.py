"""Synthetic CPU tests only; never loads the088 real input manifest or scores."""
import copy,hashlib,inspect,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import interval_math as m
from run import build_library,sha

class Test088(unittest.TestCase):
    def fixture(self,n=12):return np.arange(n,dtype=np.int64)*4,np.full((9,n),-3.,np.float32)
    def test_disconnected_peaks_and_plus_one(self):
        f,x=self.fixture();x[0,1:3]=1.;x[0,7:9]=3.
        r,n,a=m.generate_pair('v',2,f,x)
        self.assertEqual((r['H'][0]['start'],r['H'][0]['end']),(4,33))
        self.assertEqual((r['C'][0]['start'],r['C'][0]['end']),(28,33))
        self.assertEqual(r['selection'][0]['selectable_components'],2)
        self.assertEqual(a['H_candidates'],1);self.assertEqual(a['C_candidates'],1)
    def test_empty_and_min5(self):
        f,x=self.fixture();x[:]=0.
        r,n,a=m.generate_pair('v',0,f,x);self.assertEqual(a['H_candidates'],0);self.assertEqual(a['C_candidates'],0)
        self.assertTrue(all(z is None for rep in n['null_H'] for z in rep))
        x[0,2]=2.;r,_,_=m.generate_pair('v',0,f,x);self.assertIsNone(r['H'][0]);self.assertIsNone(r['C'][0])
        x[0,4]=2.;r,_,a=m.generate_pair('v',0,f,x);self.assertEqual(r['H'][0]['frame_length'],9);self.assertIsNone(r['C'][0]);self.assertEqual(a['H_candidates'],1);self.assertEqual(a['C_candidates'],0)
        x[1,2:4]=2.;r,_,_=m.generate_pair('v',0,f,x);self.assertEqual(r['C'][1]['frame_length'],5)
    def test_mean_probability_peak_and_early_ties(self):
        f,x=self.fixture(16);p=np.zeros_like(x);p[0,1:3]=[.75,.75];p[0,5:7]=[.625,.875];p[0,10:12]=[.625,.875]
        # Isolate the selector with exactly representable sigmoid probabilities.
        with patch.object(m,'probability_audit',return_value=(p,p>.5,{})):
            r,_,_=m.generate_pair('v',0,f,x)
        self.assertEqual(r['C'][0]['first_index'],5)
        self.assertEqual(r['C'][0]['mean_probability'],.75)
        self.assertEqual(r['C'][0]['peak_probability'],.875)
    def test_numeric_boundary_is_failure(self):
        _,x=self.fixture();x[0,0]=np.float32(1e-8)
        with self.assertRaisesRegex(ValueError,'threshold boundary mismatch'):m.probability_audit(x)
        x[0,0]=np.nan
        with self.assertRaises(ValueError):m.probability_audit(x)
    def test_null_capacity_lengths_shared_integer_rng(self):
        f,x=self.fixture(20);x[0,1:4]=1.;x[0,13:15]=3.;x[2,4:8]=2.
        r,n,a=m.generate_pair('video',17,f,x)
        for rep in range(100):
            for q in range(9):
                h,digest=m.shared_hash64('video',17,q,rep)
                self.assertEqual(digest,hashlib.sha256(f'88/video/17/{q}/{rep}'.encode()).hexdigest())
                self.assertEqual(str(h),n['shared_hash64_numerators'][rep][q])
                for arm in ('H','C'):
                    real,random=r[arm][q],n['null_'+arm][rep][q]
                    self.assertEqual(real is None,random is None)
                    if real:
                        self.assertEqual(real['sample_length'],random['sample_length']);self.assertEqual(real['frame_length'],random['frame_length'])
                        self.assertEqual(random['first_index'],h*(len(f)-real['sample_length']+1)//2**64)
                        self.assertLessEqual(random['end'],int(f[-1])+1)
        z=m.random_interval(r['C'][0],f,2**64-1)
        self.assertEqual(z['last_index'],len(f)-1)
        self.assertEqual(list(inspect.signature(m.shared_hash64).parameters),['video','pair','query','repetition'])
        self.assertNotIn('GT',inspect.signature(m.generate_pair).parameters)
    def test_repeated_GT_reuses_candidates_and_no_pooling(self):
        f=np.arange(6,dtype=np.int64)*4;hit=m.interval(f,0,0,1);miss=m.interval(f,0,4,5)
        real={'H':[hit]+[None]*8,'C':[hit]+[None]*8}
        separate=[[hit]+[None]*8]+[[miss]+[None]*8 for _ in range(99)]
        null={'null_H':separate,'null_C':copy.deepcopy(separate)}
        before=json.dumps([real,null],sort_keys=True)
        a=m.evaluate_event(dict(video_name='v',gt_index=0,begin_fid=0,end_fid=5),real,null)
        b=m.evaluate_event(dict(video_name='v',gt_index=1,begin_fid=0,end_fid=5),real,null)
        self.assertEqual(a['null_C']['expected_hit'],.01);self.assertEqual(a['null_C']['hit_count'],1)
        self.assertEqual(a['null_C'],b['null_C']);self.assertEqual(before,json.dumps([real,null],sort_keys=True))
    def test_video_equal_shared_bootstrap_and_denominators(self):
        def row(v,h,c,nh,nc):
            return dict(video_name=v,H={'hit':h},C={'hit':c},null_H={'expected_hit':nh/100,'hit_count':nh,'hits_by_repetition':[1]*nh+[0]*(100-nh)},null_C={'expected_hit':nc/100,'hit_count':nc,'hits_by_repetition':[1]*nc+[0]*(100-nc)})
        rows=[row('a',False,True,1,1) for _ in range(10)]+[row('b',True,False,1,1)]
        s,idx=m.summarize(rows)
        self.assertEqual(s['comparisons']['C_minus_H']['video_equal_difference'],0.)
        self.assertEqual(s['integer_counts']['net_gain'],9);self.assertEqual(s['integer_counts']['denominator'],11)
        self.assertEqual(s['null_expected_counts']['null_C']['expected_hit_count'],.11)
        s2,idx2=m.summarize(rows);self.assertEqual(s,s2);np.testing.assert_array_equal(idx,idx2)
        v=np.array([[0.,1.],[1.,0.]])[idx].mean(1)
        expected=np.quantile(v[:,1]-v[:,0],[.025,.975]).tolist()
        self.assertEqual(expected,s['comparisons']['C_minus_H']['ci95'])
    def test_decision_priority_and_one_sided_bounds(self):
        def comp(effect,lo,hi):return dict(video_equal_difference=effect,ci95=[lo,hi],passes_minimum_effect=effect>=.05 and lo>0)
        s=dict(video_equal_rates={'H':.1,'C':.15},comparisons={'C_minus_H':comp(.05,.01,.1),'C_minus_null_C':comp(.05,.01,.1),'H_minus_null_H':comp(.02,-.1,.1)})
        self.assertEqual(m.decide(s)['status'],'SUPPORTED')
        s['comparisons']['C_minus_H']=comp(0,-.04,.049);self.assertEqual(m.decide(s)['exit'],'A')
        s['comparisons']['C_minus_H']=comp(.02,-.04,.08);s['comparisons']['C_minus_null_C']=comp(.01,-.1,.049);s['comparisons']['H_minus_null_H']=comp(.01,-.1,.049)
        self.assertEqual(m.decide(s)['exit'],'B')
        s['comparisons']['C_minus_null_C']=comp(.01,-.1,.05);self.assertEqual(m.decide(s)['status'],'INCONCLUSIVE')
    def test_sealed_candidate_artifacts_from_synthetic_score(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);out=base/'out';out.mkdir();f,x=self.fixture();x[0,2:5]=2
            score=base/'synthetic.npz';np.savez(score,frames=f,native_masks=x)
            manifest={'score_files':{'v/0':{'path':str(score),'sha256':sha(score)}}}
            r,n,a,seal=build_library(base/'synthetic_manifest.json',manifest,{'v/0':{'origin':0,'N_valid':len(f)}},{},out)
            self.assertTrue(seal['sealed_before_GT_evaluation']);self.assertEqual(seal['candidates_sha256'],sha(out/'candidates.json'))
            self.assertEqual(len(r['v/0']['H']),9);self.assertEqual(len(n['v/0']['null_H']),100)

if __name__=='__main__':unittest.main()
