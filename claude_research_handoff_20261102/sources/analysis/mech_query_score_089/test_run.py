"""Synthetic-only tests. Never opens research inputs."""
import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np
import query_math as m
import run


def logits():
    z = np.full((9,51), -10., dtype=np.float32)
    z[:,2] = 1
    return z

def interval(q=0, a=0, b=9):
    return dict(query=q, first_index=0,last_index=2,start=a,end=b,sample_length=3,frame_length=b-a)

def event(video='a', f=0, t=1, exists=True):
    r = dict(video_name=video, T_exists=exists, score_drift=False)
    for arm in ('H','C'):
        for label, value in (('F',f),('T',t),('U',max(f,t))):
            r[label+'_'+arm+'_hit'] = value
            r[label+'_'+arm+'_candidate_count'] = 1
        r[arm+'_gain'] = int(t and not f);r[arm+'_loss'] = int(f and not t)
    r['U_union_hit'] = max(f,t);r['U_union_candidate_count'] = 2
    return r

class QueryTests(unittest.TestCase):
    def test_filter_softmax_not_raw_logit(self):
        z = logits(); z[0,1] = 10; z[0,2] = 11; z[1,1] = 3; z[1,2] = 0
        s = m.select_queries(z,1,1)
        self.assertEqual(s['current_target_argmax'],1)
        self.assertEqual(s['T'],1)
        self.assertNotIn(0,s['eligible_queries'])

    def test_filter_changes_query(self):
        z = logits(); z[0,1] = 4;z[0,2] = 4.01
        z[1,1] = 3;z[1,0] = 8
        s = m.select_queries(z,1,0)
        self.assertEqual(s['current_target_argmax'],0)
        self.assertEqual(s['T'],1)
        self.assertTrue(s['per_query'][1]['background_dominates_foreground'])

    def test_no_eligible_kept(self):
        s = m.select_queries(logits(),1,0)
        self.assertIsNone(s['T']);self.assertFalse(s['T_exists'])
        pair = dict(H=[interval()]+[None]*8,C=[interval()]+[None]*8)
        r = m.evaluate_event(s,pair,0,9)
        self.assertEqual(r['T_C_hit'],0);self.assertEqual(r['F_C_hit'],1)

    def test_ties_lowest_index_and_class(self):
        z = np.zeros((9,51))
        s = m.select_queries(z,1,0)
        self.assertEqual(s['T'],0);self.assertEqual(s['current_target_argmax'],0)
        self.assertEqual(s['per_query'][0]['foreground_top1'],1)
        self.assertEqual(s['per_query'][0]['target_query_rank'],1)
        self.assertFalse(m.select_queries(z,2,0)['T_exists'])

    def test_empty_no_fallback_and_same_selector(self):
        z = logits();z[0,1] = 4;z[1,1] = 3
        s = m.select_queries(z,1,0)
        p = dict(H=[None,interval(1)]+[None]*7,C=[None,interval(1)]+[None]*7)
        a = m.evaluate_event(s,p,0,9);b = m.evaluate_event(s,p,20,29)
        self.assertEqual(a['T_C_candidate_count'],0);self.assertEqual(a['U_C_hit'],1)
        self.assertEqual(a['T_C_query'],b['T_C_query'])
        self.assertEqual(list(inspect.signature(m.select_queries).parameters),['class_logits','target_pid','legacy_query'])

    def test_common_support_denominator(self):
        rows = [event('a',0,0,True),event('a',0,1,True),event('b',0,0,False)]
        full,_ = m.summarize(rows)
        common,ix = m.summarize([r for r in rows if r['T_exists']])
        self.assertEqual((full['N_GT'],common['N_GT'],common['N_videos']),(3,2,1))
        self.assertEqual(common['event_rates']['T_C'],.5)
        self.assertEqual(ix.shape,(10000,1))

    def test_video_equal_paired_shared_bootstrap(self):
        rows = [event('a',0,1)]*9+[event('b',1,0)]
        s,ix = m.summarize(rows)
        self.assertEqual(s['comparisons']['C']['mean'],0)
        np.testing.assert_array_equal(ix,np.random.default_rng(89).integers(0,2,(10000,2)))
        expected = np.percentile(np.array([1.,-1.])[ix].mean(1),[2.5,97.5])
        np.testing.assert_array_equal(s['comparisons']['C']['ci95'],expected)
        self.assertEqual(s['comparisons']['H'],s['comparisons']['C'])

    def test_decisions_and_drift_priority(self):
        rows=[event(str(i),0,1) for i in range(10)]
        s,_=m.summarize(rows)
        self.assertEqual(m.decide(s,s,0),'SUPPORTED_QUERY_RESELECTION')
        self.assertEqual(m.decide(s,s,1),'INCONCLUSIVE_SCORE_DRIFT')
        rows=[event(str(i//4),0,0) for i in range(40)]
        s,_=m.summarize(rows)
        self.assertEqual(m.decide(s,s,0),'SUPPORTED_RESIDUAL_INTERVAL_DEFICIT')
        small,_=m.summarize(rows[:39])
        self.assertEqual(m.decide(s,small,0),'INCONCLUSIVE')
        empty,_=m.summarize([])
        self.assertEqual(m.decide(s,empty,0),'INCONCLUSIVE')

    def test_shape_and_nonfinite(self):
        for z in (np.zeros((8,51)),np.full((9,51),np.nan)):
            with self.assertRaises(ValueError):m.select_queries(z,1,0)

    def test_candidate_mapping_and_corruption(self):
        pair=dict(N_valid=3,origin=0,frames=[0,4,8],H=[interval()]+[None]*8,C=[interval()]+[None]*8)
        run.audit_candidates(pair,dict(N_valid=3,origin=0),np.array([0,4,8]))
        pair['C'][0]['end']=12
        with self.assertRaises(ValueError):run.audit_candidates(pair,dict(N_valid=3,origin=0),np.array([0,4,8]))

    def test_source_materialization_sha(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); material=base/'cached';material.write_bytes(b'synthetic')
            ref=dict(path=str(base/'missing'), materialized_path=str(material),sha256=run.sha(material))
            self.assertEqual(run.reference(base,ref,{}),material.resolve())
            ref['sha256']='0'*64
            with self.assertRaises(ValueError):run.reference(base,ref,{})

if __name__=='__main__':unittest.main()
