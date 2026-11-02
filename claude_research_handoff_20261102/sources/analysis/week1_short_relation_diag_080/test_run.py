import unittest
from run import classify, companion_flags, interval_iou, oracle_summary, same_optional_number

class ProtocolTests(unittest.TestCase):
    def test_optional_support_join(self):
        self.assertTrue(same_optional_number(None,None))
        self.assertFalse(same_optional_number(None,0))
        self.assertFalse(same_optional_number(1,None))
        self.assertTrue(same_optional_number(1,1+1e-9))

    def test_threshold_equality_and_missing(self):
        self.assertEqual(classify(3,10,True),'L-L')
        self.assertEqual(classify(2.99,10,False),'S-L-solo')
        self.assertEqual(classify(2.99,9.99,True),'S-S')
        self.assertEqual(classify(2.99,10,True),'S-L-comp')
        for co in (None,0,2,float('nan')):
            self.assertEqual(classify(3,co,False),'UNKNOWN')

    def test_companion_identity_overlap_and_duration(self):
        target=dict(video_name='a',gt_index=1,subject_tid=2,object_tid=3,
                    predicate='watch',relation_sec=1,begin_fid=100,end_fid=130)
        base=dict(target,gt_index=2,predicate='next_to',relation_sec=10,begin_fid=0,end_fid=300)
        _,f=companion_flags(target,[target,base])
        self.assertTrue(all(f.values()))
        for changes in ({'video_name':'b'},{'subject_tid':3,'object_tid':2},{'relation_sec':2.99}):
            _,f=companion_flags(target,[dict(base,**changes)])
            self.assertFalse(any(f.values()))
        for changes in ({'predicate':'watch'},{'begin_fid':130,'end_fid':430}):
            _,f=companion_flags(target,[dict(base,**changes)])
            self.assertTrue(f['literal_other_ge3_anywhere'])
            self.assertFalse(f['overlap_other_pred_ge3'])
        _,f=companion_flags(target,[dict(base,begin_fid=115,end_fid=415)])
        self.assertTrue(f['overlap_other_pred_ge3_half'])
        _,f=companion_flags(target,[dict(base,begin_fid=116,end_fid=416)])
        self.assertFalse(f['overlap_other_pred_ge3_half'])

    def test_half_open_intervals(self):
        self.assertEqual(interval_iou(0,5,5,10),0)
        self.assertEqual(interval_iou(0,10,0,5),.5)
        self.assertEqual(interval_iou(0,5,0,5),1)

    def test_oracle_has_no_duration_or_deployed_availability(self):
        row=dict(source_scope='GT27',slice_definition='literal',analysis_population='all',
                 slice='S-S',video_name='v',oracle_tiou=.5)
        summary=oracle_summary([row])
        ss=next(r for r in summary if r['slice']=='S-S')
        self.assertEqual(ss['miss_count_tiou05'],0)
        self.assertEqual(ss['miss_count_tiou07'],1)
        self.assertEqual(ss['mean_tiou'],.5)
        self.assertIsNone(ss['duration_bias'])
        self.assertNotIn('pred_available',ss)

if __name__=='__main__':unittest.main()
