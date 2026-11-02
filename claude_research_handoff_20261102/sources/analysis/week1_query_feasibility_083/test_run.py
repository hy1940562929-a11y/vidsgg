import unittest
from run import stage, STAGES, summarize, index, boolean


class DecompositionTests(unittest.TestCase):
    def example(self, a, b, c, d, e, f):
        return dict(default_min5=a, default_unfiltered=b, grid_min5=c,
                    grid_unfiltered=d, sigmoid=e, raw_logit=f)

    def test_all_seven_mutually_exclusive_cases(self):
        cases = [(.5,.5,.8,.9,.6,.7), (.1,.1,.2,.3,.1,.1),
                 (.1,.2,.4,.8,.2,.3), (.1,.7,.8,.9,.2,.3),
                 (.1,.2,.8,.9,.6,.7), (.1,.2,.8,.9,.3,.6),
                 (.1,.2,.8,.9,.3,.4)]
        self.assertEqual([stage(self.example(*x),.5) for x in cases], list(STAGES))

    def test_threshold_equality_is_hit(self):
        self.assertEqual(stage(self.example(.5,.5,.5,.5,.5,.5),.5), 'default_hit')

    def test_default_min5_priority_retains_overlap(self):
        self.assertEqual(stage(self.example(.1,.7,.9,.9,.8,.8),.5), 'default_min5_rejected')

    def test_impossible_hierarchy_refused(self):
        with self.assertRaises(ValueError): stage(self.example(.8,.8,.7,.9,.8,.8),.5)

    def test_bad_value_refused(self):
        with self.assertRaises(ValueError): stage(self.example(float('nan'),.2,.8,.9,.3,.4),.5)

    def test_duplicate_gt_refused(self):
        with self.assertRaises(ValueError): index([dict(video_name='v',gt_index=1)]*2)

    def test_boolean_not_arbitrary_truthiness(self):
        self.assertFalse(boolean('False'))
        with self.assertRaises(ValueError): boolean('NA')

    def test_empty_cohort_is_na(self):
        r=summarize([])
        self.assertEqual(len(r),24)
        self.assertTrue(all(x['N']==0 and x['default_hit_fraction'] is None for x in r))


if __name__ == '__main__': unittest.main()
