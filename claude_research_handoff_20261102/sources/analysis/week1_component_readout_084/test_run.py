import unittest
from run import compare, records


class PairedTests(unittest.TestCase):
    def test_gain_and_loss_are_both_preserved(self):
        group=[dict(video_name='v',l=a,r=b) for a,b in ((.5,.6),(.5,.4),(.4,.5),(.4,.4))]
        out=compare(group,'l','r',.5)
        self.assertEqual([out[k] for k in ('both_hit','hull_only_loss','component_only_gain','neither')],[1,1,1,1])
        self.assertEqual(out['net_gain'],0)
        self.assertEqual(out['N'],4)

    def test_zero_prediction_score_remains_in_denominator(self):
        out=compare([dict(video_name='v',l=0,r=0)],'l','r',.5)
        self.assertEqual(out['neither'],1)
        self.assertEqual(out['hull_mean_tiou'],0)

    def test_empty_group_has_undefined_rates(self):
        out=compare([],'l','r',.5)
        self.assertIsNone(out['net_gain_fraction'])
        self.assertIsNone(out['hull_mean_tiou'])

    def test_duplicate_input_rejected(self):
        with self.assertRaises(ValueError): records(b'video_name,gt_index\nv,0\nv,0\n')


if __name__=='__main__': unittest.main()
