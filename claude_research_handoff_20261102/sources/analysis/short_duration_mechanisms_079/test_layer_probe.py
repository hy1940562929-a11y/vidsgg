import unittest
import numpy as np
from layer_math import position_features, annotation_union, components, iou, average_precision, video_bootstrap


class LayerMathTests(unittest.TestCase):
    def test_union_halfopen_multiple_same_predicate(self):
        y=annotation_union([dict(predicate_id=2,begin=3,end=9),dict(predicate_id=2,begin=12,end=17)],0,4,6)
        np.testing.assert_array_equal(y[1],[False,True,True,True,True,False])
        self.assertEqual(int(y.sum()),4)

    def test_components_use_true_frame_coordinates_and_min5(self):
        self.assertEqual(components([1,1,0,1,0,1,1],2,4),[(2,7),(22,27)])
        self.assertEqual(components([1],0,4),[])

    def test_position_has_equal_capacity_without_annotation_inputs(self):
        x=position_features(13)
        self.assertEqual(x.shape,(256,13));self.assertTrue((x[9:]==0).all())

    def test_ap_ties_do_not_reward_label_order(self):
        self.assertEqual(average_precision([1,0],[.5,.5]),.5)
        self.assertEqual(average_precision([0,1],[.5,.5]),.5)
        self.assertEqual(average_precision([1,0],[.9,.2]),1.)
        self.assertIsNone(average_precision([0,0],[.1,.2]))

    def test_iou_halfopen(self):
        self.assertEqual(iou((0,5),(5,10)),0)
        self.assertAlmostEqual(iou((0,5),(0,10)),.5)

    def test_bootstrap_reports_losses_and_denominator(self):
        rows=[dict(video_name='a',arms=dict(fine=dict(hit=True),fused=dict(hit=False))),
              dict(video_name='b',arms=dict(fine=dict(hit=False),fused=dict(hit=True)))]
        value=video_bootstrap(rows,'fine','fused',100,79)
        self.assertEqual((value['new'],value['lost'],value['net'],value['N_GT']),(1,1,0,2))

    def test_train_statistics_are_train_only(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable; mandatory on server')
        from layer_probe import fit_train_statistics
        sample=dict(N_valid=3,targets=torch.zeros(50,3),features={a:torch.full((256,3),2.) for a in ('fine','fused','position')})
        sample['targets'][0,0]=1
        stats=fit_train_statistics(torch,[[sample]])
        self.assertEqual(stats['N_frames'],3);self.assertEqual(stats['positive_weight'][0].item(),2.)
        self.assertTrue(torch.equal(stats['means']['fine'],torch.full((256,),2.)))
        self.assertEqual(stats['supported'].sum().item(),1)


if __name__=='__main__':unittest.main()
