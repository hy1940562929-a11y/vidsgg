"""Synthetic correctness tests only; these are not research results."""
import unittest
import numpy as np
from sampling_math import (sampled_frames, interpolate_features, visual_interpolation,
                           envelope_intervals, interval_iou, duration_bin, paired_bootstrap,
                           forward_hold_binary_iou)


class F:
    @staticmethod
    def interpolate(y, scale_factor, mode):
        assert mode == 'nearest'
        return np.repeat(y, int(scale_factor), axis=-1)


class FPN1D_Fuse:
    """Tiny NumPy stand-in tests runtime edit / restoration, not model quality."""
    scale_factor = 2

    def forward(self, inputs, fpn_masks=None):
        for idx in range(len(inputs)-1, -1, -1):
            if idx == len(inputs)-1:
                y = inputs[idx]
            else:
                cur_fpn = inputs[idx]
                y = cur_fpn + F.interpolate(y, scale_factor=self.scale_factor, mode="nearest")
        return y


class SamplingMathTests(unittest.TestCase):
    def test_frame_origin_and_half_open_end(self):
        np.testing.assert_array_equal(sampled_frames(101, 114, 4, 2), [103,107,111])
        self.assertEqual(envelope_intervals(np.array([[True,False,True]]), [103,107,111]), [(103,112)])
        self.assertAlmostEqual(interval_iou((103,112),(103,112)), 1.)

    def test_fixed_physical_minimum(self):
        self.assertEqual(envelope_intervals(np.ones((1,2),dtype=bool), [10,12]), [None])
        self.assertEqual(envelope_intervals(np.ones((1,3),dtype=bool), [10,12,14]), [(10,15)])
        self.assertEqual(envelope_intervals(np.zeros((1,3),dtype=bool), [10,12,14]), [None])
        with self.assertRaises(TypeError):
            envelope_intervals(np.ones((1,3),dtype=np.float32),[10,12,14])

    def test_binary_occupancy_is_not_hull_and_preserves_gt_extent(self):
        mask=np.array([[True,False,True]])
        self.assertEqual(envelope_intervals(mask,[0,4,8]),[(0,9)])
        self.assertAlmostEqual(forward_hold_binary_iou(mask,[0,4,8],10,(0,10))[0],.6)
        self.assertAlmostEqual(forward_hold_binary_iou(mask,[0,4,8],10,(0,12))[0],.5)
        # Unsampled prefix is not silently filled; short mask has no 5-frame
        # candidate but does retain its distinct raw occupancy measurement.
        self.assertAlmostEqual(forward_hold_binary_iou(np.array([[True]]),[3],5,(0,5))[0],.4)

    def test_interpolation_uses_frames_not_tensor_length(self):
        y = interpolate_features(np.array([[0,4,8]], dtype=np.float32), [10,14,18], [10,12,14,16,18,20])
        np.testing.assert_array_equal(y, [[0,2,4,6,8,8]])

    def test_visual_only_preserves_dense_geometry(self):
        src = np.tile(np.array([0,4,8], np.float32), (2069,1))
        dense = np.full((2069,5), 37, np.float32)
        out = visual_interpolation(src,dense,[10,14,18],[10,12,14,16,18])
        np.testing.assert_array_equal(out[2048:], dense[2048:])
        np.testing.assert_array_equal(out[0], [0,2,4,6,8])
        with self.assertRaises(ValueError):
            visual_interpolation(src[:2048],dense,[10,14,18],[10,12,14,16,18])

    def test_phase_filter_not_a_new_denominator(self):
        self.assertEqual(len(sampled_frames(100,105,4,0)), 2)
        self.assertEqual(len(sampled_frames(100,105,4,3)), 1)
        # GT row remains in accounting, although native loader filters phase3.
        rows=[dict(video='v',a={'hit':True},b={'hit':False})]
        out=paired_bootstrap(rows,'a','b','hit',20)
        self.assertEqual(out['GT'],1);self.assertEqual(out['net_pp'],-100.)

    def test_bins(self):
        self.assertEqual([duration_bin(x) for x in [.9,1,2.9,3,10]],['sub1','1to3','1to3','3to10','10plus'])

    def test_fpn_gate_patch_is_exact_and_restores_on_error(self):
        from sampling_probe import coarse_gate
        neck=FPN1D_Fuse();inputs=[np.array([2.,2.]),np.array([10.])]
        original=neck.forward(inputs)
        with coarse_gate(neck,1.):np.testing.assert_array_equal(neck.forward(inputs),original)
        with coarse_gate(neck,.5):np.testing.assert_array_equal(neck.forward(inputs),[7.,7.])
        with coarse_gate(neck,0.):np.testing.assert_array_equal(neck.forward(inputs),[2.,2.])
        try:
            with coarse_gate(neck,0.):raise RuntimeError('synthetic interruption')
        except RuntimeError:pass
        np.testing.assert_array_equal(neck.forward(inputs),original)
        self.assertFalse(hasattr(neck,'_probe079_coarse_gate'))

    def test_common_support_excludes_pair_availability_from_visual_effect(self):
        from sampling_probe import blank_arm,update_gt,finish_rows,summarize
        arms=['baseline','dense2','visual_interp2','dense2_common','visual_interp2_common']
        row=dict(video='v',subject_tid=1,object_tid=2,predicate='watch',start=0,end=5,
                 duration_seconds=1.,duration_bin='1to3',long_copresence_short=False,
                 common_visual_pair_segments_overlapping_gt=0)
        for arm in arms:row[arm]=blank_arm()
        # This pair has no baseline interpolation source, so only the
        # unrestricted dense arm is updated. All five GT denominators remain.
        update_gt([row],'dense2',(1,2,0,5),np.ones((1,3)),np.array([[0.,2.]]),
                  np.ones((1,3),dtype=bool),np.array([0,2,4]),{'watch':1})
        finish_rows([row],arms)
        out=summarize([row],arms,{'bootstrap_reps':20},['v'],[],True,0.)
        self.assertEqual(out['results']['all']['comparisons']['dense2__minus__visual_interp2']['any_query_hit']['net'],1)
        self.assertEqual(out['results']['all']['comparisons']['dense2_common__minus__visual_interp2_common']['any_query_hit']['net'],0)
        self.assertEqual(out['results']['all_common_visual_support']['N_GT'],0)

    def test_gate_requires_both_values_and_ten_support_videos(self):
        from sampling_probe import advance_decisions
        good={'videos':10,'net_pp':2.,'ci95_pp':[.1,3.]}
        summary={'complete':True,'results':{
            'short_lt3_common_visual_support':{'comparisons':{'dense2_common__minus__visual_interp2_common':{'any_query_hit':dict(good)}}},
            'short_lt3':{'comparisons':{name:{'any_query_hit':dict(good)} for name in (
                'dense2__minus__baseline','coarse_gate_half__minus__baseline','coarse_gate_zero__minus__baseline')}}}}
        self.assertEqual(advance_decisions(summary,{})['coarse_fusion_path']['status'],'STRONG_DIAGNOSTIC_SUPPORT')
        summary['results']['short_lt3']['comparisons']['coarse_gate_zero__minus__baseline']['any_query_hit']['videos']=9
        out=advance_decisions(summary,{})
        self.assertEqual(out['coarse_fusion_path']['status'],'INCONCLUSIVE_NO_EXPANSION')
        self.assertFalse(out['automatically_expand'])
        summary['complete']=False
        self.assertEqual(advance_decisions(summary,{})['added_visual_observation']['status'],'INCONCLUSIVE_NO_EXPANSION')

    def test_torch_native_sigmoid_binary_near_zero(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable locally; server no-op also checks actual model binary outputs')
        values=torch.tensor([[1e-9,-1e-9,1.]],dtype=torch.float32)
        binary=(values.sigmoid()>.5).numpy()
        self.assertFalse(bool(binary[0,0]))  # positive logit can still round to sigmoid .5
        self.assertEqual(envelope_intervals(binary,[0,4,8]),[None])


if __name__ == '__main__': unittest.main()
