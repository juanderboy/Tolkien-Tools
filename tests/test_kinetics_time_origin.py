import io
import sys
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'TolkienTools/kinetics'))
from kinet_common import Experiment
from kinet_cli import build_parser
from kinet_preprocessing import apply_preprocessing_choices, preprocess_experiment
from kinet_models import concentration_profile_mbfe3_sulfide_binding_autocatalytic as profile


class TimeOriginTests(unittest.TestCase):
    def setUp(self):
        self.raw=Experiment(t=np.array([20.4,20.9,21.4,21.9]),
            wavelength=np.array([400.,500.,750.,800.,850.]),
            absorbance=np.arange(20.).reshape(5,4))

    def apply(self,t0,drop=None,reference=None):
        args=build_parser().parse_args([])
        corrected,_,_=apply_preprocessing_choices(args,self.raw,drop or [],t0,None,
            'none',None,20,(400.,850.),initial_spectrum_index=reference)
        return args,corrected

    def test_between_frames_t0_is_used_exactly(self):
        args,corrected=self.apply(20.41)
        np.testing.assert_allclose(corrected.t,[.49,.99,1.49],atol=1e-14)
        self.assertEqual(args.reaction_time_origin,20.41)
        np.testing.assert_array_equal(corrected.absorbance,self.raw.absorbance[:,1:])

    def test_dropping_frame_at_t0_does_not_shift_origin(self):
        args,corrected=self.apply(20.4,drop=[0])
        np.testing.assert_allclose(corrected.t,[.5,1.,1.5])
        self.assertEqual(args.reaction_time_origin,20.4)
        # This must be the original model trajectory after omitting its t=0,
        # not a new reaction initialized at the first remaining observation.
        full=profile(self.raw.t-20.4,.5,.1,.001,.004)
        np.testing.assert_allclose(profile(corrected.t,.5,.1,.001,.004),full[:,1:],atol=1e-14)

    def test_exact_frame_and_before_first_measurement(self):
        _,c=self.apply(20.9)
        np.testing.assert_allclose(c.t,[0,.5,1.])
        _,c=self.apply(20.)
        np.testing.assert_allclose(c.t,[.4,.9,1.4,1.9])

    def test_default_and_explicit_zero(self):
        _,c=self.apply(None,drop=[0])
        np.testing.assert_allclose(c.t,[0,.5,1.])
        _,c=self.apply(0.)
        np.testing.assert_array_equal(c.t,self.raw.t)
        _,c=self.apply(None,reference=0)
        np.testing.assert_allclose(c.t,[0,.5,1.,1.5])
        _,c=self.apply(20.9,reference=0)
        np.testing.assert_allclose(c.t,[0,.5,1.])

    def test_interactive_question_count_unchanged_and_exact_t0(self):
        args=build_parser().parse_args([]);args.lambda_min,args.lambda_max=400.,500.
        replies=['','20.41','','no','no','750-850','400-500','1','yes']
        with patch('builtins.input',side_effect=replies) as ask,redirect_stdout(io.StringIO()), \
             patch('kinet_plotting.plot_preprocessed_experiment_preview'):
            result=preprocess_experiment(args,self.raw,initial_spectrum_label='A',
                final_spectrum_label='B',preferred_baseline_region=(750.,850.))
        self.assertEqual(ask.call_count,9)
        np.testing.assert_allclose(result[0].t,[.49,.99,1.49])
        self.assertEqual(args.reaction_time_origin,20.41)
        self.assertIn('use exactly t = t_original - t0',ask.call_args_list[1].args[0])

if __name__=='__main__':unittest.main()
