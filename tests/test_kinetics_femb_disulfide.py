import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'TolkienTools/kinetics'))
from kinet_common import Experiment, SPECIAL_MODEL_LABELS
from kinet_cli import build_parser, kinetics_menu, apply_model_default_wavelength_range
from kinet_femb_disulfide import fractions, spectral_fractions, select_candidate, MODEL
from kinet_fitting import fit_model_with_auto_k_max
from kinet_models import concentration_profile_a_rev_b_to_c


class FeMbDisulfideTests(unittest.TestCase):
    def experiment(self):
        wave=np.arange(400.,451.)
        spectra=np.column_stack([np.exp(-((wave-peak)/width)**2)
            for peak,width in [(409,7),(428,6),(434,6)]])
        spectra[:,1] *= .65
        t=np.unique(np.r_[np.linspace(0,10,21),np.linspace(10,250,35)])
        f=fractions(t,.4,.01,.1)
        return Experiment(t=t,wavelength=wave,absorbance=spectra@f),spectra

    def test_profile_matches_general_reversible_model(self):
        t=np.array([.5,1.,5.,20.,100.])
        np.testing.assert_allclose(fractions(t,.4,.01,.1),
            concentration_profile_a_rev_b_to_c(t,.4,.1,.01),atol=2e-14)

    def test_simplex_concentrations_and_mass_conservation(self):
        experiment,spectra=self.experiment()
        estimated=spectral_fractions(spectra,experiment.absorbance)
        np.testing.assert_allclose(estimated,fractions(experiment.t,.4,.01,.1),atol=1e-12)
        np.testing.assert_allclose(estimated.sum(axis=0),1)
        self.assertGreaterEqual(estimated.min(),-1e-14)

    def test_selection_prioritizes_profiles_only_among_good_absorption_and_peaks(self):
        candidates=[dict(rmse_mAU=1.,profile_score=.02,chemical_pass=True),
            dict(rmse_mAU=1.0005,profile_score=.01,chemical_pass=True),
            dict(rmse_mAU=1.01,profile_score=0.,chemical_pass=True),
            dict(rmse_mAU=1.,profile_score=0.,chemical_pass=False)]
        chosen,minimum,n,nchem=select_candidate(candidates)
        self.assertIs(chosen,candidates[1]);self.assertEqual(nchem,2)
        chosen,_,_,nchem=select_candidate([candidates[-1]])
        self.assertIs(chosen,candidates[-1]);self.assertEqual(nchem,0)

    def test_menu_and_system_defaults_without_additional_questions(self):
        args=build_parser().parse_args(['file.KD'])
        index=7
        with patch('builtins.input',side_effect=['4',str(index)]),redirect_stdout(io.StringIO()):
            self.assertTrue(kinetics_menu(args))
        self.assertEqual(args.model,MODEL)
        apply_model_default_wavelength_range(args,MODEL)
        self.assertEqual((args.lambda_min,args.lambda_max),(400.,650.))

    def test_real_fit_is_physical_and_exports_selection_data(self):
        experiment,_=self.experiment()
        result,bounds=fit_model_with_auto_k_max(MODEL,experiment,c0=5e-6,
            n_components=3,method='nnls',k_bounds=(1e-8,.1),auto_expand=True,
            expand_factor=10,max_expand_steps=4)
        self.assertEqual(result.model,MODEL)
        self.assertEqual(bounds,(1e-7,5.))
        d=result.selection_diagnostics
        self.assertGreater(d['n_candidates'],4)
        self.assertTrue(d['chemical_filter_passed'])
        self.assertGreaterEqual(d['selected']['B_peak_over_C_peak'],.5)
        self.assertLessEqual(d['selected']['B_peak_over_C_peak'],.95)
        self.assertLessEqual(d['selected']['rmse_mAU'],d['minimum_rmse_mAU']*1.001+1e-9)
        np.testing.assert_allclose(result.c.sum(axis=0),5e-6,atol=1e-14)
        np.testing.assert_allclose(result.estimated_concentrations.sum(axis=0),5e-6,atol=1e-14)
        np.testing.assert_allclose(result.spectra@result.c,result.absorbance_calc,atol=1e-13)
        self.assertLess(np.sqrt(np.mean(result.residuals**2)),1e-5)

    def test_known_reference_amplitude_bounds_are_respected(self):
        experiment,spectra=self.experiment()
        c0=5e-6
        from kinet_femb_disulfide import fit_femb_disulfide
        labels=('MbFeIII','MbFeIII-S2','MbFeII')
        result=fit_femb_disulfide(experiment,c0=c0,known_spectra=spectra/c0,
            known_species=(labels[1],),known_spectrum_scale_bounds={labels[1]:(.7,1.3)})
        self.assertLess(abs(result.params['k_slow']/.01-1),.01)
        self.assertGreaterEqual(result.known_spectrum_scales[labels[1]],.7)
        self.assertLessEqual(result.known_spectrum_scales[labels[1]],1.3)
        np.testing.assert_allclose(result.spectra[:,0],spectra[:,0]/c0,atol=1e-10)
        np.testing.assert_allclose(result.spectra[:,2],spectra[:,2]/c0,atol=1e-10)

if __name__=='__main__':unittest.main()
