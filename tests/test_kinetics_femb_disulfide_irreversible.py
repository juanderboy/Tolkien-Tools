import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'TolkienTools'/'kinetics'))
from kinet_common import Experiment, MODEL_SPECIES
from kinet_cli import build_parser, kinetics_menu, apply_model_default_wavelength_range
from kinet_femb_disulfide import fractions, IRREVERSIBLE_MODEL
from kinet_fitting import fit_model_with_auto_k_max
from kinet_models import concentration_profile_a_to_b_to_c


class IrreversibleDisulfideTests(unittest.TestCase):
    def test_zero_off_matches_consecutive_model_including_equal_rates(self):
        times=np.r_[0.,.1,1.,10.,100.]
        for kon,kred in ((.2,.015),(.015,.2),(.1,.1)):
            c=fractions(times,kon,kred,0.)
            np.testing.assert_allclose(c,concentration_profile_a_to_b_to_c(times,kon,kred),atol=1e-14)
            np.testing.assert_allclose(c.sum(axis=0),1.,atol=1e-14)
            self.assertGreaterEqual(c.min(),-1e-14)

    def test_menu_position_between_postbinding_and_reversible(self):
        args=build_parser().parse_args(['experiment.KD'])
        with patch('builtins.input',side_effect=['4','6']),redirect_stdout(io.StringIO()):
            self.assertTrue(kinetics_menu(args))
        self.assertEqual(args.model,IRREVERSIBLE_MODEL)
        apply_model_default_wavelength_range(args,args.model)
        self.assertEqual((args.lambda_min,args.lambda_max),(400.,650.))
        import kinet_cli
        models=kinet_cli.SPECIAL_SPECTRAL_MODELS
        i=models.index(IRREVERSIBLE_MODEL)
        self.assertEqual(models[i-1],'mbfe3_hss_no_binding')
        self.assertEqual(models[i+1],'femb_disulfide_reversible')

    def test_fit_uses_only_two_rates_and_respects_known_spectra(self):
        wave=np.arange(400.,451.)
        spectra=np.column_stack([np.exp(-((wave-p)/w)**2)*a
            for p,w,a in [(409,7,1.),(432,6,.75),(434,6,1.)]])
        times=np.unique(np.r_[np.linspace(0,10,31),np.linspace(10,400,40)])
        observed=spectra@fractions(times,.2,.015,0.)
        experiment=Experiment(times,wave,observed)
        result,bounds=fit_model_with_auto_k_max(IRREVERSIBLE_MODEL,experiment,
            c0=5e-6,n_components=3,method='nnls',k_bounds=(1e-8,.1),
            auto_expand=True,expand_factor=10,max_expand_steps=4,
            known_spectra=spectra/5e-6)
        self.assertEqual(result.model,IRREVERSIBLE_MODEL)
        self.assertEqual(result.species_labels,MODEL_SPECIES[IRREVERSIBLE_MODEL])
        self.assertEqual(set(result.params),{'k_on','k_slow'})
        self.assertEqual(result.selection_diagnostics['fixed_parameters'],{'k_off':0.})
        self.assertNotIn('k_off',result.selection_diagnostics['rate_bounds'])
        self.assertEqual(result.selection_diagnostics['peak_windows_nm'][1],(427.,433.))
        self.assertTrue(result.selection_diagnostics['chemical_filter_passed'])
        self.assertLess(abs(result.params['k_slow']/.015-1),.002)
        self.assertLess(abs(result.params['k_on']/.2-1),.002)
        self.assertTrue(all(r['source']=='multistart' for r in result.selection_diagnostics['candidates']))
        np.testing.assert_allclose(result.spectra,spectra/5e-6,rtol=1e-11,atol=1e-9)
        np.testing.assert_allclose(result.spectra@result.c,result.absorbance_calc,atol=1e-13)
        np.testing.assert_allclose(result.c.sum(axis=0),5e-6,atol=1e-14)
        self.assertLess(np.sqrt(np.mean(result.residuals**2)),1e-5)

if __name__=='__main__':unittest.main()
