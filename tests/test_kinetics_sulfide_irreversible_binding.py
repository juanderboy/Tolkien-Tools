import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'TolkienTools/kinetics'))
from kinet_common import Experiment
from kinet_cli import build_parser, endpoint_spectrum_menu_labels, kinetics_menu
from kinet_fitting import fit_model
from kinet_models import concentration_profile_mbfe3_sulfide_binding_autocatalytic as profile

MODEL = 'mbfe3_sulfide_irreversible_binding_autocatalytic'


class IrreversibleBindingTests(unittest.TestCase):
    def test_zero_off_has_exponential_reactant_and_conserves_mass(self):
        t = np.linspace(0, 100, 101)
        c = profile(t, .3, 0., .015, .05, c0=2.)
        np.testing.assert_allclose(c[0], 2*np.exp(-.3*t), atol=2e-9)
        np.testing.assert_allclose(c.sum(axis=0), 2., atol=1e-12)
        self.assertGreaterEqual(c.min(), -1e-12)
        with self.assertRaises(ValueError):
            profile(t, .3, -.01, .015, .05)

    def test_menu_cli_and_endpoints(self):
        args = build_parser().parse_args(['--model', MODEL])
        with patch('builtins.input', side_effect=['4', '3', 'data.KD']), redirect_stdout(io.StringIO()):
            self.assertTrue(kinetics_menu(args))
        self.assertEqual(args.model, MODEL)
        self.assertEqual(endpoint_spectrum_menu_labels(MODEL, True, None), ('MbFeIII', 'MbFeII'))

    def test_dispatch_optimizes_only_three_rates_and_reconstructs_data(self):
        t = np.linspace(0, 100, 61)
        rates = dict(k_on=.3, k_slow=.015, k_cat=.05)
        c = profile(t, rates['k_on'], 0., rates['k_slow'], rates['k_cat'])
        spectra = np.array([[.9,.3,.2],[.4,.8,.5],[.2,.4,.9],[.5,.2,.6]])
        experiment = Experiment(t=t, wavelength=np.array([409.,428.,434.,550.]), absorbance=spectra@c)
        def optimize(objective, names, *args, **kwargs):
            self.assertEqual(names, ('k_on','k_slow','k_cat'))
            self.assertNotIn('k_off', kwargs['parameter_bounds'])
            self.assertNotIn('k_off', kwargs['initial_parameters'])
            self.assertLess(objective(rates), 1e-12)
            return rates
        with patch('kinet_fitting.optimize_kinetic_parameters', side_effect=optimize):
            result = fit_model(MODEL, experiment, c0=1., n_components=3, method="nnls", k_bounds=(1e-4,1.))
        self.assertEqual(result.model, MODEL)
        self.assertEqual(result.params, rates)
        np.testing.assert_allclose(result.absorbance_calc, experiment.absorbance, atol=1e-12)
        # Exercise the real optimizer too; synthetic endpoints at these times
        # are not fixed to avoid imposing a pure final species prematurely.
        fitted = fit_model(MODEL, experiment, c0=1., n_components=3,
                           method="nnls", k_bounds=(1e-4, 1.))
        self.assertEqual(set(fitted.params), set(rates))
        self.assertLess(np.sqrt(np.mean(fitted.residuals**2)), 1e-4)



if __name__ == '__main__':
    unittest.main()
