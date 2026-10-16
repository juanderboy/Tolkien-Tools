import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'TolkienTools'/'kinetics'))
from kinet_activation import estimate_uncertainty, evaluate, R

class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([-7., -4., 115., 27., 24., 7.55])
        grid = [(t, ph) for t in (285., 295., 305., 315.) for ph in (6., 7., 8., 9.)]
        t, ph = np.array(grid).T
        self.j = evaluate(self.p, t, ph)[1]
        self.residuals = np.linspace(-.05, .05, len(t))

    def test_covariance_and_entropy_cross_terms(self):
        out = estimate_uncertainty(self.p, self.j, self.residuals, np.arange(6), 298.15)
        expected = np.linalg.inv(self.j.T @ self.j)*(self.residuals @ self.residuals)/10
        np.testing.assert_allclose(out['covariance'], expected, rtol=1e-8)
        g = np.array([1000*R, 0, 1000/298.15, 0, 0, 0])
        self.assertAlmostEqual(out['derived_standard_errors']['S_HA_J_mol_K'], np.sqrt(g @ expected @ g))

    def test_absolute_sigmas_do_not_rescale_with_residuals(self):
        a = estimate_uncertainty(self.p, self.j, self.residuals, np.arange(6), 298.15, True)
        b = estimate_uncertainty(self.p, self.j, self.residuals*5, np.arange(6), 298.15, True)
        np.testing.assert_array_equal(a['covariance'], b['covariance'])

    def test_fixed_pka_has_no_reported_se(self):
        out = estimate_uncertainty(self.p, self.j[:, :5], self.residuals, np.arange(5), 298.15)
        self.assertTrue(out['available'])
        self.assertIsNone(out['standard_errors']['pKa_ref'])
        self.assertEqual(out['degrees_of_freedom'], 11)

    def test_rank_deficiency_does_not_report_false_precision(self):
        j = self.j.copy(); j[:, 1] = j[:, 0]
        out = estimate_uncertainty(self.p, j, self.residuals, np.arange(6), 298.15)
        self.assertFalse(out['available'])
        self.assertIsNone(out['covariance'])
