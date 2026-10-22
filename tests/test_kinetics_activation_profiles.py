"""Numerical checks for nuisance-parameter profiling and its exports."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'TolkienTools'/'kinetics'))
from kinet_activation import NAMES, evaluate, fit_records
from kinet_activation_profiles import analyze_profiles, export_profiles


class ActivationProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = np.array([-7., -4., 115., 27., 24., 7.55])
        grid = [(t, ph) for t in (285., 295., 305., 315.) for ph in (6., 7., 8., 9.)]
        cls.rows = [dict(id=str(i), T_K=t, pH=ph,
                        k=float(np.exp(evaluate(p, t, ph)[0] + .005*np.sin(i))),
                        sigma_log_k=.03)
                    for i, (t, ph) in enumerate(grid)]
        cls.result = fit_records(cls.rows, starts=3, seed=12)
        cls.result['records'] = cls.rows
        cls.analysis = analyze_profiles(cls.result, grid_points=3, extent_se=1.)

    def test_centers_reproduce_weighted_fit_and_fixed_grid_coordinates(self):
        analysis = self.analysis
        best = np.array([self.result['parameters'][name] for name in NAMES])
        self.assertEqual(analysis['manifest']['pair_count'], 15)
        self.assertEqual(analysis['manifest']['failed_nodes'], 0)
        self.assertEqual(analysis['manifest']['display'], 'objective_ratio')
        for (i, j), grid in analysis['surfaces'].items():
            xcenter = np.flatnonzero(grid['x'] == best[i])[0]
            ycenter = np.flatnonzero(grid['y'] == best[j])[0]
            self.assertAlmostEqual(grid['objective'][ycenter, xcenter], self.result['objective'], places=8)
            np.testing.assert_allclose(grid['parameters'][:, :, i], np.broadcast_to(grid['x'], grid['objective'].shape))
            np.testing.assert_allclose(grid['parameters'][:, :, j], np.broadcast_to(grid['y'][:, None], grid['objective'].shape))
        for i, profile in analysis['profiles'].items():
            center = np.flatnonzero(profile['values'] == best[i])[0]
            self.assertAlmostEqual(profile['objective'][center], self.result['objective'], places=8)

    def test_nuisance_parameters_are_refitted_instead_of_frozen(self):
        best = np.array([self.result['parameters'][name] for name in NAMES])
        grid = self.analysis['surfaces'][0, 2]
        frozen = best.copy()
        frozen[0], frozen[2] = grid['x'][0], grid['y'][0]
        t = np.array([r['T_K'] for r in self.rows])
        ph = np.array([r['pH'] for r in self.rows])
        observed = np.log([r['k'] for r in self.rows])
        residual = (evaluate(frozen, t, ph)[0]-observed)/.03
        self.assertLess(grid['objective'][0, 0], residual @ residual - 1e-5)
        fitted = grid['parameters'][0, 0]
        residual = (evaluate(fitted, t, ph)[0]-observed)/.03
        self.assertAlmostEqual(grid['objective'][0, 0], residual @ residual, places=10)

    def test_fixed_pka_is_never_refitted_and_matrix_has_five_parameters(self):
        result = fit_records(self.rows, fixed_pka=7.55, starts=2)
        result['records'] = self.rows
        analysis = analyze_profiles(result, grid_points=3, extent_se=.5)
        self.assertEqual(analysis['manifest']['pair_count'], 10)
        self.assertEqual(analysis['manifest']['panel_count'], 25)
        self.assertNotIn(5, analysis['axes'])
        self.assertTrue(np.isnan(analysis['correlation'][5]).all())
        for grid in analysis['surfaces'].values():
            np.testing.assert_array_equal(grid['parameters'][:, :, 5][grid['success']], 7.55)

    def test_zero_objective_and_unavailable_covariance_use_documented_fallback(self):
        p = np.array([-7., -4., 115., 27., 24., 7.55])
        rows = [dict(r, k=float(np.exp(evaluate(p, r['T_K'], r['pH'])[0])), sigma_log_k=None)
                for r in self.rows]
        result = fit_records(rows, starts=2)
        result['records'] = rows
        result['uncertainty']['standard_errors'] = dict.fromkeys(NAMES)
        result['uncertainty']['covariance'] = None
        analysis = analyze_profiles(result, grid_points=3, extent_se=.02)
        self.assertEqual(analysis['manifest']['display'], 'objective_increase')
        self.assertTrue(all('fallback' in source for source in analysis['manifest']['axis_sources'].values()))
        self.assertTrue(np.isnan(analysis['correlation']).all())

    def test_failed_nodes_are_masked_and_exports_are_valid_json(self):
        analysis = analyze_profiles(self.result, grid_points=3, extent_se=4., max_nfev=1)
        self.assertGreater(analysis['manifest']['failed_nodes'], 0)
        with tempfile.TemporaryDirectory() as directory:
            export_profiles(analysis, directory, plot=False)
            path = Path(directory)
            for filename in ('profiles.json', 'parameter_correlation.json', 'profile_matrix_manifest.json'):
                text = (path/filename).read_text()
                self.assertNotIn('NaN', text)
                json.loads(text)
            self.assertEqual(len(list((path/'matrix_pair_grids').glob('*.npz'))), 15)
            with np.load(path/'matrix_pair_grids/pair_0_1.npz') as grid:
                np.testing.assert_array_equal(grid['success'], np.isfinite(grid['objective']))

    def test_a_better_minimum_is_flagged_without_replacing_the_original_fit(self):
        result = copy.deepcopy(self.result)
        result['parameters'][NAMES[0]] += .2
        original = result['parameters'].copy()
        analysis = analyze_profiles(result, grid_points=3, extent_se=.1)
        self.assertTrue(analysis['manifest']['lower_minimum_found'])
        self.assertLess(analysis['manifest']['lowest_profile_objective'],
                        analysis['manifest']['baseline_objective'])
        self.assertEqual(result['parameters'], original)
        self.assertEqual(analysis['best_parameters'][0], original[NAMES[0]])

    def test_invalid_grid_configuration_is_rejected(self):
        for points in (1, 4, 3.5):
            with self.assertRaises(ValueError):
                analyze_profiles(self.result, grid_points=points)
        for extent in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                analyze_profiles(self.result, extent_se=extent)
