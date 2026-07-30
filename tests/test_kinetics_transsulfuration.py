from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_common import Experiment  # noqa: E402
from kinet_fitting import (  # noqa: E402
    fit_mbfe3_sulfide_hss_transsulfuration_no_auto,
)
from kinet_models import (  # noqa: E402
    concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto,
)


class TranssulfurationWithoutAutocatalysisTests(unittest.TestCase):
    def test_no_hss_reduces_to_intrinsic_slow_path(self) -> None:
        t = np.linspace(0.0, 200.0, 101)
        k_slow = 0.01
        c0 = 2.5

        concentrations = (
            concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                t,
                k_slow=k_slow,
                k_ts=3.0,
                k_fast=0.2,
                hss_ratio=0.0,
                c0=c0,
            )
        )

        expected_reduced = c0 * (1.0 - np.exp(-k_slow * t))
        np.testing.assert_allclose(concentrations[1], expected_reduced, rtol=2e-7)
        np.testing.assert_allclose(concentrations.sum(axis=0), c0, atol=1e-12)

    def test_initial_reduction_rate_is_set_by_k_slow(self) -> None:
        dt = 1e-8
        c0 = 1.7
        k_slow = 0.012

        concentrations = (
            concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                np.array([0.0, dt]),
                k_slow=k_slow,
                k_ts=5.0,
                k_fast=0.4,
                hss_ratio=10.0,
                c0=c0,
            )
        )

        numerical_initial_rate = concentrations[1, 1] / dt
        np.testing.assert_allclose(
            numerical_initial_rate,
            k_slow * c0,
            rtol=3e-5,
        )

    def test_fit_reports_three_kinetic_parameters_without_k_auto(self) -> None:
        t = np.linspace(0.0, 120.0, 31)
        concentrations = (
            concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                t,
                k_slow=0.004,
                k_ts=0.08,
                k_fast=0.03,
                hss_ratio=2.0,
            )
        )
        pure_spectra = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.7],
                [0.1, 0.9],
                [0.6, 0.3],
            ]
        )
        experiment = Experiment(
            t=t,
            wavelength=np.array([400.0, 450.0, 500.0, 550.0]),
            absorbance=pure_spectra @ concentrations,
        )

        result = fit_mbfe3_sulfide_hss_transsulfuration_no_auto(
            experiment,
            hss_ratio=2.0,
            k_bounds=(1e-5, 1.0),
            fix_initial_spectrum=True,
            fix_final_spectrum=True,
        )

        self.assertEqual(
            result.model,
            "mbfe3_sulfide_hss_transsulfuration_no_auto",
        )
        self.assertEqual(set(result.params), {"k_slow", "k_ts", "k_fast"})
        self.assertNotIn("k_auto", result.params)
        self.assertTrue(np.isfinite(result.error))
        self.assertLess(result.error, 0.1)
