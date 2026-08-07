from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_mcr import mcr_als_decompose  # noqa: E402
from kinet_common import Experiment  # noqa: E402
from kinet_fitting import fit_model  # noqa: E402
from kinet_models import concentration_profile_a_rev_b_to_c  # noqa: E402


class McrAlsTests(unittest.TestCase):
    def test_reversible_dispatch_returns_mcr_result(self) -> None:
        wavelength = np.linspace(400.0, 500.0, 40)
        time = np.linspace(0.0, 80.0, 41)
        concentrations = concentration_profile_a_rev_b_to_c(
            time,
            k1=0.08,
            k_1=0.01,
            k2=0.02,
        )
        spectra = np.vstack(
            (
                np.exp(-0.5 * ((wavelength - 409.0) / 7.0) ** 2),
                0.8 * np.exp(-0.5 * ((wavelength - 428.0) / 8.0) ** 2),
                0.7 * np.exp(-0.5 * ((wavelength - 434.0) / 9.0) ** 2),
            )
        ).T
        experiment = Experiment(
            t=time,
            wavelength=wavelength,
            absorbance=spectra @ concentrations,
        )
        result = fit_model(
            "a_rev_b_to_c",
            experiment,
            c0=1.0,
            n_components=3,
            method="mcr_als",
            k_bounds=(1e-4, 1.0),
            mcr_kinetic_weight=10.0,
        )
        self.assertEqual(result.method, "mcr_als")
        self.assertEqual(set(result.params), {"k1", "k_1", "k2"})
        self.assertTrue(np.isfinite(result.error))

    def test_recovers_nonnegative_components_with_fixed_endmembers(self) -> None:
        wavelength = np.linspace(400.0, 500.0, 60)
        time = np.linspace(0.0, 20.0, 41)
        spectra_true = np.vstack(
            (
                np.exp(-0.5 * ((wavelength - 409.0) / 7.0) ** 2),
                0.8 * np.exp(-0.5 * ((wavelength - 428.0) / 8.0) ** 2),
                0.7 * np.exp(-0.5 * ((wavelength - 434.0) / 9.0) ** 2),
            )
        ).T
        concentrations = np.vstack(
            (
                np.exp(-0.15 * time),
                0.6 * (1.0 - np.exp(-0.25 * time)),
                1.0 - np.exp(-0.15 * time) - 0.6 * (1.0 - np.exp(-0.25 * time)),
            )
        )
        data = spectra_true @ concentrations
        known = np.full_like(spectra_true, np.nan)
        known[:, 0] = spectra_true[:, 0]
        known[:, 2] = spectra_true[:, 2]

        spectra, recovered, error, _ = mcr_als_decompose(
            data,
            n_components=3,
            initial_concentrations=concentrations,
            known_spectra=known,
            closure_total=1.0,
            kinetic_reference=concentrations,
            kinetic_weight=10.0,
        )

        self.assertLess(error, 1e-8)
        np.testing.assert_allclose(spectra[:, 0], spectra_true[:, 0])
        np.testing.assert_allclose(spectra[:, 2], spectra_true[:, 2])
        self.assertTrue(np.all(spectra[:, 1] >= 0.0))
        self.assertTrue(np.all(recovered >= 0.0))
        np.testing.assert_allclose(recovered.sum(axis=0), 1.0)
        self.assertAlmostEqual(
            float(np.argmax(spectra[:, 1])),
            float(np.argmax(spectra_true[:, 1])),
            delta=1.0,
        )

    def test_rejects_partial_known_spectrum_columns(self) -> None:
        data = np.ones((4, 5))
        initial = np.ones((2, 5)) / 2.0
        known = np.full((4, 2), np.nan)
        known[0, 0] = 1.0

        with self.assertRaises(ValueError):
            mcr_als_decompose(
                data,
                n_components=2,
                initial_concentrations=initial,
                known_spectra=known,
            )


if __name__ == "__main__":
    unittest.main()
