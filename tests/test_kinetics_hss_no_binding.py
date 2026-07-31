from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_common import Experiment, MODEL_SPECIES  # noqa: E402
from kinet_fitting import fit_model  # noqa: E402
from kinet_models import concentration_profile_a_to_b  # noqa: E402


class HssNoBindingModelTests(unittest.TestCase):
    def test_model_dispatch_fits_mbfe3_hss_to_mbfe2_as_a_to_b(self) -> None:
        wavelength = np.linspace(390.0, 500.0, 80)
        time = np.linspace(0.0, 120.0, 61)
        expected_k = 0.025
        concentrations = concentration_profile_a_to_b(time, expected_k)
        hss_spectrum = np.exp(-0.5 * ((wavelength - 428.0) / 10.0) ** 2)
        ferrous_spectrum = 0.7 * np.exp(
            -0.5 * ((wavelength - 434.0) / 12.0) ** 2
        )
        experiment = Experiment(
            t=time,
            wavelength=wavelength,
            absorbance=np.column_stack((hss_spectrum, ferrous_spectrum))
            @ concentrations,
        )

        result = fit_model(
            "mbfe3_hss_no_binding",
            experiment,
            c0=1.0,
            n_components=2,
            method="nnls",
            k_bounds=(1e-4, 1.0),
        )

        self.assertEqual(result.model, "mbfe3_hss_no_binding")
        self.assertEqual(result.species_labels, ("MbFeIII-HSS", "MbFeII"))
        self.assertEqual(
            MODEL_SPECIES["mbfe3_hss_no_binding"],
            result.species_labels,
        )
        self.assertAlmostEqual(result.params["k"], expected_k, delta=5e-4)


if __name__ == "__main__":
    unittest.main()
