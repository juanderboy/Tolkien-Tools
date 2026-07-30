from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_common import Experiment  # noqa: E402
from kinet_t0 import estimate_sulfide_no_binding_t0  # noqa: E402


class SulfideNoBindingT0EstimatorTests(unittest.TestCase):
    def build_experiment(self) -> Experiment:
        wavelength = np.arange(390.0, 851.0)
        t = np.arange(0.0, 121.0, 0.5)
        ferric = np.exp(-0.5 * ((wavelength - 409.0) / 8.0) ** 2)
        intermediate = 0.55 * np.exp(-0.5 * ((wavelength - 428.0) / 12.0) ** 2)
        product = 0.65 * np.exp(-0.5 * ((wavelength - 434.0) / 11.0) ** 2)

        injection_time = 20.0
        after = np.maximum(t - injection_time, 0.0)
        bound = np.where(t >= injection_time, 1.0 - np.exp(-after / 1.5), 0.0)
        reduced = np.where(t >= injection_time, 0.03 * after / 100.0, 0.0)
        reduced = np.minimum(reduced, 0.03)
        ferric_fraction = 1.0 - bound
        intermediate_fraction = np.maximum(bound - reduced, 0.0)
        spectra = (
            ferric[:, None] * ferric_fraction
            + intermediate[:, None] * intermediate_fraction
            + product[:, None] * reduced
        )
        # Time-dependent instrumental offset should be removed by 750-850 nm.
        spectra += (0.01 * np.sin(t / 7.0))[None, :]
        return Experiment(t=t, wavelength=wavelength, absorbance=spectra)

    def test_estimator_finds_post_binding_plateau(self) -> None:
        estimate = estimate_sulfide_no_binding_t0(self.build_experiment())

        self.assertAlmostEqual(estimate.addition_time, 20.0, delta=0.5)
        self.assertGreater(estimate.recommended_time, estimate.addition_time)
        self.assertLess(estimate.recommended_time, 35.0)
        self.assertLessEqual(
            estimate.plateau_distance[estimate.recommended_index],
            0.005,
        )
        self.assertEqual(estimate.baseline_region, (750.0, 850.0))

    def test_estimator_requires_baseline_region(self) -> None:
        experiment = self.build_experiment()
        truncated = Experiment(
            t=experiment.t,
            wavelength=experiment.wavelength[experiment.wavelength <= 700.0],
            absorbance=experiment.absorbance[experiment.wavelength <= 700.0],
        )

        with self.assertRaisesRegex(ValueError, "Baseline correction"):
            estimate_sulfide_no_binding_t0(truncated)
