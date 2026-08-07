from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_common import Experiment  # noqa: E402
from kinet_global import (  # noqa: E402
    GlobalExperiment,
    fit_global_transsulfuration,
    read_global_manifest,
)
from kinet_models import (  # noqa: E402
    concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto,
)


class GlobalTranssulfurationTests(unittest.TestCase):
    def test_manifest_resolves_relative_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("placeholder", encoding="utf-8")
            (root / "b.txt").write_text("placeholder", encoding="utf-8")
            manifest = root / "series.csv"
            manifest.write_text(
                "filename,R,c0_after\na.txt,5,8e-6\nb.txt,10,7.8e-6\n",
                encoding="utf-8",
            )
            records = read_global_manifest(manifest)

        self.assertEqual([record[0].name for record in records], ["a.txt", "b.txt"])
        self.assertEqual([record[1] for record in records], [5.0, 10.0])

    def test_recovers_shared_no_auto_parameters_from_multiple_ratios(self) -> None:
        wavelength = np.linspace(410.0, 650.0, 25)
        spectra = np.column_stack(
            (
                8.0e4 * np.exp(-0.5 * ((wavelength - 425.0) / 14.0) ** 2),
                7.0e4 * np.exp(-0.5 * ((wavelength - 434.0) / 13.0) ** 2),
            )
        )
        true = {"k_slow": 9.0e-5, "k_ts": 2.0e3, "k_fast": 1.1e-2}
        series = []
        for ratio, c0 in ((5.0, 7.87e-6), (10.0, 7.74e-6), (30.0, 7.87e-6)):
            time = np.linspace(0.0, 800.0, 45)
            concentrations = (
                concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                    time,
                    true["k_slow"],
                    true["k_ts"],
                    true["k_fast"],
                    hss_ratio=ratio,
                    c0=c0,
                )
            )
            experiment = Experiment(
                t=time,
                wavelength=wavelength,
                absorbance=spectra @ concentrations,
            )
            series.append(
                GlobalExperiment(
                    name=f"R{ratio:g}",
                    source_path=Path(f"R{ratio:g}.txt"),
                    experiment=experiment,
                    hss_ratio=ratio,
                    c0=c0,
                )
            )

        result = fit_global_transsulfuration(
            series,
            fix_initial_spectrum=True,
            fix_final_spectrum=False,
            initial_parameters=true,
            max_starts=1,
        )

        self.assertEqual(set(result.params), {"k_slow", "k_ts", "k_fast"})
        self.assertLess(result.error, 2e-5)
        for name, expected in true.items():
            self.assertAlmostEqual(np.log(result.params[name] / expected), 0.0, delta=0.25)

    def test_can_fix_independently_measured_k_fast(self) -> None:
        wavelength = np.array([410.0, 434.0, 500.0, 600.0])
        spectra = np.array(
            [[8.0e4, 2.0e4], [3.0e4, 7.0e4], [1.0e4, 4.0e4], [2.0e4, 1.0e4]]
        )
        k_fast = 1.0e-3
        series = []
        for ratio in (5.0, 20.0):
            time = np.linspace(0.0, 1000.0, 35)
            c0 = 8.0e-6
            concentrations = (
                concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                    time,
                    9.0e-5,
                    2.0e3,
                    k_fast,
                    hss_ratio=ratio,
                    c0=c0,
                )
            )
            series.append(
                GlobalExperiment(
                    name=f"R{ratio:g}",
                    source_path=Path(f"R{ratio:g}.txt"),
                    experiment=Experiment(time, wavelength, spectra @ concentrations),
                    hss_ratio=ratio,
                    c0=c0,
                )
            )

        result = fit_global_transsulfuration(
            series,
            fix_initial_spectrum=True,
            fix_final_spectrum=False,
            fixed_parameters={"k_fast": k_fast},
            max_starts=1,
        )

        self.assertEqual(result.params["k_fast"], k_fast)
        self.assertLess(result.error, 1e-4)


if __name__ == "__main__":
    unittest.main()
