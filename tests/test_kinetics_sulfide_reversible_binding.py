from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_common import Experiment  # noqa: E402
from kinet_cli import endpoint_spectrum_menu_labels  # noqa: E402
from kinet_fitting import fit_mbfe3_sulfide_binding_autocatalytic  # noqa: E402
from kinet_models import (  # noqa: E402
    concentration_profile_mbfe3_sulfide_binding_autocatalytic,
)
from kinet_known_spectra import (  # noqa: E402
    KnownSpectrumSpec,
    build_known_spectra_matrix,
)


class SulfideReversibleBindingTests(unittest.TestCase):
    def test_known_spectrum_b_alias_maps_to_chemical_intermediate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "B.dat"
            path.write_text("# wavelength B\n400 10\n500 20\n", encoding="utf-8")
            known, labels = build_known_spectra_matrix(
                [KnownSpectrumSpec(species=("B",), path=path)],
                ("MbFeIII", "MbFeIII-HS", "MbFeII"),
                np.array([400.0, 450.0, 500.0]),
            )

        self.assertEqual(labels, ("MbFeIII-HS",))
        assert known is not None
        np.testing.assert_allclose(known[:, 1], [10.0, 15.0, 20.0])
        self.assertTrue(np.all(np.isnan(known[:, 0])))
        self.assertTrue(np.all(np.isnan(known[:, 2])))

    def test_interactive_menu_offers_both_endpoint_spectra(self) -> None:
        self.assertEqual(
            endpoint_spectrum_menu_labels(
                "mbfe3_sulfide_binding_autocatalytic",
                interactive=True,
                initial_spectrum_index=None,
            ),
            ("MbFeIII", "MbFeII"),
        )

    def test_initial_fluxes_and_mass_conservation(self) -> None:
        dt = 1e-7
        c0 = 2.3
        k_on = 0.7
        c = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
            np.array([0.0, dt]),
            k_on=k_on,
            k_off=0.4,
            k_slow=0.03,
            k_cat=0.08,
            c0=c0,
        )

        np.testing.assert_allclose((c[0, 1] - c0) / dt, -k_on * c0, rtol=2e-6)
        np.testing.assert_allclose(c[1, 1] / dt, k_on * c0, rtol=2e-6)
        np.testing.assert_allclose(c.sum(axis=0), c0, atol=1e-12)

    def test_dissociation_changes_the_binding_transient(self) -> None:
        t = np.linspace(0.0, 20.0, 101)
        common = dict(k_on=0.8, k_slow=0.01, k_cat=0.02)
        weak_off = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
            t, k_off=1e-8, **common
        )
        strong_off = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
            t, k_off=0.5, **common
        )

        self.assertGreater(strong_off[0, 20], weak_off[0, 20])
        self.assertLess(strong_off[1, 20], weak_off[1, 20])
        np.testing.assert_allclose(strong_off.sum(axis=0), 1.0, atol=1e-12)

    def test_fit_reports_four_parameters_and_accepts_both_fixed_endpoints(self) -> None:
        t = np.linspace(0.0, 80.0, 41)
        c = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
            t, k_on=0.3, k_off=0.08, k_slow=0.015, k_cat=0.05
        )
        spectra = np.array(
            [
                [0.9, 0.3, 0.2],
                [0.4, 0.8, 0.5],
                [0.2, 0.4, 0.9],
                [0.5, 0.2, 0.6],
            ]
        )
        experiment = Experiment(
            t=t,
            wavelength=np.array([409.0, 428.0, 434.0, 550.0]),
            absorbance=spectra @ c,
        )

        result = fit_mbfe3_sulfide_binding_autocatalytic(
            experiment,
            k_bounds=(1e-4, 1.0),
            fix_initial_spectrum=True,
            fix_final_spectrum=True,
        )

        self.assertEqual(set(result.params), {"k_on", "k_off", "k_slow", "k_cat"})
        self.assertTrue(result.fixed_initial_spectrum)
        self.assertTrue(result.fixed_final_spectrum)
        self.assertTrue(np.isfinite(result.error))
        self.assertLess(result.error, 0.5)


if __name__ == "__main__":
    unittest.main()
