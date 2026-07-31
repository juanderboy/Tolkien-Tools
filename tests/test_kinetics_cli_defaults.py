from __future__ import annotations

import sys
import unittest
from pathlib import Path


KINETICS_DIR = Path(__file__).resolve().parents[1] / "TolkienTools" / "kinetics"
sys.path.insert(0, str(KINETICS_DIR))

from kinet_cli import apply_model_default_wavelength_range, build_parser  # noqa: E402


class KineticsCliDefaultTests(unittest.TestCase):
    def test_special_models_default_to_385_650_nm(self) -> None:
        args = build_parser().parse_args(
            ["experiment.dat", "--model", "mbfe3_hss_no_binding"]
        )

        apply_model_default_wavelength_range(args, args.model)

        self.assertEqual((args.lambda_min, args.lambda_max), (385.0, 650.0))

    def test_explicit_special_model_range_is_preserved(self) -> None:
        args = build_parser().parse_args(
            [
                "experiment.dat",
                "--model",
                "mbfe3_hss_no_binding",
                "--lambda-min",
                "400",
                "--lambda-max",
                "700",
            ]
        )

        apply_model_default_wavelength_range(args, args.model)

        self.assertEqual((args.lambda_min, args.lambda_max), (400.0, 700.0))

    def test_general_models_keep_broad_default_range(self) -> None:
        args = build_parser().parse_args(["experiment.dat", "--model", "a_to_b"])

        apply_model_default_wavelength_range(args, args.model)

        self.assertEqual((args.lambda_min, args.lambda_max), (320.0, 820.0))


if __name__ == "__main__":
    unittest.main()
