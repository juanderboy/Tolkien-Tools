import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "TolkienTools"
    / "charge_spin"
)
sys.path.insert(0, str(MODULE_DIR))

from charge_spin_coordination import (  # noqa: E402
    proposal_named_groups,
    propose_coordination_fragments,
)
from charge_spin_cli import (  # noqa: E402
    parse_atom_selection_with_ranges,
    select_primary_population_analysis,
    should_process_additional_analysis,
)
import charge_spin_cli  # noqa: E402
import charge_spin_viewer  # noqa: E402
from charge_spin_common import (  # noqa: E402
    get_population_analysis_config,
    parse_global_entity_list,
)
from charge_spin_global import (  # noqa: E402
    collect_global_hist_data,
    infer_entities_from_previous_analysis,
)
from charge_spin_stats import normalize_selected_spin_values  # noqa: E402


def atom(index, element, x, y=0.0, z=0.0):
    return {
        "index": index,
        "model_index": index,
        "element": element,
        "x": x,
        "y": y,
        "z": z,
    }


class CoordinationProposalTests(unittest.TestCase):
    def test_all_processes_mulliken_when_hirshfeld_is_primary(self):
        population_config = get_population_analysis_config("all")

        self.assertTrue(
            should_process_additional_analysis(
                population_config,
                primary_analysis_kind="hirshfeld",
                analysis_kind="mulliken",
            )
        )

    def test_primary_analysis_falls_back_from_hirshfeld_to_loewdin(self):
        population_config = get_population_analysis_config("all")

        selected = select_primary_population_analysis(
            population_config,
            {
                "hirshfeld": False,
                "loewdin": True,
                "mulliken": True,
                "chelpg_hirshfeld": False,
            },
        )

        self.assertEqual(selected, "loewdin")

    def test_primary_analysis_falls_back_from_loewdin_to_mulliken(self):
        population_config = get_population_analysis_config("all")

        selected = select_primary_population_analysis(
            population_config,
            {
                "hirshfeld": False,
                "loewdin": False,
                "mulliken": True,
                "chelpg_hirshfeld": False,
            },
        )

        self.assertEqual(selected, "mulliken")

    def test_separates_coordinated_and_uncoordinated_components(self):
        atoms = [
            atom(0, "Fe", 0.0),
            atom(1, "N", 2.0),
            atom(2, "C", 3.3),
            atom(3, "O", 6.0),
            atom(4, "C", 7.3),
        ]

        proposal = propose_coordination_fragments(atoms)

        self.assertEqual([metal["id"] for metal in proposal["metals"]], ["Fe0"])
        self.assertEqual(len(proposal["components"]), 2)
        coordinated, uncoordinated = proposal["components"]
        self.assertEqual(coordinated["id"], "L1")
        self.assertEqual(coordinated["atom_ids"], [1, 2])
        self.assertTrue(coordinated["coordinated"])
        self.assertEqual(coordinated["donor_atom_ids"], [1])
        self.assertEqual(coordinated["denticity"], 1)
        self.assertEqual(uncoordinated["id"], "L2")
        self.assertEqual(uncoordinated["atom_ids"], [3, 4])
        self.assertFalse(uncoordinated["coordinated"])

    def test_named_groups_include_metals_and_components(self):
        atoms = [
            atom(0, "Fe", 0.0),
            atom(1, "N", 2.0),
            atom(2, "C", 3.3),
        ]

        groups = proposal_named_groups(propose_coordination_fragments(atoms))

        self.assertEqual(groups["fe0"], [0])
        self.assertEqual(groups["l1"], [1, 2])

    def test_returns_empty_proposal_without_transition_metal(self):
        proposal = propose_coordination_fragments(
            [atom(0, "O", 0.0), atom(1, "H", 0.96)]
        )

        self.assertEqual(proposal, {"metals": [], "components": [], "contacts": []})

    def test_fragment_expression_combines_detected_groups_and_atoms(self):
        selected = parse_atom_selection_with_ranges(
            "Fe88 + L1 90-91",
            named_groups={"fe88": [88], "l1": [54, 55, 56, 57]},
        )

        self.assertEqual(selected, [88, 54, 55, 56, 57, 90, 91])

    def test_fragment_selection_can_leave_atoms_outside_analysis(self):
        answers = iter(["selected_region", "1 2", ""])
        with (
            patch("builtins.input", side_effect=lambda _prompt: next(answers)),
            patch.object(
                charge_spin_cli,
                "prompt_numbered_choice",
                return_value="exclude",
            ),
        ):
            fragments = charge_spin_cli.prompt_spin_fragment_configs(
                [1, 2, 3, 4],
                {1: "C", 2: "N", 3: "O", 4: "H"},
            )

        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0]["label"], "selected_region")
        self.assertEqual(fragments[0]["atom_ids"], [1, 2])

    def test_spin_fraction_uses_only_selected_entity_values(self):
        selected_spins = np.array([[0.6, 0.2], [0.3, 0.3]])

        fractions = normalize_selected_spin_values(selected_spins)

        np.testing.assert_allclose(
            fractions,
            np.array([[0.75, 0.25], [0.50, 0.50]]),
        )

    def test_wsl_viewer_opens_containing_folder_with_explorer(self):
        viewer_path = Path("/tmp/example-viewer/coordination_ligand_viewer.html")
        with (
            patch.dict(
                charge_spin_viewer.os.environ,
                {"WSL_DISTRO_NAME": "Ubuntu"},
                clear=False,
            ),
            patch.object(charge_spin_viewer.subprocess, "Popen") as popen,
        ):
            charge_spin_viewer.open_html_viewer(viewer_path)

        popen.assert_called_once_with(
            ["explorer.exe", "."],
            cwd=str(viewer_path.parent),
            stdout=charge_spin_viewer.subprocess.DEVNULL,
            stderr=charge_spin_viewer.subprocess.DEVNULL,
        )

    def test_coordination_viewer_styles_atoms_by_zero_based_index(self):
        atoms = [
            atom(0, "Fe", 0.0),
            atom(1, "O", 1.8),
            atom(2, "H", 2.7),
        ]
        proposal = propose_coordination_fragments(atoms)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "viewer.html"
            charge_spin_viewer.write_coordination_fragment_viewer_from_atoms(
                "test",
                atoms,
                output_path,
                proposal,
            )
            html = output_path.read_text()

        self.assertIn('setStyle({"index": 0}', html)
        self.assertIn('"color": "#7F7F7F"', html)
        self.assertIn('setStyle({"index": 1}', html)
        self.assertIn("donor atoms: 1", html)
        self.assertNotIn('"serial":', html)

    def test_global_parser_accepts_actor_labels(self):
        self.assertEqual(
            parse_global_entity_list("88 Fe-Porphyrin X1"),
            [88, "Fe_Porphyrin", "X1"],
        )

    def test_global_analysis_reuses_current_fragment_actors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            for system_name, actor_token in (
                ("system_a", "Fe-Porphyrin"),
                ("system_b", "Fe_Porphyrin"),
            ):
                system_dir = base / system_name
                system_dir.mkdir()
                (system_dir / "spin_fragment_definitions.dat").write_text(
                    "# fragment_id fragment_label atom_id atom_type\n"
                    f"fragment_{actor_token} {actor_token} 1 C\n"
                )
                (system_dir / f"actor_{actor_token}_hirshfeld_timeseries.dat").write_text(
                    "# time_ps charge spin_fraction\n"
                    "0.0 0.1 0.7\n"
                    "1.0 0.2 0.8\n"
                )
                # A stale atom file must not override the current fragment report.
                (system_dir / "atom_88_hirshfeld_timeseries.dat").write_text(
                    "# time_ps charge spin_fraction\n0.0 0.0 0.0\n"
                )

            entity_map = {}
            for system_name in ("system_a", "system_b"):
                entity_map[system_name] = infer_entities_from_previous_analysis(
                    base / system_name,
                    "hirshfeld",
                )
            self.assertEqual(
                entity_map,
                {
                    "system_a": ["Fe_Porphyrin"],
                    "system_b": ["Fe_Porphyrin"],
                },
            )

            systems_data, missing, spin_labels = collect_global_hist_data(
                base,
                entity_map,
                "hirshfeld",
            )

        self.assertEqual(missing, [])
        self.assertEqual(set(systems_data), {"system_a", "system_b"})
        self.assertEqual(systems_data["system_a"]["Fe_Porphyrin"]["charge"].size, 2)
        self.assertEqual(spin_labels["system_b"], {"spin_fraction"})


if __name__ == "__main__":
    unittest.main()
