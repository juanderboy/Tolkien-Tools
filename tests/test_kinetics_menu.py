import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'TolkienTools'/'kinetics'))
import kinet_cli as cli
import kinet_activation as activation
from kinet_common import GENERAL_MODEL_LABELS, SPECIAL_MODEL_LABELS

class MenuTests(unittest.TestCase):
    def choose(self, replies, source=None):
        args = cli.build_parser().parse_args([source] if source else [])
        with patch('builtins.input', side_effect=replies), redirect_stdout(io.StringIO()):
            start = cli.kinetics_menu(args)
        return start, args

    def test_each_classic_mechanism_requests_input_after_choice(self):
        for index, model in enumerate(GENERAL_MODEL_LABELS, 1):
            start, args = self.choose([str(index), 'data.KD'])
            self.assertTrue(start)
            self.assertEqual(args.model, model)
            self.assertEqual(args.input, 'data.KD')

    def test_provided_input_is_not_requested_again(self):
        start, args = self.choose(['3'], 'provided.KD')
        self.assertTrue(start)
        self.assertEqual(args.input, 'provided.KD')
        self.assertEqual(args.model, 'a_rev_b_to_c')

    def test_all_special_mechanisms_are_reachable(self):
        for index, model in enumerate(SPECIAL_MODEL_LABELS, 1):
            start, args = self.choose(['4', str(index), 'data.KD'])
            self.assertTrue(start)
            self.assertEqual(args.model, model)

    def test_thermal_dispatch_preserves_provided_file(self):
        with patch.object(activation, 'main') as run:
            start, args = self.choose(['4', '6'], 'rates.csv')
        self.assertFalse(start)
        run.assert_called_once_with(['rates.csv', '--interactive'])

    def test_thermal_without_input_defers_prompt_to_study(self):
        with patch.object(activation, 'main') as run:
            self.choose(['4', '6'])
        run.assert_called_once_with(['--interactive'])

    def test_back_and_invalid_choices_do_not_request_data(self):
        start, args = self.choose(['wrong', '4', 'wrong', '0', '0'])
        self.assertFalse(start)
        self.assertIsNone(args.input)

    def test_cancel_file_returns_to_menu(self):
        start, args = self.choose(['1', '', '0'])
        self.assertFalse(start)
        self.assertIsNone(args.input)

    def test_global_study_reuses_manifest_from_terminal(self):
        start, args = self.choose(['4', '7'], 'manifest.csv')
        self.assertTrue(start)
        self.assertEqual(args.global_manifest, 'manifest.csv')
        self.assertEqual(args.model, 'mbfe3_sulfide_hss_transsulfuration')
