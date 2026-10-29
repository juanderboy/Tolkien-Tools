"""Regression tests for spin-free state identity and orbital export validation."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'TolkienTools' / 'td_dft'))
import td_transitions as td


OUTPUT = '''Program Version 6.0.1
STATE  1: E= 0.1 au 3.059 eV 24670 cm**-1 Mult 1
  116a -> 117a : 0.984183 (c= -0.99206013)
STATE 31: E= 0.1 au 2.963 eV 23894 cm**-1 Mult 3
  116b -> 120b : 0.726499 (c= -0.85234927)
  116a -> 117a : 0.187621 (c= -0.43315259)
ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
Transition Energy Energy Wavelength fosc(D2)
0-1A -> 1-3A 2.962544 23894.5 418.5 0.000000000
0-1A -> 1-1A 3.058714 24670.2 405.3 0.000531245
ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS
0-1A -> 1-1A 3.058714 24670.2 405.3 0.99
SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
0-1A -> 1-1A 2.938211 23698.3 422.0 0.88
'''


def parse(text=OUTPUT):
    with patch.object(Path, 'read_text', return_value=text):
        return td.read_transitions('fixture.out')


class TransitionTests(unittest.TestCase):
    def test_global_triplet_id_and_local_absorption_label(self):
        states, soc = parse()
        self.assertTrue(soc)
        self.assertEqual([s.state for s in states], [1, 31])
        self.assertEqual(states[1].label, '1-3A')
        self.assertEqual(states[1].fosc, 0.)
        self.assertAlmostEqual(states[0].fosc, .000531245)
        pair = states[1].contributions[0]
        self.assertEqual((pair.source, pair.source_spin, pair.target, pair.target_spin), (116, 'b', 120, 'b'))
        self.assertLess(pair.coefficient, 0.)

    def test_reject_concatenated_or_unmatched_outputs(self):
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            parse(OUTPUT + OUTPUT)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            parse(OUTPUT.replace('2.962544', '8.0'))
        with self.assertRaisesRegex(ValueError, 'No supported'):
            parse('not an ORCA output')

    def test_standard_singlet_only(self):
        text = OUTPUT[:OUTPUT.index('STATE 31:')] + OUTPUT[OUTPUT.index('ABSORPTION SPECTRUM'):]
        text = '\n'.join(line for line in text.splitlines() if '1-3A' not in line)
        states, _ = parse(text)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].label, '1-1A')

    def test_open_shell_multiplicity_is_explicitly_estimated(self):
        text = 'UHF/UKS reference: multiplicity estimated\n' + OUTPUT
        text = text.replace('24670 cm**-1 Mult 1', '24670 cm**-1 <S**2> = 0.792924 Mult 1')
        states, _ = parse(text)
        self.assertTrue(states[0].multiplicity_estimated)
        self.assertAlmostEqual(states[0].spin_squared, .792924)

    def test_vmd_startup_has_absolute_fallback_and_escaped_path(self):
        folder = Path('/tmp/example $x [test]')
        with patch.object(Path, 'write_text') as write:
            td.write_vmd(folder, [folder / 'orbital_116_a.cube'])
            script = write.call_args.args[0]
        self.assertIn('set here "/tmp/example \\$x \\[test\\]"', script)
        self.assertIn('info script', script)
        self.assertIn('orbital_116_a.cube', script)

    def test_pair_selection(self):
        states, _ = parse()
        pairs = states[1].contributions
        self.assertEqual(len(td.select_pairs('all', pairs)), 2)
        self.assertEqual(td.select_pairs('2,2', pairs), [pairs[1]])
        for spec in ['0', '-1', '3', 'bad', '']:
            with self.assertRaises(ValueError):
                td.select_pairs(spec, pairs)

    def test_negative_atom_orbital_cube_and_truncation(self):
        cube = 'ORCA\nMO\n-1 0 0 0\n1 1 0 0\n1 0 1 0\n1 0 0 1\n1 1 0 0 0\n1 116\n0.5\n'
        def validate(text, orbital=116):
            with patch.object(Path, 'open', return_value=io.StringIO(text)):
                return td.validate_cube(Path('test.cube'), [('H', 0., 0., 0.)], expected_orbital=orbital)
        self.assertEqual(len(validate(cube)), 4)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate(cube.rsplit('0.5', 1)[0])
        with self.assertRaisesRegex(ValueError, 'different orbital'):
            validate(cube, 117)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            validate(cube.replace('0.5', 'nan'))

    def test_list_needs_no_gbw_and_writes_nothing(self):
        with patch.object(Path, 'read_text', return_value=OUTPUT), patch.object(Path, 'write_text') as write, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(td.main(['fixture.out', '--list']), 0)
            write.assert_not_called()

    def test_missing_state_and_gbw_fail_without_prompting(self):
        for args in [['--state', '99'], ['--state', '1']]:
            with patch.object(Path, 'read_text', return_value=OUTPUT), patch('builtins.input', side_effect=AssertionError('unexpected prompt')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(td.main(['fixture.out', *args]), 1)


if __name__ == '__main__':
    unittest.main()
