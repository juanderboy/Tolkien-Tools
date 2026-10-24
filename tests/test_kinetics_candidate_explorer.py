import sys
import unittest
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"TolkienTools"/"kinetics"))
from kinet_common import Experiment
from kinet_femb_disulfide import fit_femb_disulfide, fractions
from kinet_candidate_explorer import CandidateExplorer, equivalent_candidate_indices, explore_candidates


class CandidateExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wave=np.arange(400.,451.)
        spectra=np.column_stack([np.exp(-((wave-p)/w)**2)
            for p,w in [(409,7),(428,6),(434,6)]])
        spectra[:,1]*=.65
        times=np.unique(np.r_[np.linspace(0,10,21),np.linspace(10,250,35)])
        cls.experiment=Experiment(times,wave,spectra@fractions(times,.4,.01,.1))
        cls.result=fit_femb_disulfide(cls.experiment,c0=5e-6)

    def tearDown(self):
        plt.close('all')

    def test_only_equivalent_candidates_and_reconstructed_solutions_are_offered(self):
        result=self.result
        d=result.selection_diagnostics
        indices=equivalent_candidate_indices(result)
        self.assertGreater(len(indices),1)
        self.assertIn(d['selected_candidate_index'],indices)
        for index in (indices[0],indices[-1]):
            candidate=result.candidate_reconstructor(index)
            self.assertLessEqual(1000*np.sqrt(np.mean(candidate.residuals**2)),
                d['minimum_rmse_mAU']*(1+d['relative_rmse_tolerance'])+1e-9)
            np.testing.assert_allclose(candidate.spectra@candidate.c,candidate.absorbance_calc,atol=1e-13)
            np.testing.assert_allclose(candidate.residuals,
                self.experiment.absorbance-candidate.absorbance_calc,atol=1e-13)
            np.testing.assert_allclose(candidate.c.sum(axis=0),5e-6,atol=1e-14)

    def test_navigation_accepts_complete_fit_without_mutating_recommendation(self):
        explorer=CandidateExplorer(self.experiment,self.result)
        explorer.on_slide(0)
        current=explorer.current
        np.testing.assert_allclose(explorer.spectral_lines[1].get_ydata(),current.spectra[:,1])
        np.testing.assert_allclose(explorer.conc_lines[1].get_ydata(),current.c[1]*1e6)
        np.testing.assert_allclose(explorer.mesh.get_array().ravel(),(current.residuals*1000).T.ravel())
        self.assertIsNone(explorer.accepted)
        explorer.accept()
        accepted=explorer.accepted
        self.assertEqual(accepted.selection_diagnostics['selection_mode'],'manual')
        self.assertEqual(self.result.selection_diagnostics['selection_mode'],'automatic')
        self.assertEqual(accepted.params,current.params)
        np.testing.assert_array_equal(accepted.spectra,current.spectra)
        np.testing.assert_array_equal(accepted.c,current.c)

    def test_exports_use_the_manually_accepted_candidate(self):
        import io, json, tempfile
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from kinet_cli import build_parser
        from kinet_export import export_final_fit_outputs
        explorer=CandidateExplorer(self.experiment,self.result)
        explorer.on_slide(len(explorer.pool)-1)
        explorer.accept()
        accepted=explorer.accepted
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'trial.KD'
            with patch('kinet_export.plot_result'),redirect_stdout(io.StringIO()):
                export_final_fit_outputs(source,source,self.experiment,accepted,(1e-7,5.),
                    build_parser().parse_args([]),5e-6,0.,250.,accepted.model)
            out=Path(directory)/'results_trial'
            selection=json.loads((out/'trial_selection.json').read_text())
            self.assertEqual(selection['selection_mode'],'manual')
            self.assertEqual(selection['selected']['parameters'],
                             accepted.selection_diagnostics['selected']['parameters'])
            self.assertIn('candidate_selection_mode\tmanual',
                          (out/'trial_fit_summary.dat').read_text())
            spectra=np.loadtxt(out/'trial_pure_spectra.dat')[:,1:]
            np.testing.assert_allclose(spectra,accepted.spectra,rtol=1e-11)
            simulated=np.genfromtxt(out/'trial_simulated.csv',delimiter=',')[1:,1:]
            np.testing.assert_allclose(simulated,accepted.absorbance_calc,rtol=1e-11)

    def test_flags_reset_and_close_do_not_accept_preview(self):
        explorer=CandidateExplorer(self.experiment,self.result)
        explorer.toggle_flags(None)
        self.assertEqual(len(explorer.pool),len(explorer.indices))
        explorer.reset()
        self.assertEqual(explorer.current_index,self.result.selection_diagnostics['selected_candidate_index'])
        plt.close(explorer.fig)
        self.assertIsNone(explorer.accepted)
        self.assertIs(explore_candidates(self.experiment,self.result),self.result)

if __name__=='__main__':
    unittest.main()
