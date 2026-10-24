import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'TolkienTools/kinetics'))
from kinet_cli import build_parser, install_endpoint_references, endpoint_spectrum_menu_labels
from kinet_common import Experiment
from kinet_preprocessing import (parse_endpoint_reference, ask_endpoint_reference,
                                 preprocess_experiment)
from kinet_fitting import fit_direct_spectra


class EndpointReferenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = Experiment(t=np.arange(6.)*10,
            wavelength=np.array([400.,500.,750.,800.,850.]),
            absorbance=np.array([[1,2,3,4,5,6],[2,4,6,8,10,12],
                                 [0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5.]]))

    def preprocess(self, initial, final, drop='', approval=None):
        args=build_parser().parse_args([])
        args.lambda_min, args.lambda_max=400.,500.
        replies=[drop,'20','40',initial,final,'750-850','400-500','2',approval or 'yes']
        with patch('builtins.input',side_effect=replies), \
             patch('kinet_plotting.plot_preprocessed_experiment_preview'), \
             redirect_stdout(io.StringIO()):
            result=preprocess_experiment(args,self.raw,initial_spectrum_label='A',final_spectrum_label='B',
                                         preferred_baseline_region=(750.,850.))
        return args,result

    def test_every_active_model_uses_the_same_per_species_reference_dialog(self):
        import tempfile
        from kinet_common import MODEL_SPECIES, HSS_TRANSSULFURATION_MODELS
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.dat'
            path.write_text('400 10 20 30\n500 12 24 36\n')
            for model, labels in MODEL_SPECIES.items():
                if model in HSS_TRANSSULFURATION_MODELS:
                    continue
                for selection in ('no', '2', '2-3', str(path)):
                    with self.subTest(model=model, selection=selection):
                        args=build_parser().parse_args(['--model',model])
                        args.lambda_min,args.lambda_max=400.,500.
                        replies=['','20','40']
                        for label in labels:
                            replies.append(selection)
                            if selection==str(path):
                                replies.append('no')
                        replies += ['750-850','400-500','2','yes']
                        with patch('builtins.input',side_effect=replies) as prompts, \
                             patch('kinet_plotting.plot_preprocessed_experiment_preview'), \
                             redirect_stdout(io.StringIO()):
                            preprocess_experiment(args,self.raw,
                                initial_spectrum_label=labels[0],final_spectrum_label=labels[-1],
                                species_labels=labels,preferred_baseline_region=(750.,850.))
                        questions=[call.args[0] for call in prompts.call_args_list]
                        self.assertEqual(sum('Known spectrum for species' in q for q in questions),len(labels))
                        self.assertEqual(len(args.fixed_reference_files),len(labels) if selection==str(path) else 0)

    def test_reference_question_is_explicit_and_options_are_separate_lines(self):
        output=io.StringIO()
        with patch('builtins.input',return_value='no'),redirect_stdout(output):
            ask_endpoint_reference('B','intermediate',0,6,[])
        text=output.getvalue()
        self.assertIn('Use a known spectrum for species B?',text)
        for option in ('no', 'N', 'N-M', 'file.dat'):
            self.assertIn('\n  '+option,text)
        self.assertIn('no automatic',text)

    def test_original_indices_and_inclusive_ranges_validate(self):
        self.assertEqual(parse_endpoint_reference('23-25',30,[]),(22,23,24))
        self.assertEqual(parse_endpoint_reference('23',30,[]),(22,))
        for bad in ['0','7','4-2','2-7','1,2','abc']:
            with self.assertRaises(ValueError): parse_endpoint_reference(bad,6,[])
        with self.assertRaises(ValueError): parse_endpoint_reference('2-4',6,[2])

    def test_yes_no_have_same_semantics_at_both_endpoints(self):
        for endpoint in ['first','last']:
            with patch('builtins.input',return_value='sí'),redirect_stdout(io.StringIO()):
                self.assertEqual(ask_endpoint_reference('A',endpoint,0,6,[]),(True,None))
            with patch('builtins.input',return_value='no'),redirect_stdout(io.StringIO()):
                self.assertEqual(ask_endpoint_reference('A',endpoint,0,6,[]),(False,None))
        _,result=self.preprocess('yes','yes')
        self.assertEqual(result[5:7],(True,True))
        self.assertIsNone(result[8]);self.assertIsNone(result[9])
        _,result=self.preprocess('no','no')
        self.assertEqual(result[5:7],(False,False))

    def test_outside_window_means_use_baseline_without_moving_t0(self):
        args,result=self.preprocess('1-2','5-6')
        corrected,_,c0,*_=result
        np.testing.assert_array_equal(corrected.t,[0,10,20])
        # Baseline is [0,1,2,3,4,5] for each frame.
        np.testing.assert_allclose(result[8],[1,2.5])
        np.testing.assert_allclose(result[9],[1,6.5])
        self.assertEqual(args.initial_reference_indices,(0,1))
        self.assertEqual(args.final_reference_indices,(4,5))
        fixed=install_endpoint_references(None,(),('A','B'),result[8],result[9],c0)
        np.testing.assert_allclose(fixed[:,0],result[8]/2)
        np.testing.assert_allclose(fixed[:,1],result[9]/2)
        concentrations=np.array([[2,1,0],[0,1,2.]])
        # Deliberately inconsistent observations: fixed references must not rescale.
        recovered=fit_direct_spectra(np.ones((2,3))*99,concentrations,spectra_method="nnls",known_spectra=fixed)
        np.testing.assert_array_equal(recovered,fixed)

    def test_single_product_index_and_dropped_reference(self):
        _,result=self.preprocess('2','4')
        np.testing.assert_allclose(result[8],[1,3])
        np.testing.assert_allclose(result[9],[1,5])
        with self.assertRaises(ValueError): self.preprocess('2','4',drop='2')

    def test_explicit_t0_takes_priority_over_noninteractive_reference(self):
        args=build_parser().parse_args(['--no-plot', '--initial-spectrum-index', '2',
            '--reaction-start-time', '20', '--reaction-end-time', '40',
            '--lambda-min', '400', '--lambda-max', '500',
            '--baseline-lambda-min', '750', '--baseline-lambda-max', '850'])
        result=preprocess_experiment(args,self.raw,initial_spectrum_label='A')
        np.testing.assert_array_equal(result[0].t,[0,10,20])
        np.testing.assert_allclose(result[8],[1,3])
        self.assertEqual(args.initial_reference_indices,[1])
        self.assertIsNone(result[9])

    def test_summary_records_original_reference_indices(self):
        import tempfile
        from types import SimpleNamespace
        from kinet_export import write_fit_summary_dat
        args,result=self.preprocess('1-2','5-6')
        fit=SimpleNamespace(method='nnls',error=0.,spectra=np.ones((2,2)),
            residuals=np.zeros_like(self.raw.absorbance),
            fixed_initial_spectrum=False,fixed_final_spectrum=False,
            known_species=(),known_spectrum_scales={},params={'k':.01},
            species_labels=('A','B'),singular_values=np.array([1.,.5]))
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/'summary.dat'
            write_fit_summary_dat(dest,Path('data.KD'),Path('data.txt'),self.raw,
                                  fit,(1e-8,.1),args,2.,20.,40.,'a_to_b')
            text=dest.read_text()
        self.assertIn('fixed_initial_reference_original_indices_1based\t1,2',text)
        self.assertIn('fixed_final_reference_original_indices_1based\t5,6',text)
        self.assertIn('fixed_final_reference_aggregation\tarithmetic_mean',text)

    def preprocess_species(self, initial, middle, final, work_range='400-500', scale_choices=None):
        args=build_parser().parse_args([])
        args.lambda_min,args.lambda_max=400.,500.
        replies=['','20','40']
        for alias, selection in zip('ABC',(initial,middle,final)):
            replies.append(selection)
            if Path(selection).is_file():
                answer=(scale_choices or {}).get(alias,'no')
                replies.append(answer)
        replies.extend(['750-850',work_range,'2','yes'])
        with patch('builtins.input',side_effect=replies), \
             patch('kinet_plotting.plot_preprocessed_experiment_preview'), \
             redirect_stdout(io.StringIO()):
            result=preprocess_experiment(args,self.raw,initial_spectrum_label='A',
                final_spectrum_label='C',species_labels=('A','B','C'),
                preferred_baseline_region=(750.,850.))
        return args,result

    def test_file_references_are_fixed_molar_values_without_baseline_or_case_loss(self):
        import tempfile
        from kinet_known_spectra import FixedSpectrumFile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'My Spectrum.DAT'
            path.write_text('# wavelength A B C\n400 10 20 30\n500 12 24 36\n')
            args,result=self.preprocess_species(str(path),str(path),str(path))
            self.assertIsInstance(result[7],FixedSpectrumFile)
            np.testing.assert_allclose(result[8],[20,24])  # c0=2, A selected
            np.testing.assert_allclose(result[9],[60,72])  # C selected, not A
            np.testing.assert_allclose(args.additional_fixed_references[1],[20,24])
            self.assertIsNone(args.initial_reference_indices)
            self.assertIsNone(args.final_reference_indices)
            self.assertEqual(len(args.fixed_reference_files),3)
            self.assertEqual(args.fixed_reference_files[1]['species'],'C')
            self.assertEqual(args.fixed_reference_files[0]['path'],str(path))
            self.assertEqual(len(args.fixed_reference_files[0]['sha256']),64)
            matrix=install_endpoint_references(None,(),('A','B','C'),result[8],result[9],2.)
            matrix[:,1]=args.additional_fixed_references[1]
            recovered=fit_direct_spectra(np.ones((2,3))*99,np.eye(3)*2,
                                        spectra_method='nnls',known_spectra=matrix)
            np.testing.assert_array_equal(recovered,[[10,20,30],[12,24,36]])

    def test_intermediate_can_use_original_mean_or_remain_free(self):
        args,result=self.preprocess_species('no','1-2','no')
        np.testing.assert_allclose(args.additional_fixed_references[1],[.5,1.25])
        self.assertEqual(args.intermediate_reference_indices,{'B':(0,1)})
        args,_=self.preprocess_species('no','no','no')
        self.assertEqual(args.additional_fixed_references,{})
        with self.assertRaises(ValueError): self.preprocess_species('no','yes','no')

    def test_file_coverage_clips_default_fit_range_without_extrapolation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'A.dat'
            path.write_text('450 10\n500 20\n')
            args,result=self.preprocess_species(str(path),'no','no',work_range='')
            self.assertEqual(result[1],(450.,500.))
            np.testing.assert_allclose(result[8],[40])

    def test_fixed_file_reader_rejects_ambiguous_or_invalid_spectra(self):
        import tempfile
        from kinet_known_spectra import FixedSpectrumFile, read_fixed_spectrum_file
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.dat'
            for text in ['400 10\n400 20\n', '400 nan\n500 20\n',
                         '# wavelength A C\n400 1 2\n500 3 4\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_fixed_spectrum_file(FixedSpectrumFile(path,'B'))
            path.write_text('400 10\n500 20\n')
            w,v=read_fixed_spectrum_file(FixedSpectrumFile(path,'C'))
            np.testing.assert_array_equal(v,[10,20])
        with patch('builtins.input',return_value='/does/not/exist.dat'), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError): ask_endpoint_reference('A','first',0,6,[])

    def test_main_uses_one_reference_question_per_species_and_protects_files(self):
        import tempfile
        from types import SimpleNamespace
        import kinet_cli as cli
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'Fixed.DAT'
            path.write_text('400 10\n500 20\n')
            replies=['3','','20','40',str(path),'yes','no',str(path),'no','750-850','400-500','2','yes']
            fake=SimpleNamespace()
            with patch.object(sys,'argv',['kinet_python.py','data.KD']), \
                 patch('builtins.input',side_effect=replies) as prompts, \
                 patch.object(cli,'resolve_input_file',return_value=Path('data.txt')), \
                 patch.object(cli,'read_experiment',return_value=self.raw), \
                 patch.object(cli,'plot_experiment_overview',return_value=(750.,850.)), \
                 patch('kinet_preprocessing.suggest_auto_baseline_region',return_value=(750.,850.)), \
                 patch('kinet_plotting.plot_preprocessed_experiment_preview'), \
                 patch.object(cli,'fit_model_with_auto_k_max',return_value=(fake,(1e-8,.1))) as fit, \
                 patch.object(cli,'print_fit_report'),patch.object(cli,'plot_result'), \
                 patch.object(cli,'ask_post_fit_spectra_to_drop',return_value=None), \
                 patch.object(cli,'export_final_fit_outputs') as export,redirect_stdout(io.StringIO()):
                cli.main()
            kwargs=fit.call_args.kwargs
            self.assertEqual(kwargs['known_species'],('A',))
            self.assertEqual(kwargs['known_spectrum_scale_bounds'],{'A':(.7,1.3)})
            np.testing.assert_array_equal(kwargs['known_spectra'][:,0],[10,20])
            np.testing.assert_array_equal(kwargs['known_spectra'][:,2],[10,20])
            self.assertTrue(np.isnan(kwargs['known_spectra'][:,1]).all())
            self.assertIn(path,export.call_args.kwargs['protected_input_paths'])
            texts=[call.args[0] for call in prompts.call_args_list]
            self.assertEqual(sum('Known spectrum for species' in text for text in texts),3)
            self.assertFalse(any('Do you have known spectra' in text for text in texts))

    def test_epsilon_file_amplitude_choice_is_independent_per_species(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.dat'
            path.write_text('400 10\n500 20\n')
            args,result=self.preprocess_species(str(path),str(path),'no',
                scale_choices={'A':'yes','B':'no'})
            np.testing.assert_allclose(result[8],[20,40]) # epsilon is multiplied by c0
            np.testing.assert_allclose(args.additional_fixed_references[1],[10,20]) # molar
            first=next(r for r in args.fixed_reference_files if r['species']=='A')
            middle=next(r for r in args.fixed_reference_files if r['species']=='B')
            self.assertEqual(first['units'],'M^-1 cm^-1')
            self.assertTrue(first['fit_amplitude'])
            self.assertEqual(first['amplitude_bounds'],(.7,1.3))
            self.assertFalse(middle['fit_amplitude'])

    def test_extra_questions_only_for_files_and_remember_amplitude_choice(self):
        import tempfile
        from kinet_known_spectra import FixedSpectrumFile
        for reply in ['yes','no','2','2-3']:
            with patch('builtins.input',return_value=reply) as read, redirect_stdout(io.StringIO()):
                ask_endpoint_reference('A','first',0,6,[])
            self.assertEqual(read.call_count,1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.dat';path.write_text('400 1\n500 2\n')
            previous=FixedSpectrumFile(path.resolve(),'A',True)
            output=io.StringIO()
            with patch('builtins.input',side_effect=['','']) as read,redirect_stdout(output):
                _,selection=ask_endpoint_reference('A','first',0,6,[],default_indices=previous,alias='A')
            self.assertEqual(selection,previous)
            self.assertEqual(read.call_count,2)
            self.assertIn("epsilon, in M^-1 cm^-1",output.getvalue())

    def test_real_fit_recovers_relative_multipliers_with_small_molar_concentration(self):
        from kinet_fitting import fit_model_with_auto_k_max
        t=np.linspace(0,200,41); c0=1e-5
        c=np.vstack((np.exp(-.02*t),1-np.exp(-.02*t)))*c0
        reference=np.array([[1.,.1],[.2,1.2],[.8,.4]])*1e5
        scales=np.array([.8,1.2])
        experiment=Experiment(t=t,wavelength=np.array([400.,450.,500.]),
                              absorbance=(reference*scales)@c)
        fitted,_=fit_model_with_auto_k_max('a_to_b',experiment,c0=c0,n_components=2,
            method='nnls',k_bounds=(1e-4,1),auto_expand=False,expand_factor=10,
            max_expand_steps=0,known_spectra=reference,known_species=('A','B'),
            known_spectrum_scale_bounds={'A':(.7,.9),'B':(1.1,1.3)})
        self.assertAlmostEqual(fitted.params['k'],.02,delta=1e-4)
        self.assertAlmostEqual(fitted.known_spectrum_scales['A'],.8,delta=1e-3)
        self.assertAlmostEqual(fitted.known_spectrum_scales['B'],1.2,delta=1e-3)
        self.assertLess(np.sqrt(np.mean(fitted.residuals**2)),1e-4)

    def test_amplitude_bounds_are_validated_and_legacy_defaults_remain(self):
        from kinet_fitting import direct_parameter_names_and_bounds, known_scale_parameter_name
        name=known_scale_parameter_name('A')
        _,bounds=direct_parameter_names_and_bounds(('k',),('A',),(1e-8,.1))
        self.assertEqual(bounds[name],(1e-4,1e4))
        for bad in [(0,1),(2,1),(.5,np.inf)]:
            with self.assertRaises(ValueError):
                direct_parameter_names_and_bounds(('k',),('A',),(1e-8,.1),{'A':bad})

    def test_conflict_with_imported_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            install_endpoint_references(None,('A',),('A','B'),np.ones(2),None,1)
        self.assertEqual(endpoint_spectrum_menu_labels('mbfe3_sulfide_autocatalytic',True,None),('MbFeIII-SH','MbFeII'))

if __name__=='__main__': unittest.main()
