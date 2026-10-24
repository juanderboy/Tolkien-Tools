"""Interactive inspection of absorption-equivalent reversible kinetic fits."""
from __future__ import annotations

import numpy as np
from kinet_linalg import concentration_point_indices


def equivalent_candidate_indices(result):
    """Retain the solver's RMSE tolerance, independently of chemical flags."""
    diagnostics = result.selection_diagnostics
    threshold = diagnostics['minimum_rmse_mAU'] * (1 + diagnostics['relative_rmse_tolerance']) + 1e-9
    unique = {}
    for index, record in enumerate(diagnostics['candidates']):
        if record['rmse_mAU'] > threshold:
            continue
        key = tuple(round(np.log(value), 9) for _, value in sorted(record['parameters'].items()))
        unique.setdefault(key, index)
        if index == diagnostics.get('selected_candidate_index'):
            unique[key] = index
    return list(unique.values())


class CandidateExplorer:
    """Browse complete solutions; only explicit acceptance changes the result."""
    def __init__(self, experiment, result, input_filename=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, CheckButtons
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        self.experiment = experiment
        self.automatic = result
        self.accepted = None
        self.current = result
        self.records = result.selection_diagnostics['candidates']
        self.indices = equivalent_candidate_indices(result)
        self.cache = {result.selection_diagnostics['selected_candidate_index']: result}
        self.include_flagged = not any(self.records[i]['chemical_pass'] for i in self.indices)
        self.point_ids = concentration_point_indices(experiment.t)
        self.c0 = float(result.c[:, 0].sum())
        # Cache only species spectra for the background, not every full data matrix.
        self.background_spectra = {}
        for index in self.indices:
            self.background_spectra[index] = (result.candidate_spectra[index]
                if result.candidate_spectra is not None else self.get_candidate(index).spectra)
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.subplots_adjust(bottom=.27, top=.91, hspace=.40, wspace=.30)
        self.fig.suptitle((input_filename + ' | ' if input_filename else '') +
                          'Equivalent reversible fits: inspect and choose')
        self.ax_zoom, self.ax_spectra, self.ax_conc, self.ax_res = axes.flat
        self.ax_zoom.set(xlabel='Wavelength (nm)', ylabel='epsilon (M^-1 cm^-1)',
                         title='B family (gray); current A/B/C (colored)', xlim=(400, 450))
        self.ax_spectra.set(xlabel='Wavelength (nm)', ylabel='epsilon (M^-1 cm^-1)',
                            title='Species spectra, full fitted range')
        self.ax_conc.set(xlabel='Time', ylabel='Concentration (uM)',
                         title='Model (lines) and spectral projection (points)')
        self.ax_conc.set_ylim(-.035*self.c0*1e6, 1.035*self.c0*1e6)
        self.ax_res.set(xlabel='Wavelength (nm)', ylabel='Time',
                        title='Residuals: experimental - simulated (mAU)')
        self.background = LineCollection([], colors='0.65', linewidths=.6, alpha=.25, rasterized=True)
        self.ax_zoom.add_collection(self.background)
        self.spectral_lines = []
        self.zoom_lines = []
        self.conc_lines = []
        self.points = []
        for j, color in enumerate(('black', '#B23B3B', '#245B85')):
            label = 'ABC'[j] + ': ' + result.species_labels[j]
            self.spectral_lines.append(self.ax_spectra.plot([], [], color=color, label=label)[0])
            self.zoom_lines.append(self.ax_zoom.plot([], [], color=color, lw=1.7 if j==1 else 1.)[0])
            self.conc_lines.append(self.ax_conc.plot([], [], color=color, label='ABC'[j])[0])
            self.points.append(self.ax_conc.scatter([], [], s=18, color=color,
                               edgecolors='white', linewidths=.3, zorder=3))
        self.ax_spectra.legend(fontsize=8)
        self.ax_conc.legend(fontsize=8)
        cmap = LinearSegmentedColormap.from_list('candidate_residuals',
            ['#315F7B','#9DBFD0','#F9F9F6','#E0B89B','#985B40'])
        def edges(values):
            if len(values)==1:
                return np.array([values[0]-.5,values[0]+.5])
            return np.r_[values[0]-(values[1]-values[0])/2,
                         (values[:-1]+values[1:])/2,
                         values[-1]+(values[-1]-values[-2])/2]
        self.mesh = self.ax_res.pcolormesh(edges(experiment.wavelength), edges(experiment.t),
            (result.residuals*1000).T, cmap=cmap, vmin=-20, vmax=20,
            shading='flat', rasterized=True)
        self.fig.colorbar(self.mesh, ax=self.ax_res, label='mAU', ticks=[-20,0,20], extend='both')
        self.info = self.fig.text(.07, .165, '', family='monospace', fontsize=8, va='top')
        self.fig.text(.07,.025,'Accept exports the displayed solution. Closing this window keeps the automatic solution.',fontsize=9)
        self.slider = Slider(self.fig.add_axes([.16,.195,.65,.025]),
                             'Candidate', 0, max(len(self.indices)-1,1), valinit=0, valstep=1)
        self.check = CheckButtons(self.fig.add_axes([.70,.045,.27,.055]),
                                 ['Include chemically flagged fits'], [self.include_flagged])
        self.accept_button = Button(self.fig.add_axes([.08,.05,.25,.04]), 'Accept displayed solution')
        self.reset_button = Button(self.fig.add_axes([.36,.05,.25,.04]), 'Recommended solution')
        self.slider.on_changed(self.on_slide)
        self.check.on_clicked(self.toggle_flags)
        self.accept_button.on_clicked(self.accept)
        self.reset_button.on_clicked(self.reset)
        self.rebuild_pool(result.selection_diagnostics['selected_candidate_index'])

    def get_candidate(self, index):
        if index in self.cache:
            return self.cache[index]
        candidate = self.automatic.candidate_reconstructor(index)
        # Bounded cache avoids keeping a wavelength x time matrix per candidate.
        if len(self.cache) >= 3:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = candidate
        return candidate

    def rebuild_pool(self, preferred_index=None):
        self.pool = [i for i in self.indices if self.include_flagged or self.records[i]['chemical_pass']]
        self.pool.sort(key=lambda i: (float(self.background_spectra[i][:,1].max()),
                                     self.records[i]['parameters']['k_slow']))
        selected = self.pool.index(preferred_index) if preferred_index in self.pool else 0
        self.slider.eventson = False
        self.slider.valmax = max(len(self.pool)-1,1)
        self.slider.ax.set_xlim(0,self.slider.valmax)
        self.slider.set_val(selected)
        self.slider.eventson = True
        self.slider.set_active(len(self.pool)>1)
        subset = np.unique(np.linspace(0,len(self.pool)-1,min(80,len(self.pool))).astype(int))
        wave = self.experiment.wavelength
        self.background.set_segments([np.column_stack((wave,self.background_spectra[self.pool[k]][:,1])) for k in subset])
        self.update(selected)

    def on_slide(self, value):
        self.update(min(int(value),len(self.pool)-1))

    def update(self, position):
        self.slider.valtext.set_text(f"{position+1}/{len(self.pool)}")
        self.current_index = self.pool[position]
        self.current = candidate = self.get_candidate(self.current_index)
        wave, times = self.experiment.wavelength, self.experiment.t
        for j in range(3):
            self.spectral_lines[j].set_data(wave,candidate.spectra[:,j])
            self.zoom_lines[j].set_data(wave,candidate.spectra[:,j])
            self.conc_lines[j].set_data(times,candidate.c[j]*1e6)
            self.points[j].set_offsets(np.column_stack((times[self.point_ids],
                candidate.estimated_concentrations[j,self.point_ids]*1e6)))
        self.ax_spectra.relim(); self.ax_spectra.autoscale_view()
        self.ax_zoom.relim(); self.ax_zoom.autoscale_view(scalex=False)
        self.ax_conc.relim(); self.ax_conc.autoscale_view(scaley=False)
        self.mesh.set_array((1000*candidate.residuals).T.ravel())
        record = self.records[self.current_index]
        p = candidate.params
        minimum = self.automatic.selection_diagnostics['minimum_rmse_mAU']
        increase = 100*(record['rmse_mAU']/minimum-1) if minimum>0 else 0.
        flag = 'PASS' if record['chemical_pass'] else 'FLAGGED: outside configured chemical bounds'
        rms = ' / '.join(f'{100*x:.2f}' for x in record['profile_rms'])
        self.info.set_text(
            f"{position+1}/{len(self.pool)} shown; {len(self.indices)} RMSE-equivalent fits | chemistry: {flag}\n"
            f"k_on={p['k_on']:.5g}  k_off={p['k_off']:.5g}  k_red={p['k_slow']:.5g} s^-1\n"
            f"RMSE={record['rmse_mAU']:.5g} mAU ({increase:+.4f}% vs minimum) | RMS profiles A/B/C: {rms}%\n"
            f"lambda B max={record['peaks_nm'][1]:.0f} nm | "
            f"epsilon B max={candidate.spectra[:,1].max():.5g} | B/C={record['B_peak_over_C_peak']:.3f} | "
            f"B409/Bmax={record['B409_over_max']:.3f}")
        self.info.set_color('black' if record['chemical_pass'] else '#9B2929')
        self.accept_button.label.set_text('Accept displayed solution' if record['chemical_pass'] else 'Accept flagged solution')
        self.fig.canvas.draw_idle()

    def toggle_flags(self, _label):
        current = self.current_index
        self.include_flagged = not self.include_flagged
        if not self.include_flagged and not any(self.records[i]['chemical_pass'] for i in self.indices):
            self.include_flagged = True
            self.check.eventson = False
            self.check.set_active(0)
            self.check.eventson = True
        self.rebuild_pool(current)

    def reset(self, _event=None):
        recommended = self.automatic.selection_diagnostics['selected_candidate_index']
        self.rebuild_pool(recommended)

    def accept(self, _event=None):
        import matplotlib.pyplot as plt
        diagnostics = dict(self.current.selection_diagnostics)
        diagnostics.update(selection_mode='manual',
            automatic_selected=self.automatic.selection_diagnostics['selected'],
            manual_chemical_filter_passed=bool(self.records[self.current_index]['chemical_pass']),
            chemical_filter_passed=bool(self.records[self.current_index]['chemical_pass']),
            explorer_order='increasing intermediate peak absorptivity')
        self.accepted = self.current
        # Replace the dictionary rather than mutating the automatic diagnostics.
        from dataclasses import replace
        self.accepted = replace(self.current,selection_diagnostics=diagnostics)
        plt.close(self.fig)


def explore_candidates(experiment, result, input_filename=None):
    """Return the explicitly accepted candidate, or the unchanged automatic fit."""
    import matplotlib
    import matplotlib.pyplot as plt
    if result.candidate_reconstructor is None or result.selection_diagnostics is None:
        return result
    if 'agg' == matplotlib.get_backend().lower() or matplotlib.get_backend().lower() in {'pdf','svg','ps','template','cairo'}:
        print('Candidate explorer needs an interactive Matplotlib backend; keeping automatic selection.')
        return result
    print('Inspect equivalent fits with the slider; Accept selects a complete fit. Close keeps the automatic result.')
    explorer = CandidateExplorer(experiment,result,input_filename)
    plt.show(block=True)
    return result if explorer.accepted is None else explorer.accepted
