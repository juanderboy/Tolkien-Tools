"""Reoptimized objective profiles for the two-form activation model.

Contours describe compensation in the fitted objective, not confidence regions.
The numerical grids retain failed optimizations and competing minima explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from kinet_activation import HIGH, LOW, NAMES, evaluate

FALLBACK_STEPS = np.array([1., 1., 40., 40., 60., 1.])


def analyze_profiles(result, grid_points=33, extent_se=3., max_nfev=600, progress=None):
    """Fix one/two free parameters and refit all remaining free parameters.

    Warm starts and distinct converged multistart basins supplement the fitted
    optimum at each grid node. A fixed pKa remains fixed in every refit.
    """
    if not isinstance(grid_points, (int, np.integer)) or grid_points < 3 or grid_points % 2 != 1:
        raise ValueError('Profile grid points must be an odd integer >= 3.')
    if not np.isfinite(extent_se) or extent_se <= 0:
        raise ValueError('Profile extent must be finite and positive.')
    if max_nfev < 1:
        raise ValueError('Profile max_nfev must be >= 1.')
    records = result.get('records', result['predictions'])
    temperature = np.array([r['T_K'] for r in records])
    ph = np.array([r['pH'] for r in records])
    observed = np.log([r['k'] for r in records])
    sigma = np.array([r.get('sigma_log_k') or 1. for r in records])
    reference = result['configuration']['reference_K']
    best = np.array([result['parameters'][name] for name in NAMES])
    free = list(range(6 if result['configuration']['fixed_pka'] is None else 5))
    residuals = (evaluate(best, temperature, ph, reference)[0] - observed) / sigma
    baseline = float(residuals @ residuals)
    uncertainty = result['uncertainty']
    covariance = uncertainty.get('covariance')
    correlation = np.full((6, 6), np.nan)
    if covariance is not None:
        covariance = np.array(covariance)
        sd = np.sqrt(np.maximum(0., np.diag(covariance)))
        denominator = np.outer(sd, sd)
        np.divide(covariance, denominator, out=correlation, where=denominator > 0)
        correlation = np.clip(correlation, -1., 1.)

    axes = {}
    axis_sources = {}
    for i in free:
        se = uncertainty['standard_errors'][NAMES[i]]
        if se is not None and np.isfinite(se) and se > 1e-8:
            halfwidth = extent_se * se
            axis_sources[NAMES[i]] = 'local SE'
        else:
            halfwidth = extent_se * FALLBACK_STEPS[i]
            axis_sources[NAMES[i]] = 'fallback exploration scale (not an SE)'
        # Keep the fitted value exactly on every axis, also near a bound.
        lower = max(LOW[i], best[i] - halfwidth)
        upper = min(HIGH[i], best[i] + halfwidth)
        half = grid_points // 2
        axes[i] = np.unique(np.r_[np.linspace(lower, best[i], half+1),
                                  np.linspace(best[i], upper, half+1)])

    seeds = [best]
    for fit in sorted(result.get('multistart', []), key=lambda f: f['objective']):
        candidate = np.array(fit['parameters'])
        if not fit['success'] or not np.all(np.isfinite(candidate)):
            continue
        if all(np.linalg.norm((candidate-s)/FALLBACK_STEPS) > .05 for s in seeds):
            seeds.append(candidate)
        if len(seeds) == 3:
            break

    def refit(fixed, warm=()):
        nuisance = [i for i in free if i not in fixed]
        candidates = []
        for start in [*seeds, *warm]:
            if not np.all(np.isfinite(start)):
                continue
            p = np.array(start, copy=True)
            # In particular, never use another pKa in fixed-pKa mode.
            p[5] = best[5] if 5 not in free else p[5]
            for i, value in fixed.items():
                p[i] = value
            if any(np.allclose(p[nuisance], previous[nuisance], atol=1e-10, rtol=1e-10)
                   for previous in candidates):
                continue
            candidates.append(p)
        fits = []
        for start in candidates:
            def unpack(values):
                p = start.copy()
                p[nuisance] = values
                return p
            def fun(values):
                return (evaluate(unpack(values), temperature, ph, reference)[0]-observed)/sigma
            def jac(values):
                return evaluate(unpack(values), temperature, ph, reference)[1][:, nuisance]/sigma[:, None]
            try:
                fit = least_squares(fun, np.clip(start[nuisance], LOW[nuisance]+1e-9, HIGH[nuisance]-1e-9),
                    jac=jac, bounds=(LOW[nuisance], HIGH[nuisance]), x_scale='jac',
                    xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev)
            except (ValueError, FloatingPointError):
                continue
            objective = float(fit.fun @ fit.fun)
            if fit.success and np.isfinite(objective):
                fits.append((objective, unpack(fit.x), int(fit.nfev)))
        if not fits:
            return np.nan, np.full(6, np.nan), 0
        return min(fits, key=lambda f: f[0])

    profiles = {}
    surfaces = {}
    failures = 0
    minimum = baseline
    minimum_parameters = best.copy()
    for i in free:
        axis = axes[i]
        objective = np.full(len(axis), np.nan)
        parameters = np.full((len(axis), 6), np.nan)
        for indices in (np.flatnonzero(axis >= best[i]), np.flatnonzero(axis < best[i])[::-1]):
            warm = best
            for k in indices:
                objective[k], parameters[k], _ = refit({i: axis[k]}, [warm])
                if np.isfinite(objective[k]):
                    warm = parameters[k]
        profiles[i] = dict(values=axis, objective=objective, parameters=parameters,
                           success=np.isfinite(objective))
        failures += int(np.count_nonzero(~np.isfinite(objective)))
        if np.any(np.isfinite(objective)) and np.nanmin(objective) < minimum:
            k = int(np.nanargmin(objective))
            minimum, minimum_parameters = float(objective[k]), parameters[k]
        if progress:
            progress(f'Profile {NAMES[i]} complete.')

    for a, i in enumerate(free):
        for j in free[a+1:]:
            x, y = axes[i], axes[j]
            objective = np.full((len(y), len(x)), np.nan)
            parameters = np.full((len(y), len(x), 6), np.nan)
            for row, y_value in enumerate(y):
                warm = best
                # Alternate traversal to preserve warm starts across adjacent rows.
                indices = range(len(x)) if row % 2 == 0 else reversed(range(len(x)))
                for col in indices:
                    neighbors = [warm, profiles[i]['parameters'][col], profiles[j]['parameters'][row]]
                    if row:
                        neighbors.append(parameters[row-1, col])
                    objective[row, col], parameters[row, col], _ = refit({i: x[col], j: y_value}, neighbors)
                    if np.isfinite(objective[row, col]):
                        warm = parameters[row, col]
            success = np.isfinite(objective)
            surfaces[i, j] = dict(x=x, y=y, objective=objective, parameters=parameters, success=success)
            failures += int(np.count_nonzero(~success))
            if np.any(success) and np.nanmin(objective) < minimum:
                row, col = np.unravel_index(np.nanargmin(objective), objective.shape)
                minimum, minimum_parameters = float(objective[row, col]), parameters[row, col]
            if progress:
                progress(f'Compensation map {NAMES[i]} / {NAMES[j]} complete ({np.count_nonzero(~success)} failed cells).')

    # An exact/noiseless fit has no usable SSE ratio. Do not divide by zero or
    # manufacture a variance; show objective increases in that case instead.
    display = 'objective_ratio' if baseline > 1e-12 else 'objective_increase'
    notes = ['Contours are descriptive objective levels, not confidence regions.',
             'Local r is a linearized parameter correlation, not a correlation of measurements.',
             'Profiles include the supplied weighting and keep a fixed pKa fixed.',
             'Finite grids and multiple starts do not establish global identifiability.']
    if any(source != 'local SE' for source in axis_sources.values()):
        notes.append('Some axes use fallback exploration scales because local SE is unavailable or effectively zero.')
    if failures:
        notes.append(f'{failures} grid nodes failed to converge and are masked; they are not high-objective evidence.')
    improved = minimum < baseline - max(1e-10, baseline*1e-6)
    if improved:
        notes.append('Profiles found a lower objective than the original fit; refit before interpreting uncertainty.')
    return dict(axes=axes, profiles=profiles, surfaces=surfaces, correlation=correlation, best_parameters=best,
                free_indices=free, minimum_parameters=minimum_parameters,
                manifest=dict(parameter_names=list(NAMES), free_parameters=[NAMES[i] for i in free],
                    fixed_parameters={} if 5 in free else {'pKa_ref': float(best[5])},
                    reference_K=reference, grid_points=grid_points, extent_local_SE=extent_se,
                    axis_sources=axis_sources,
                    axis_ranges={NAMES[i]: dict(minimum=float(axes[i][0]), maximum=float(axes[i][-1]),
                                                points=len(axes[i])) for i in free},
                    fallback_steps=dict(zip(NAMES, FALLBACK_STEPS.tolist())),
                    fitted_parameters=dict(zip(NAMES, best.tolist())),
                    lowest_profile_parameters=dict(zip(NAMES, minimum_parameters.tolist())),
                    source_sha256=result.get('source_sha256'),
                    seed=result['configuration'].get('seed'),
                    pair_count=len(surfaces), panel_count=len(free)**2,
                    baseline_objective=baseline, lowest_profile_objective=minimum,
                    lower_minimum_found=bool(improved), failed_nodes=failures,
                    display=display, weighting=result['configuration']['weighting'],
                    objective=result['configuration']['objective'], max_nfev=max_nfev,
                    xtol=1e-10, ftol=1e-10, gtol=1e-10, x_scale='jac',
                    optimizer='scipy.optimize.least_squares; analytic Jacobian; nuisance parameters refitted',
                    initialization='fitted optimum, up to two distinct converged multistart basins, neighboring solutions',
                    notes=notes))


def export_profiles(analysis, output, plot=True):
    """Export all objective grids and an optional mirrored profile matrix."""
    output = Path(output)
    directory = output/'matrix_pair_grids'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(analysis['manifest'])
    manifest['grid_files'] = []
    for (i, j), surface in analysis['surfaces'].items():
        name = f'matrix_pair_grids/pair_{i}_{j}.npz'
        np.savez_compressed(output/name, **surface)
        manifest['grid_files'].append(name)
    profiles = {}
    for i, profile in analysis['profiles'].items():
        profiles[NAMES[i]] = {key: value.tolist() for key, value in profile.items()}
    def clean(value):
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (output/'profiles.json').write_text(json.dumps(clean(profiles), indent=2, allow_nan=False)+'\n')
    (output/'profile_matrix_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    correlation = clean(analysis['correlation'].tolist())
    (output/'parameter_correlation.json').write_text(json.dumps(dict(parameter_names=list(NAMES),
        correlation=correlation, method='local linearized covariance'), indent=2, allow_nan=False)+'\n')
    if plot:
        plot_profile_matrix(analysis, output)
    return manifest


def plot_profile_matrix(analysis, output):
    """Draw one-dimensional profiles and pairwise reoptimized objective maps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    manifest = analysis['manifest']
    free = analysis['free_indices']
    axes = analysis['axes']
    baseline = manifest['baseline_objective']
    ratio = manifest['display'] == 'objective_ratio'
    # Keep the original optimum as the reference even if profiling finds a
    # competing minimum; the latter is reported separately in the manifest.
    centers = dict(enumerate(analysis['best_parameters']))
    def display(q):
        return q/baseline if ratio else np.maximum(q-baseline, 0.)
    if ratio:
        vmin = min(1., manifest['lowest_profile_objective']/baseline)
        vmax = 3.
        levels = [1.05, 1.2, 1.5, 2., 3.]
        color_label = 'Weighted SSE / fitted SSE' if manifest['weighting'] != 'equal' else 'SSE / fitted SSE'
    else:
        finite = np.concatenate([s['objective'][s['success']] for s in analysis['surfaces'].values()])
        vmax = max(float(np.percentile(np.maximum(finite-baseline, 0.), 70)) if finite.size else 1., 1e-6)
        vmin = 0.
        levels = [vmax*x for x in (.1, .25, .5, .75)]
        color_label = 'Weighted SSE increase' if manifest['weighting'] != 'equal' else 'SSE increase'
    labels = [r'$\ln k_{HA}(T_{ref})$', r'$\ln k_A(T_{ref})$',
              r'$\Delta H^{‡}_{HA}$'+'\n(kJ/mol)', r'$\Delta H^{‡}_A$'+'\n(kJ/mol)',
              r'$\Delta H_a$'+'\n(kJ/mol)', r'$pK_a(T_{ref})$']
    n = len(free)
    fig, panels = plt.subplots(n, n, figsize=(3*n, 3*n), squeeze=False)
    fig.subplots_adjust(left=.1, right=.97, bottom=.14, top=.94, wspace=.18, hspace=.18)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('YlGnBu')
    for row, i in enumerate(free):
        for col, j in enumerate(free):
            ax = panels[row, col]
            ax.tick_params(labelsize=8)
            ax.set_xlim(axes[j][0], axes[j][-1])
            ax.locator_params(axis='x', nbins=3)
            if i == j:
                profile = analysis['profiles'][i]
                ax.plot(profile['values'], display(profile['objective']), color='#153b64', lw=1.8)
                ax.axvline(centers[i], color='#c63334', ls='--', lw=1)
                ax.plot(centers[i], 1. if ratio else 0., '+', color='#c63334', ms=9)
                ax.set_ylim((min(.96, vmin), 3.05) if ratio else (-.03*vmax, vmax))
                if ratio:
                    ax.set_yticks([1, 2, 3])
                    ax.axhline(1.05, color='.5', ls=':', lw=.8)
                ax.grid(alpha=.15)
                ax.text(.96, .08, 'SSE ratio' if ratio else 'SSE increase', transform=ax.transAxes,
                        ha='right', fontsize=8)
            else:
                a, b = sorted((i, j))
                surface = analysis['surfaces'][a, b]
                x, y, z = surface['x'], surface['y'], display(surface['objective'])
                if j == b:
                    x, y, z = y, x, z.T
                z = np.ma.masked_invalid(z)
                ax.set_facecolor('#dddddd')
                if z.count():
                    ax.contourf(x, y, z, levels=np.linspace(vmin, vmax, 41), cmap=cmap, norm=norm, extend='both')
                    inside = [level for level in levels if z.min() < level < z.max()]
                    if inside:
                        contours = ax.contour(x, y, z, levels=inside, colors='#182e42', linewidths=.6)
                        labels_to_show = [level for level in inside if not ratio or level in (1.2, 2.)]
                        if labels_to_show:
                            ax.clabel(contours, levels=labels_to_show, fmt='%g', fontsize=6)
                ax.plot(centers[j], centers[i], '+', color='#c63334', ms=10, mew=1.5)
                ax.set_ylim(axes[i][0], axes[i][-1])
                ax.locator_params(axis='y', nbins=3)
                r = analysis['correlation'][i, j]
                ax.text(.04, .94, f'r = {r:+.3f}' if np.isfinite(r) else 'r unavailable',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(facecolor='white', alpha=.82, edgecolor='none', pad=2))
            if row == 0:
                ax.set_title(labels[j], fontsize=11, pad=10)
            if col == 0:
                ax.set_ylabel(labels[i], fontsize=11)
            if row == n-1:
                ax.set_xlabel(labels[j], fontsize=11)
            else:
                ax.tick_params(labelbottom=False)
            if col and row != col:
                ax.tick_params(labelleft=False)
    cax = fig.add_axes([.28, .055, .44, .015])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='max')
    colorbar.set_label(color_label)
    if ratio:
        colorbar.set_ticks([1, 1.2, 1.5, 2, 2.5, 3])
    fig.suptitle(f'Reoptimized parameter profiles | Tref = {manifest["reference_K"]:g} K', fontsize=14)
    fig.text(.5, .025, 'Other free parameters refitted at each node. Descriptive contours, not confidence regions.',
             ha='center', fontsize=10)
    suffix = 'six_parameters' if n == 6 else 'five_parameters'
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(Path(output)/f'profile_compensation_matrix_{suffix}.{extension}', dpi=220)
    plt.close(fig)
