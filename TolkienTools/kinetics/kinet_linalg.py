#!/usr/bin/env python3
"""Linear algebra helpers for TolKinet."""

from __future__ import annotations

import numpy as np


def factor_analysis(absorbance: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, W and singular values from A = U S Vt, with A ~= Q W.

    Q is the truncated spectral basis. W contains the corresponding temporal
    profiles in factor-analysis coordinates.
    """
    u, singular_values, vt = np.linalg.svd(absorbance, full_matrices=False)
    q = u[:, :n_components]
    w = singular_values[:n_components, None] * vt[:n_components, :]
    return q, w, singular_values



def concentration_point_indices(times: np.ndarray) -> np.ndarray:
    """Choose sparse display times independently of fit residuals."""
    times = np.asarray(times)
    targets = np.r_[times[0], 5., 15., 30., np.linspace(times[0], times[-1], 10)]
    return np.unique([int(np.argmin(abs(times - value))) for value in targets
                      if times[0] <= value <= times[-1]])


def projected_concentrations(experiment, result, c0=None) -> np.ndarray:
    """Project spectra with nonnegative concentrations summing to total c0.

    Enumerate simplex faces for small species sets; use constrained least
    squares for larger sets. This is a diagnostic, not a kinetic refit.
    """
    if result.estimated_concentrations is not None:
        return result.estimated_concentrations
    from itertools import combinations

    if c0 is None:
        c0 = float(np.sum(result.c[:, 0]))
    if not np.isfinite(c0) or c0 <= 0:
        raise ValueError("Spectral projection requires a positive total concentration")
    spectra = np.asarray(result.spectra) * c0
    observed = experiment.absorbance
    count = spectra.shape[1]
    if count > 8:
        from scipy.optimize import minimize
        scale = max(float(np.linalg.norm(spectra)), 1e-30)
        a, y = spectra / scale, observed / scale
        fractions = []
        for frame in y.T:
            opt = minimize(lambda f: np.sum((a @ f - frame)**2),
                           np.full(count, 1. / count),
                           jac=lambda f: 2 * a.T @ (a @ f - frame),
                           bounds=[(0., 1.)] * count,
                           constraints={'type': 'eq', 'fun': lambda f: f.sum() - 1,
                                        'jac': lambda f: np.ones(count)},
                           method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
            if not opt.success:
                raise ValueError(f"Spectral projection failed: {opt.message}")
            fractions.append(opt.x)
        return np.asarray(fractions).T * c0
    best = np.zeros((count, observed.shape[1]))
    errors = np.full(observed.shape[1], np.inf)
    for size in range(1, count + 1):
        for support in combinations(range(count), size):
            ids = list(support)
            trial = np.zeros_like(best)
            if size == 1:
                trial[ids[0]] = 1.
            else:
                base = spectra[:, ids[0], None]
                coefficients = np.linalg.lstsq(
                    spectra[:, ids[1:]] - base, observed - base, rcond=None)[0]
                trial[ids[1:]] = coefficients
                trial[ids[0]] = 1. - coefficients.sum(axis=0)
            valid = (trial >= -1e-12).all(axis=0)
            trial = np.maximum(trial, 0.)
            trial /= trial.sum(axis=0)
            error = np.sum((observed - spectra @ trial)**2, axis=0)
            improve = valid & (error < errors)
            best[:, improve] = trial[:, improve]
            errors[improve] = error[improve]
    return best * c0
