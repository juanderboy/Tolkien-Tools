#!/usr/bin/env python3
"""Constrained MCR-ALS helpers for exploratory kinetic decompositions.

The routines in this module deliberately keep the kinetic model soft.  A
kinetic concentration profile can be supplied as a reference, but the ALS
updates are allowed to move away from it.  This is useful for diagnosing
whether a hard A -> B -> C fit is forcing kinetic errors into the recovered
intermediate spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _validate_matrix(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _update_spectra(
    absorbance: np.ndarray,
    concentrations: np.ndarray,
    known_spectra: np.ndarray | None,
) -> np.ndarray:
    """Update E from D ~= E C with nonnegative free spectra."""
    n_wavelengths = absorbance.shape[0]
    n_components = concentrations.shape[0]
    if known_spectra is None:
        known_spectra = np.full((n_wavelengths, n_components), np.nan)

    known_mask = np.all(np.isfinite(known_spectra), axis=0)
    spectra = np.zeros((n_wavelengths, n_components), dtype=float)
    spectra[:, known_mask] = known_spectra[:, known_mask]
    free = np.where(~known_mask)[0]
    if free.size == 0:
        return spectra

    target = absorbance - spectra[:, known_mask] @ concentrations[known_mask]
    design = concentrations[free].T
    for wavelength in range(n_wavelengths):
        spectra[wavelength, free], _ = nnls(design, target[wavelength])
    return spectra


def _update_concentrations(
    absorbance: np.ndarray,
    spectra: np.ndarray,
    kinetic_reference: np.ndarray | None,
    kinetic_weight: float,
) -> np.ndarray:
    """Update C from D ~= E C with optional soft kinetic anchoring."""
    n_components = spectra.shape[1]
    n_times = absorbance.shape[1]
    design = spectra
    if kinetic_reference is not None and kinetic_weight > 0:
        penalty = np.sqrt(kinetic_weight) * np.eye(n_components)
        design = np.vstack((design, penalty))

    concentrations = np.zeros((n_components, n_times), dtype=float)
    for time_index in range(n_times):
        target = absorbance[:, time_index]
        if kinetic_reference is not None and kinetic_weight > 0:
            target = np.concatenate(
                (
                    target,
                    np.sqrt(kinetic_weight) * kinetic_reference[:, time_index],
                )
            )
        concentrations[:, time_index], _ = nnls(design, target)
    return concentrations


def _enforce_closure(concentrations: np.ndarray, total: float) -> np.ndarray:
    """Normalize every time point to the supplied total concentration."""
    concentrations = np.maximum(concentrations, 0.0)
    sums = concentrations.sum(axis=0)
    valid = sums > 0
    concentrations[:, valid] *= total / sums[valid]
    return concentrations


def mcr_als_decompose(
    absorbance: np.ndarray,
    n_components: int,
    initial_concentrations: np.ndarray,
    known_spectra: np.ndarray | None = None,
    closure_total: float | None = None,
    kinetic_reference: np.ndarray | None = None,
    kinetic_weight: float = 0.0,
    max_iter: int = 200,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run a deterministic nonnegative, closure-constrained MCR-ALS fit.

    Parameters
    ----------
    absorbance:
        Data matrix with shape ``n_wavelengths x n_times``.
    n_components:
        Number of chemical components.
    initial_concentrations:
        Initial nonnegative concentration estimate with shape
        ``n_components x n_times``.  A kinetic profile is a useful choice.
    known_spectra:
        Optional matrix with shape ``n_wavelengths x n_components``.  Complete
        finite columns are held fixed; NaN columns are updated by ALS.
    closure_total:
        If supplied, each concentration column is normalized to this total.
    kinetic_reference:
        Optional concentration profile used as a soft prior.
    kinetic_weight:
        Weight of the kinetic prior relative to one spectral observation per
        wavelength.  Zero gives unconstrained ALS updates for concentrations.

    Returns
    -------
    spectra, concentrations, error, iterations
        The recovered E and C matrices, Frobenius residual norm, and number of
        completed ALS iterations.
    """
    d = _validate_matrix("absorbance", absorbance, 2)
    c = _validate_matrix("initial_concentrations", initial_concentrations, 2).copy()
    if c.shape != (n_components, d.shape[1]):
        raise ValueError("initial_concentrations has an incompatible shape")
    if np.any(c < -1e-10):
        raise ValueError("initial_concentrations must be nonnegative")
    c = np.maximum(c, 0.0)
    if kinetic_reference is not None:
        reference = _validate_matrix("kinetic_reference", kinetic_reference, 2)
        if reference.shape != c.shape:
            raise ValueError("kinetic_reference has an incompatible shape")
        if np.any(reference < -1e-10):
            raise ValueError("kinetic_reference must be nonnegative")
        reference = np.maximum(reference, 0.0)
    else:
        reference = None
    if kinetic_weight < 0:
        raise ValueError("kinetic_weight must be nonnegative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if closure_total is not None and closure_total <= 0:
        raise ValueError("closure_total must be positive")

    if known_spectra is not None:
        known = np.asarray(known_spectra, dtype=float)
        if known.ndim != 2 or np.any(np.isinf(known)):
            raise ValueError("known_spectra must be a 2D array without infinities")
        if known.shape != (d.shape[0], n_components):
            raise ValueError("known_spectra has an incompatible shape")
        partially_known = np.any(np.isfinite(known), axis=0) & ~np.all(
            np.isfinite(known), axis=0
        )
        if np.any(partially_known):
            raise ValueError("known_spectra columns must be complete or all NaN")
    else:
        known = None

    if closure_total is not None:
        c = _enforce_closure(c, closure_total)

    previous_error = np.inf
    spectra = _update_spectra(d, c, known)
    for iteration in range(1, max_iter + 1):
        c = _update_concentrations(d, spectra, reference, kinetic_weight)
        if closure_total is not None:
            c = _enforce_closure(c, closure_total)
        spectra = _update_spectra(d, c, known)
        residual = d - spectra @ c
        error = float(np.linalg.norm(residual))
        if np.isfinite(previous_error) and abs(previous_error - error) <= tolerance * max(
            1.0, previous_error
        ):
            return spectra, c, error, iteration
        previous_error = error

    return spectra, c, previous_error, max_iter
