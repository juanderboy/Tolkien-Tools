#!/usr/bin/env python3
"""Global kinetic fitting routines for TolKinet."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls

from kinet_common import Experiment, FitResult, MODEL_SPECIES, PARAMETER_LABELS
from kinet_linalg import factor_analysis
from kinet_mcr import mcr_als_decompose
from kinet_models import (
    concentration_profile_a_rev_b_to_c,
    concentration_profile_a_to_b,
    concentration_profile_a_to_b_to_c,
    concentration_profile_mbfe3_sulfide_binding_autocatalytic,
    concentration_profile_mbfe3_sulfide_hss_transsulfuration,
    concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto,
    concentration_profile_mbfe3_sulfide_autocatalytic,
)

ProgressCallback = Callable[[dict[str, float], float], None]


def solve_small_nnls_batch(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve many small NNLS problems by enumerating active species sets.

    The spectral fits use very few species, usually two or three. For that
    regime, checking every active subset is much faster than calling scipy.nnls
    once per wavelength, and it gives the same convex least-squares solution.
    """
    n_free = design.shape[1]
    if n_free == 0:
        return np.empty((target.shape[0], 0))
    if n_free > 8:
        spectra = np.zeros((target.shape[0], n_free))
        for i, row in enumerate(target):
            spectra[i, :], _ = nnls(design, row)
        return spectra

    best_spectra = np.zeros((target.shape[0], n_free))
    best_error = np.sum(target**2, axis=1)
    tolerance = 1e-12

    for active_bits in range(1, 1 << n_free):
        active = np.array(
            [(active_bits >> index) & 1 for index in range(n_free)],
            dtype=bool,
        )
        active_design = design[:, active]
        active_solution = target @ np.linalg.pinv(active_design).T
        valid = np.all(active_solution >= -tolerance, axis=1)
        if not np.any(valid):
            continue

        active_solution = np.maximum(active_solution, 0.0)
        residual = target - active_solution @ active_design.T
        error = np.sum(residual**2, axis=1)
        improved = valid & (error < best_error)
        if not np.any(improved):
            continue

        candidate = np.zeros((int(np.count_nonzero(improved)), n_free))
        candidate[:, active] = active_solution[improved]
        best_spectra[improved] = candidate
        best_error[improved] = error[improved]

    return best_spectra


def fit_direct_spectra(
    absorbance: np.ndarray,
    c: np.ndarray,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> np.ndarray:
    """Fit pure spectra for fixed concentration profiles.

    For fixed C(t), each wavelength is an independent least-squares problem:

        A_exp(lambda, :) ~= E(lambda, :) @ C

    With spectra_method="nnls", E(lambda, species) >= 0 is enforced.

    If initial_spectrum_weight > 0, add a soft penalty that keeps the pure
    spectrum of species A close to the first measured spectrum. The weight is
    equivalent to adding that many extra time points to the least-squares fit.

    If fix_initial_spectrum is true, the first species spectrum is fixed to the
    first measured spectrum divided by the total concentration at the first
    time point. Only the remaining free species are fitted.

    If fix_final_spectrum is true, the last species spectrum is fixed to the
    last measured spectrum divided by the total concentration at the last time
    point.
    """
    if initial_spectrum_weight < 0:
        raise ValueError("--initial-spectrum-weight must be nonnegative")
    if fix_initial_spectrum and initial_spectrum_weight > 0:
        raise ValueError(
            "--initial-spectrum-weight cannot be used together with "
            "--fix-initial-spectrum"
        )
    if spectra_method != "nnls":
        raise ValueError(f"Unknown direct spectra method: {spectra_method}")

    if known_spectra is None:
        known_spectra = np.full((absorbance.shape[0], c.shape[0]), np.nan)
    if known_spectra.shape != (absorbance.shape[0], c.shape[0]):
        raise ValueError("known_spectra must have shape n_wavelengths x n_species")

    known_mask = np.any(np.isfinite(known_spectra), axis=0)
    for index, is_known in enumerate(known_mask):
        if is_known and not np.all(np.isfinite(known_spectra[:, index])):
            raise ValueError("Known spectrum columns must be complete and finite")
    known_spectrum_scales = known_spectrum_scales or {}
    if fix_initial_spectrum:
        if known_mask[0]:
            raise ValueError(
                "The first species cannot be provided as a known spectrum and "
                "fixed from the first experimental spectrum at the same time"
            )
        initial_concentration = float(np.sum(c[:, 0]))
        if not np.isfinite(initial_concentration) or initial_concentration <= 0:
            raise ValueError(
                "Cannot fix the first spectrum because the calculated initial "
                "total concentration is not positive"
            )
        known_spectra = known_spectra.copy()
        known_spectra[:, 0] = absorbance[:, 0] / initial_concentration
        known_mask = np.any(np.isfinite(known_spectra), axis=0)
    if fix_final_spectrum:
        final_index = c.shape[0] - 1
        if known_mask[final_index]:
            raise ValueError(
                "The last species cannot be provided as a known spectrum and "
                "fixed from the last experimental spectrum at the same time"
            )
        final_concentration = float(np.sum(c[:, -1]))
        if not np.isfinite(final_concentration) or final_concentration <= 0:
            raise ValueError(
                "Cannot fix the last spectrum because the calculated final "
                "total concentration is not positive"
            )
        known_spectra = known_spectra.copy()
        known_spectra[:, final_index] = absorbance[:, -1] / final_concentration
        known_mask = np.any(np.isfinite(known_spectra), axis=0)
    free_mask = ~known_mask

    spectra = np.zeros((absorbance.shape[0], c.shape[0]))
    spectra[:, known_mask] = known_spectra[:, known_mask]
    for index in np.where(known_mask)[0]:
        spectra[:, index] *= known_spectrum_scales.get(int(index), 1.0)

    target = absorbance - spectra[:, known_mask] @ c[known_mask, :]
    if not np.any(free_mask):
        return spectra

    design = c[free_mask, :].T
    if initial_spectrum_weight > 0:
        if free_mask[0]:
            penalty_row = np.zeros((1, int(np.count_nonzero(free_mask))))
            free_indices = np.where(free_mask)[0]
            a_free_index = int(np.where(free_indices == 0)[0][0])
            penalty_row[0, a_free_index] = np.sqrt(initial_spectrum_weight) * c[0, 0]
            design = np.vstack([design, penalty_row])
            target = np.column_stack(
                [
                    target,
                    np.sqrt(initial_spectrum_weight) * target[:, 0],
                ]
            )

    spectra[:, free_mask] = solve_small_nnls_batch(design, target)
    return spectra



def fit_nonnegative_spectra(
    absorbance: np.ndarray,
    c: np.ndarray,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> np.ndarray:
    """Fit nonnegative pure spectra for fixed concentration profiles."""
    return fit_direct_spectra(
        absorbance,
        c,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_spectrum_scales,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def direct_spectral_error_for_concentrations(
    c: np.ndarray,
    experiment: Experiment,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> float:
    """Objective function for a fixed concentration matrix."""
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_spectrum_scales,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    residuals = experiment.absorbance - spectra @ c
    error_squared = float(np.sum(residuals**2))
    if initial_spectrum_weight > 0:
        initial_mismatch = c[0, 0] * spectra[:, 0] - experiment.absorbance[:, 0]
        error_squared += initial_spectrum_weight * float(np.sum(initial_mismatch**2))
    return float(np.sqrt(error_squared))



def direct_nonnegative_error_for_concentrations(
    c: np.ndarray,
    experiment: Experiment,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> float:
    """Objective function for a fixed concentration matrix using NNLS."""
    return direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_spectrum_scales,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def direct_spectral_error(
    k: float,
    experiment: Experiment,
    c0: float,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> float:
    """Objective function for the A -> B direct spectral fit."""
    if k <= 0:
        return np.inf

    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    return direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_spectrum_scales,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def direct_nonnegative_error(
    k: float,
    experiment: Experiment,
    c0: float,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_spectrum_scales: dict[int, float] | None = None,
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> float:
    """Objective function for the A -> B NNLS fit."""
    if k <= 0:
        return np.inf

    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    return direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_spectrum_scales,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def validate_k_bounds(k_bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate and normalize positive bounds for the kinetic constant."""
    lower, upper = sorted(k_bounds)
    if lower <= 0 or upper <= 0:
        raise ValueError("k bounds must be positive")
    if lower == upper:
        raise ValueError("k bounds must not be equal")
    return lower, upper



def optimize_k(
    objective_for_k,
    k_bounds: tuple[float, float],
) -> float:
    """Minimize an objective as a function of log(k) within positive bounds."""
    k_min, k_max = validate_k_bounds(k_bounds)

    def objective(log_k: float) -> float:
        return objective_for_k(float(np.exp(log_k)))

    opt = minimize_scalar(
        objective,
        bounds=(np.log(k_min), np.log(k_max)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 10000},
    )
    if not opt.success:
        raise RuntimeError(f"k optimization failed: {opt.message}")

    return float(np.exp(opt.x))



def optimize_kinetic_parameters(
    objective_for_params,
    parameter_names: tuple[str, ...],
    k_bounds: tuple[float, float],
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    initial_parameters: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
    optimizer: str = "hybrid",
    max_starts: int | None = None,
) -> dict[str, float]:
    """Minimize an objective as a function of multiple positive constants."""
    if optimizer not in {"hybrid", "powell", "lbfgsb"}:
        raise ValueError(f"Unknown kinetic optimizer: {optimizer}")
    if max_starts is not None and max_starts <= 0:
        raise ValueError("max_starts must be positive")
    k_min, k_max = validate_k_bounds(k_bounds)
    parameter_bounds = parameter_bounds or {}
    bounds: list[tuple[float, float]] = []
    for name in parameter_names:
        bounds.append(parameter_bounds.get(name, (k_min, k_max)))
    for lower_bound, upper_bound in bounds:
        validate_k_bounds((lower_bound, upper_bound))
    log_bounds = [
        (np.log(lower_bound), np.log(upper_bound))
        for lower_bound, upper_bound in bounds
    ]
    lower = np.array([bound[0] for bound in log_bounds])
    upper = np.array([bound[1] for bound in log_bounds])

    center = lower + 0.5 * (upper - lower)
    if initial_parameters:
        unknown = set(initial_parameters) - set(parameter_names)
        if unknown:
            raise ValueError(
                "Initial values provided for unknown parameters: "
                + ", ".join(sorted(unknown))
            )
        for index, name in enumerate(parameter_names):
            if name not in initial_parameters:
                continue
            value = initial_parameters[name]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Initial value for {name} must be positive and finite")
            center[index] = np.clip(np.log(value), lower[index], upper[index])
    starts = [center]
    for i in range(len(parameter_names)):
        for fraction in (0.25, 0.75):
            start = center.copy()
            start[i] = lower[i] + fraction * (upper[i] - lower[i])
            starts.append(start)
    if max_starts is not None:
        starts = starts[:max_starts]

    best = None

    def objective(log_params: np.ndarray) -> float:
        params = {
            name: float(np.exp(value))
            for name, value in zip(parameter_names, log_params)
        }
        value = float(objective_for_params(params))
        if progress_callback is not None:
            progress_callback(params, value)
        return value

    for start in starts:
        clipped_start = np.clip(start, lower, upper)
        if optimizer == "powell":
            opt = minimize(
                objective,
                clipped_start,
                method="Powell",
                bounds=log_bounds,
                # The profiled NNLS objective is only piecewise smooth. Tighter
                # tolerances add hundreds of evaluations without a stable
                # improvement in the fitted residual.
                options={"ftol": 1e-4, "xtol": 1e-4, "maxiter": 300},
            )
            if best is None or opt.fun < best.fun:
                best = opt
            continue

        opt = minimize(
            objective,
            clipped_start,
            method="L-BFGS-B",
            bounds=log_bounds,
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 5000},
        )
        if best is None or opt.fun < best.fun:
            best = opt
        if optimizer == "lbfgsb":
            continue

        opt = minimize(
            objective,
            clipped_start,
            method="Nelder-Mead",
            bounds=log_bounds,
            options={"xatol": 1e-9, "fatol": 1e-7, "maxiter": 20000},
        )
        if best is None or opt.fun < best.fun:
            best = opt

    if best is None or not np.isfinite(best.fun):
        message = "unknown error" if best is None else best.message
        raise RuntimeError(f"k optimization failed: {message}")

    return {
        name: float(np.exp(value))
        for name, value in zip(parameter_names, best.x)
    }


def known_species_indices(
    model: str,
    known_species: tuple[str, ...],
) -> dict[str, int]:
    """Return model-column indices for known species labels."""
    species_to_column = {label: index for index, label in enumerate(MODEL_SPECIES[model])}
    return {
        label: species_to_column[label]
        for label in known_species
    }


def known_scale_parameter_name(label: str) -> str:
    """Return the optimizer parameter name for one known-spectrum scale."""
    return f"scale_{label}"


def known_scale_parameters(
    params: dict[str, float],
    model: str,
    known_species: tuple[str, ...],
) -> dict[int, float]:
    """Convert scale_A-style parameters to column-indexed scales."""
    indices = known_species_indices(model, known_species)
    return {
        column: params[known_scale_parameter_name(label)]
        for label, column in indices.items()
    }


def extract_known_scale_report(
    params: dict[str, float],
    known_species: tuple[str, ...],
) -> dict[str, float]:
    """Return scale factors keyed by species label."""
    return {
        label: params[known_scale_parameter_name(label)]
        for label in known_species
    }


def direct_parameter_names_and_bounds(
    kinetic_names: tuple[str, ...],
    known_species: tuple[str, ...],
    k_bounds: tuple[float, float],
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """Build optimizer names and per-parameter bounds for k and scale factors."""
    scale_names = tuple(known_scale_parameter_name(label) for label in known_species)
    parameter_names = (*kinetic_names, *scale_names)
    parameter_bounds = {
        name: k_bounds
        for name in kinetic_names
    }
    for name in scale_names:
        parameter_bounds[name] = (1e-4, 1e4)
    return parameter_names, parameter_bounds



def fit_a_to_b_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
) -> FitResult:
    """Fit k by direct reconstruction with nonnegative spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 2:
        raise ValueError("The A -> B NNLS fit requires exactly 2 components.")

    k = optimize_k(
        lambda trial_k: direct_nonnegative_error(
            trial_k,
            experiment,
            c0,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
        ),
        k_bounds,
    )
    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    spectra = fit_nonnegative_spectra(
        experiment.absorbance,
        c,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
    )

    return FitResult(
        method="nnls",
        model="a_to_b",
        params={"k": k},
        species_labels=MODEL_SPECIES["a_to_b"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
    )



def fit_a_to_b_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    result_model: str = "a_to_b",
    concentration_profile=concentration_profile_a_to_b,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A -> B by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 2:
        raise ValueError("The A -> B direct spectral fit requires exactly 2 components.")

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k",),
        known_species,
        k_bounds,
    )

    def objective(params: dict[str, float]) -> float:
        c_trial = concentration_profile(experiment.t, params["k"], c0=c0)
        return direct_spectral_error_for_concentrations(
            c_trial,
            experiment,
            spectra_method=spectra_method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                result_model,
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="powell" if spectra_method == "nnls" else "hybrid",
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(params, result_model, known_species)
    k = params["k"]
    c = concentration_profile(experiment.t, k, c0=c0)
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=spectra_method,
        model=result_model,
        params={"k": k},
        species_labels=MODEL_SPECIES[result_model],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )



def fit_a_to_b(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A -> B with NNLS spectra."""
    if method != "nnls":
        raise ValueError("Only --fit-method nnls is supported")
    return fit_a_to_b_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
        fix_final_spectrum=fix_final_spectrum,
        progress_callback=progress_callback,
    )


def fit_mbfe3_sulfide_autocatalytic(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit global autocatalytic MbFeIII-SH reduction by sulfide."""
    if method != "nnls":
        raise ValueError("Only --fit-method nnls is supported")
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    if n_components != 2:
        raise ValueError("The MbFeIII sulfide autocatalytic fit requires exactly 2 components.")

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k_slow", "k_auto"),
        known_species,
        k_bounds,
    )

    def objective(params: dict[str, float]) -> float:
        c_trial = concentration_profile_mbfe3_sulfide_autocatalytic(
            experiment.t,
            params["k_slow"],
            params["k_auto"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c_trial,
            experiment,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                "mbfe3_sulfide_autocatalytic",
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="powell",
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(
        params,
        "mbfe3_sulfide_autocatalytic",
        known_species,
    )
    c = concentration_profile_mbfe3_sulfide_autocatalytic(
        experiment.t,
        params["k_slow"],
        params["k_auto"],
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=method,
        model="mbfe3_sulfide_autocatalytic",
        params={name: params[name] for name in ("k_slow", "k_auto")},
        species_labels=MODEL_SPECIES["mbfe3_sulfide_autocatalytic"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )


def fit_mbfe3_sulfide_binding_autocatalytic(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit MbFeIII binding HS- before autocatalytic sulfide reduction."""
    if method != "nnls":
        raise ValueError("Only --fit-method nnls is supported")
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    if n_components != 3:
        raise ValueError(
            "The MbFeIII sulfide binding/autocatalytic fit requires exactly 3 components."
        )

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k_on", "k_slow", "k_auto"),
        known_species,
        k_bounds,
    )

    def objective(params: dict[str, float]) -> float:
        c_trial = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
            experiment.t,
            params["k_on"],
            params["k_slow"],
            params["k_auto"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c_trial,
            experiment,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                "mbfe3_sulfide_binding_autocatalytic",
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        initial_parameters={
            "k_on": 7e-1,
            "k_slow": 5e-4,
            "k_auto": 2e-5,
        },
        progress_callback=progress_callback,
        optimizer="powell",
        # In log space Powell reliably explores all three kinetic dimensions
        # from the bounds' midpoint. The generic seven-start search repeats
        # essentially the same fit for this model at substantial cost.
        max_starts=1,
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(
        params,
        "mbfe3_sulfide_binding_autocatalytic",
        known_species,
    )
    c = concentration_profile_mbfe3_sulfide_binding_autocatalytic(
        experiment.t,
        params["k_on"],
        params["k_slow"],
        params["k_auto"],
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=method,
        model="mbfe3_sulfide_binding_autocatalytic",
        params={name: params[name] for name in ("k_on", "k_slow", "k_auto")},
        species_labels=MODEL_SPECIES["mbfe3_sulfide_binding_autocatalytic"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )


def fit_mbfe3_sulfide_hss_transsulfuration(
    experiment: Experiment,
    c0: float = 1.0,
    hss_ratio: float = 20.0,
    n_components: int = 2,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit MbFeIII-HS reduction with added HSS- transsulfuration."""
    if method != "nnls":
        raise ValueError("Only --fit-method nnls is supported")
    if hss_ratio < 0:
        raise ValueError("--hss-ratio must be nonnegative")
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    if n_components != 2:
        raise ValueError(
            "The MbFeIII sulfide/HSS transsulfuration fit requires exactly 2 components."
        )

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k_slow", "k_auto", "k_ts", "k_fast"),
        known_species,
        k_bounds,
    )
    parameter_bounds["k_ts"] = (1e-8, 1e4)

    def objective(params: dict[str, float]) -> float:
        c_trial = concentration_profile_mbfe3_sulfide_hss_transsulfuration(
            experiment.t,
            params["k_slow"],
            params["k_auto"],
            params["k_ts"],
            params["k_fast"],
            hss_ratio=hss_ratio,
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c_trial,
            experiment,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                "mbfe3_sulfide_hss_transsulfuration",
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="lbfgsb",
        max_starts=1,
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(
        params,
        "mbfe3_sulfide_hss_transsulfuration",
        known_species,
    )
    c = concentration_profile_mbfe3_sulfide_hss_transsulfuration(
        experiment.t,
        params["k_slow"],
        params["k_auto"],
        params["k_ts"],
        params["k_fast"],
        hss_ratio=hss_ratio,
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=method,
        model="mbfe3_sulfide_hss_transsulfuration",
        params={name: params[name] for name in ("k_slow", "k_auto", "k_ts", "k_fast")},
        species_labels=MODEL_SPECIES["mbfe3_sulfide_hss_transsulfuration"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )


def fit_mbfe3_sulfide_hss_transsulfuration_no_auto(
    experiment: Experiment,
    c0: float = 1.0,
    hss_ratio: float = 20.0,
    n_components: int = 2,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit HSS- transsulfuration with a slow path and no autocatalysis."""
    if method != "nnls":
        raise ValueError("Only --fit-method nnls is supported")
    if hss_ratio < 0:
        raise ValueError("--hss-ratio must be nonnegative")
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    if n_components != 2:
        raise ValueError(
            "The MbFeIII sulfide/HSS transsulfuration fit without "
            "autocatalysis requires exactly 2 components."
        )

    model = "mbfe3_sulfide_hss_transsulfuration_no_auto"
    kinetic_names = ("k_slow", "k_ts", "k_fast")
    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        kinetic_names,
        known_species,
        k_bounds,
    )
    parameter_bounds["k_ts"] = (1e-8, 1e4)

    def objective(params: dict[str, float]) -> float:
        c_trial = concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
            experiment.t,
            params["k_slow"],
            params["k_ts"],
            params["k_fast"],
            hss_ratio=hss_ratio,
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c_trial,
            experiment,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                model,
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="powell",
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(
        params,
        model,
        known_species,
    )
    c = concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
        experiment.t,
        params["k_slow"],
        params["k_ts"],
        params["k_fast"],
        hss_ratio=hss_ratio,
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
        fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
        fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=method,
        model=model,
        params={name: params[name] for name in kinetic_names},
        species_labels=MODEL_SPECIES[model],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )



def fit_a_to_b_to_c_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A -> B -> C by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A -> B -> C direct spectral fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_to_b_to_c(
            experiment.t,
            params["k1"],
            params["k2"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c,
            experiment,
            spectra_method=spectra_method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                "a_to_b_to_c",
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k1", "k2"),
        known_species,
        k_bounds,
    )
    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="powell" if spectra_method == "nnls" else "hybrid",
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(params, "a_to_b_to_c", known_species)
    c = concentration_profile_a_to_b_to_c(
        experiment.t,
        params["k1"],
        params["k2"],
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=spectra_method,
        model="a_to_b_to_c",
        params={name: params[name] for name in ("k1", "k2")},
        species_labels=MODEL_SPECIES["a_to_b_to_c"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )


def fit_a_to_b_to_c_mcr_als(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    kinetic_weight: float = 1.0,
    max_iter: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A -> B -> C with a soft kinetic MCR-ALS refinement.

    A conventional NNLS kinetic fit supplies the initial kinetic profile.  ALS
    then updates spectra and concentrations under nonnegativity and closure,
    with ``kinetic_weight`` controlling how strongly concentrations stay near
    that profile.  The returned kinetic parameters are a projection of the ALS
    concentrations onto the A -> B -> C model and should be treated as
    diagnostic rather than as a replacement for the hard kinetic fit.
    """
    if kinetic_weight < 0:
        raise ValueError("--mcr-kinetic-weight must be nonnegative")
    if max_iter <= 0:
        raise ValueError("--mcr-max-iter must be positive")
    if n_components != 3:
        raise ValueError("The A -> B -> C MCR-ALS fit requires exactly 3 components.")

    seed = fit_a_to_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
        fix_final_spectrum=fix_final_spectrum,
        progress_callback=progress_callback,
    )
    known_for_mcr = np.full_like(seed.spectra, np.nan)
    known_mask = np.zeros(n_components, dtype=bool)
    if known_spectra is not None:
        external_mask = np.all(np.isfinite(known_spectra), axis=0)
        known_for_mcr[:, external_mask] = seed.spectra[:, external_mask]
        known_mask |= external_mask
    if fix_initial_spectrum:
        known_for_mcr[:, 0] = seed.spectra[:, 0]
        known_mask[0] = True
    if fix_final_spectrum:
        known_for_mcr[:, -1] = seed.spectra[:, -1]
        known_mask[-1] = True

    spectra, concentrations, _mcr_error, _iterations = mcr_als_decompose(
        experiment.absorbance,
        n_components=n_components,
        initial_concentrations=seed.c,
        known_spectra=known_for_mcr if np.any(known_mask) else None,
        closure_total=c0,
        kinetic_reference=seed.c,
        kinetic_weight=kinetic_weight,
        max_iter=max_iter,
    )

    parameter_names = ("k1", "k2")
    parameter_bounds = {name: k_bounds for name in parameter_names}

    def concentration_projection_error(params: dict[str, float]) -> float:
        trial = concentration_profile_a_to_b_to_c(
            experiment.t,
            params["k1"],
            params["k2"],
            c0=c0,
        )
        return float(np.linalg.norm(concentrations - trial))

    projected = optimize_kinetic_parameters(
        concentration_projection_error,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        initial_parameters={"k1": seed.params["k1"], "k2": seed.params["k2"]},
        optimizer="powell",
        max_starts=1,
    )
    residuals = experiment.absorbance - spectra @ concentrations
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    known_scale_report = seed.known_spectrum_scales or {}
    return FitResult(
        method="mcr_als",
        model="a_to_b_to_c",
        params={name: projected[name] for name in parameter_names},
        species_labels=MODEL_SPECIES["a_to_b_to_c"],
        c=concentrations,
        spectra=spectra,
        absorbance_calc=spectra @ concentrations,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=float(np.linalg.norm(residuals)),
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )



def fit_a_to_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> FitResult:
    """Fit A -> B -> C by direct reconstruction with nonnegative spectra."""
    return fit_a_to_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def fit_a_rev_b_to_c_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A <-> B -> C by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A <-> B -> C direct spectral fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_rev_b_to_c(
            experiment.t,
            params["k1"],
            params["k_1"],
            params["k2"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c,
            experiment,
            spectra_method=spectra_method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_spectrum_scales=known_scale_parameters(
                params,
                "a_rev_b_to_c",
                known_species,
            ),
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        )

    parameter_names, parameter_bounds = direct_parameter_names_and_bounds(
        ("k1", "k_1", "k2"),
        known_species,
        k_bounds,
    )
    params = optimize_kinetic_parameters(
        objective,
        parameter_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        progress_callback=progress_callback,
        optimizer="powell" if spectra_method == "nnls" else "hybrid",
    )
    known_scale_report = extract_known_scale_report(params, known_species)
    known_scale_by_index = known_scale_parameters(params, "a_rev_b_to_c", known_species)
    c = concentration_profile_a_rev_b_to_c(
        experiment.t,
        params["k1"],
        params["k_1"],
        params["k2"],
        c0=c0,
    )
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_spectrum_scales=known_scale_by_index,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )

    return FitResult(
        method=spectra_method,
        model="a_rev_b_to_c",
        params={name: params[name] for name in ("k1", "k_1", "k2")},
        species_labels=MODEL_SPECIES["a_rev_b_to_c"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
        known_species=known_species,
        known_spectrum_scales=known_scale_report,
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )


def fit_a_rev_b_to_c_mcr_als(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    kinetic_weight: float = 1.0,
    max_iter: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit A <-> B -> C with a soft kinetic MCR-ALS refinement."""
    if kinetic_weight < 0:
        raise ValueError("--mcr-kinetic-weight must be nonnegative")
    if max_iter <= 0:
        raise ValueError("--mcr-max-iter must be positive")
    if n_components != 3:
        raise ValueError(
            "The A <-> B -> C MCR-ALS fit requires exactly 3 components."
        )

    seed = fit_a_rev_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
        fix_final_spectrum=fix_final_spectrum,
        progress_callback=progress_callback,
    )
    known_for_mcr = np.full_like(seed.spectra, np.nan)
    known_mask = np.zeros(n_components, dtype=bool)
    if known_spectra is not None:
        external_mask = np.all(np.isfinite(known_spectra), axis=0)
        known_for_mcr[:, external_mask] = seed.spectra[:, external_mask]
        known_mask |= external_mask
    if fix_initial_spectrum:
        known_for_mcr[:, 0] = seed.spectra[:, 0]
        known_mask[0] = True
    if fix_final_spectrum:
        known_for_mcr[:, -1] = seed.spectra[:, -1]
        known_mask[-1] = True

    spectra, concentrations, _mcr_error, _iterations = mcr_als_decompose(
        experiment.absorbance,
        n_components=n_components,
        initial_concentrations=seed.c,
        known_spectra=known_for_mcr if np.any(known_mask) else None,
        closure_total=c0,
        kinetic_reference=seed.c,
        kinetic_weight=kinetic_weight,
        max_iter=max_iter,
    )

    parameter_names = ("k1", "k_1", "k2")

    def concentration_projection_error(params: dict[str, float]) -> float:
        trial = concentration_profile_a_rev_b_to_c(
            experiment.t,
            params["k1"],
            params["k_1"],
            params["k2"],
            c0=c0,
        )
        return float(np.linalg.norm(concentrations - trial))

    projected = optimize_kinetic_parameters(
        concentration_projection_error,
        parameter_names,
        k_bounds,
        parameter_bounds={name: k_bounds for name in parameter_names},
        initial_parameters={name: seed.params[name] for name in parameter_names},
        optimizer="powell",
        max_starts=1,
    )
    residuals = experiment.absorbance - spectra @ concentrations
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)
    return FitResult(
        method="mcr_als",
        model="a_rev_b_to_c",
        params={name: projected[name] for name in parameter_names},
        species_labels=MODEL_SPECIES["a_rev_b_to_c"],
        c=concentrations,
        spectra=spectra,
        absorbance_calc=spectra @ concentrations,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=float(np.linalg.norm(residuals)),
        known_species=known_species,
        known_spectrum_scales=seed.known_spectrum_scales or {},
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
    )



def fit_a_rev_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
) -> FitResult:
    """Fit A <-> B -> C by direct reconstruction with nonnegative spectra."""
    return fit_a_rev_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
    )



def fit_model(
    model: str,
    experiment: Experiment,
    c0: float,
    n_components: int,
    method: str,
    k_bounds: tuple[float, float],
    hss_ratio: float = 20.0,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    mcr_kinetic_weight: float = 1.0,
    mcr_max_iter: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit the selected kinetic model."""
    if method not in {"nnls", "mcr_als"}:
        raise ValueError("Unknown fit method")
    if method == "mcr_als" and model not in {"a_to_b_to_c", "a_rev_b_to_c"}:
        raise ValueError(
            "MCR-ALS is currently supported only for A -> B -> C and A <-> B -> C"
        )
    if initial_spectrum_weight < 0:
        raise ValueError("--initial-spectrum-weight must be nonnegative")
    if fix_initial_spectrum and initial_spectrum_weight > 0:
        raise ValueError(
            "--fix-initial-spectrum cannot be combined with --initial-spectrum-weight"
        )

    if model == "a_to_b":
        return fit_a_to_b(
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "mbfe3_hss_no_binding":
        return fit_a_to_b_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            result_model=model,
            progress_callback=progress_callback,
        )
    if model == "mbfe3_sulfide_autocatalytic":
        return fit_mbfe3_sulfide_autocatalytic(
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "mbfe3_sulfide_binding_autocatalytic":
        return fit_mbfe3_sulfide_binding_autocatalytic(
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "mbfe3_sulfide_hss_transsulfuration":
        return fit_mbfe3_sulfide_hss_transsulfuration(
            experiment,
            c0=c0,
            hss_ratio=hss_ratio,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "mbfe3_sulfide_hss_transsulfuration_no_auto":
        return fit_mbfe3_sulfide_hss_transsulfuration_no_auto(
            experiment,
            c0=c0,
            hss_ratio=hss_ratio,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "a_rev_b_to_c":
        if method == "mcr_als":
            return fit_a_rev_b_to_c_mcr_als(
                experiment,
                c0=c0,
                n_components=n_components,
                k_bounds=k_bounds,
                initial_spectrum_weight=initial_spectrum_weight,
                known_spectra=known_spectra,
                known_species=known_species,
                fix_initial_spectrum=fix_initial_spectrum,
                fix_final_spectrum=fix_final_spectrum,
                kinetic_weight=mcr_kinetic_weight,
                max_iter=mcr_max_iter,
                progress_callback=progress_callback,
            )
        return fit_a_rev_b_to_c_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    if model == "a_to_b_to_c":
        if method == "mcr_als":
            return fit_a_to_b_to_c_mcr_als(
                experiment,
                c0=c0,
                n_components=n_components,
                k_bounds=k_bounds,
                initial_spectrum_weight=initial_spectrum_weight,
                known_spectra=known_spectra,
                known_species=known_species,
                fix_initial_spectrum=fix_initial_spectrum,
                fix_final_spectrum=fix_final_spectrum,
                kinetic_weight=mcr_kinetic_weight,
                max_iter=mcr_max_iter,
                progress_callback=progress_callback,
            )
        return fit_a_to_b_to_c_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unknown kinetic model: {model}")



def is_near_bound(value: float, bound: float) -> bool:
    """Return whether value is numerically close to a search bound."""
    # Match the log-space optimizer tolerance so a converged value just inside
    # the boundary still triggers bound reporting and automatic expansion.
    return np.isclose(np.log(value), np.log(bound), atol=5e-4, rtol=0.0)



def parameters_near_upper_bound(result: FitResult, k_max: float) -> list[str]:
    """Return fitted parameters that are effectively at the upper search bound."""
    return [
        name
        for name, value in result.params.items()
        if is_near_bound(value, k_max)
    ]



def fit_model_with_auto_k_max(
    model: str,
    experiment: Experiment,
    c0: float,
    n_components: int,
    method: str,
    k_bounds: tuple[float, float],
    auto_expand: bool,
    expand_factor: float,
    max_expand_steps: int,
    hss_ratio: float = 20.0,
    initial_spectrum_weight: float = 0.0,
    known_spectra: np.ndarray | None = None,
    known_species: tuple[str, ...] = (),
    fix_initial_spectrum: bool = False,
    fix_final_spectrum: bool = False,
    mcr_kinetic_weight: float = 1.0,
    mcr_max_iter: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> tuple[FitResult, tuple[float, float]]:
    """Fit a model, expanding the upper k bound when fitted constants hit it."""
    k_min, k_max = validate_k_bounds(k_bounds)
    if max_expand_steps < 0:
        raise ValueError("--k-max-expand-steps must be nonnegative")
    if auto_expand and expand_factor <= 1:
        raise ValueError("--k-max-expand-factor must be greater than 1")

    result = fit_model(
        model,
        experiment,
        c0=c0,
        n_components=n_components,
        method=method,
        k_bounds=(k_min, k_max),
        hss_ratio=hss_ratio,
        initial_spectrum_weight=initial_spectrum_weight,
        known_spectra=known_spectra,
        known_species=known_species,
        fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
        mcr_kinetic_weight=mcr_kinetic_weight,
        mcr_max_iter=mcr_max_iter,
        progress_callback=progress_callback,
    )

    if not auto_expand:
        return result, (k_min, k_max)

    for _ in range(max_expand_steps):
        bounded_names = parameters_near_upper_bound(result, k_max)
        if not bounded_names:
            break

        next_k_max = k_max * expand_factor
        if not np.isfinite(next_k_max):
            raise ValueError("Expanded k maximum is not finite")

        labels = ", ".join(PARAMETER_LABELS.get(name, name) for name in bounded_names)
        print(
            f"{labels} reached upper search bound ({k_max:g}); "
            f"expanding k max to {next_k_max:g} and refitting."
        )
        k_max = next_k_max
        result = fit_model(
            model,
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=(k_min, k_max),
            hss_ratio=hss_ratio,
            initial_spectrum_weight=initial_spectrum_weight,
            known_spectra=known_spectra,
            known_species=known_species,
            fix_initial_spectrum=fix_initial_spectrum,
            fix_final_spectrum=fix_final_spectrum,
            mcr_kinetic_weight=mcr_kinetic_weight,
            mcr_max_iter=mcr_max_iter,
            progress_callback=progress_callback,
        )

    return result, (k_min, k_max)
