#!/usr/bin/env python3
"""Estimate the start of MbFeIII-HS reduction after rapid sulfide binding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from kinet_common import Experiment
from kinet_preprocessing import baseline_correct_region


@dataclass(frozen=True)
class SulfideNoBindingT0Estimate:
    """Diagnostic result for the sulfide reduction model without binding."""

    recommended_time: float
    recommended_index: int
    addition_time: float
    addition_index: int
    baseline_region: tuple[float, float]
    plateau_region: tuple[float, float]
    plateau_distance: np.ndarray
    corrected: Experiment


@dataclass(frozen=True)
class HssNoBindingT0Estimate:
    """Diagnostic result separating rapid HSS- binding from reduction."""

    recommended_time: float
    recommended_index: int
    addition_time: float
    addition_index: int
    binding_start_time: float
    binding_start_index: int
    binding_rate: float
    binding_end_time: float
    completion_time_constants: float
    fit_region: tuple[float, float]
    diagnostic_wavelength: float
    observed_trace: np.ndarray
    fitted_trace: np.ndarray
    corrected: Experiment
    baseline_region: tuple[float, float]


def _wavelength_mask(
    experiment: Experiment,
    lower: float,
    upper: float,
    label: str,
) -> np.ndarray:
    if experiment.wavelength.min() > lower or experiment.wavelength.max() < upper:
        raise ValueError(
            f"{label} requires coverage of the complete {lower:g}-{upper:g} nm region"
        )
    mask = (experiment.wavelength >= lower) & (experiment.wavelength <= upper)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"{label} requires measured wavelengths between {lower:g} and {upper:g} nm"
        )
    return mask


def estimate_sulfide_no_binding_t0(
    experiment: Experiment,
    baseline_region: tuple[float, float] = (750.0, 850.0),
    diagnostic_region: tuple[float, float] = (390.0, 650.0),
    plateau_delay: tuple[float, float] = (20.0, 60.0),
    plateau_tolerance: float = 0.005,
    addition_search_duration: float = 120.0,
) -> SulfideNoBindingT0Estimate:
    """Suggest the first spectrum representing the early MbFeIII-HS plateau.

    The largest early spectral step locates sulfide addition. The reference
    intermediate spectrum is the median spectrum acquired 20--60 s after that
    step. The recommended start is the first post-addition spectrum within
    0.5% Euclidean distance of that early plateau.
    """
    if experiment.t.size < 4:
        raise ValueError("At least four spectra are required to estimate t0")
    if plateau_tolerance <= 0:
        raise ValueError("Plateau tolerance must be positive")

    _wavelength_mask(
        experiment,
        baseline_region[0],
        baseline_region[1],
        "Baseline correction",
    )
    diagnostic_mask = _wavelength_mask(
        experiment,
        diagnostic_region[0],
        diagnostic_region[1],
        "t0 diagnostic",
    )
    corrected_absorbance = baseline_correct_region(
        experiment,
        baseline_region[0],
        baseline_region[1],
    )
    corrected = Experiment(
        t=experiment.t.copy(),
        wavelength=experiment.wavelength.copy(),
        absorbance=corrected_absorbance,
    )

    spectra = corrected_absorbance[diagnostic_mask, :].T
    steps = np.linalg.norm(np.diff(spectra, axis=0), axis=1)
    search_limit = experiment.t[0] + addition_search_duration
    searchable = np.where(experiment.t[1:] <= search_limit)[0]
    if searchable.size == 0:
        raise ValueError("No early spectra are available to locate sulfide addition")
    step_index = int(searchable[np.argmax(steps[searchable])])
    addition_index = step_index + 1
    addition_time = float(experiment.t[addition_index])

    plateau_start = addition_time + plateau_delay[0]
    plateau_end = addition_time + plateau_delay[1]
    plateau_mask = (experiment.t >= plateau_start) & (experiment.t <= plateau_end)
    if np.count_nonzero(plateau_mask) < 3:
        raise ValueError(
            "Not enough spectra 20-60 s after sulfide addition to estimate "
            "the MbFeIII-HS plateau"
        )
    plateau = np.median(spectra[plateau_mask], axis=0)
    plateau_norm = np.linalg.norm(plateau)
    if plateau_norm == 0:
        raise ValueError("The baseline-corrected plateau spectrum has zero norm")
    plateau_distance = np.linalg.norm(spectra - plateau, axis=1) / plateau_norm

    candidates = np.where(
        (np.arange(experiment.t.size) >= addition_index)
        & (experiment.t <= plateau_end)
        & (plateau_distance <= plateau_tolerance)
    )[0]
    if candidates.size:
        recommended_index = int(candidates[0])
    else:
        eligible = np.where(
            (np.arange(experiment.t.size) >= addition_index)
            & (experiment.t <= plateau_end)
        )[0]
        recommended_index = int(eligible[np.argmin(plateau_distance[eligible])])

    return SulfideNoBindingT0Estimate(
        recommended_time=float(experiment.t[recommended_index]),
        recommended_index=recommended_index,
        addition_time=addition_time,
        addition_index=addition_index,
        baseline_region=baseline_region,
        plateau_region=(plateau_start, plateau_end),
        plateau_distance=plateau_distance,
        corrected=corrected,
    )


def estimate_hss_no_binding_t0(
    experiment: Experiment,
    baseline_region: tuple[float, float] = (750.0, 850.0),
    diagnostic_region: tuple[float, float] = (390.0, 650.0),
    diagnostic_wavelength: float = 409.0,
    binding_fit_duration: float = 15.0,
    completion_time_constants: float = 4.0,
    addition_search_duration: float = 120.0,
) -> HssNoBindingT0Estimate:
    """Suggest t0 when the rapid MbFeIII + HSS- binding phase has ended.

    The early spectral step locates reagent addition. Starting from the local
    maximum of the 409-nm MbFeIII trace, the rapid decay is fitted as an
    exponential plus a linear term for the slower reduction that is already
    occurring. The suggested start is the first measured spectrum at or after
    four fitted time constants (about 98% completion of the rapid phase).
    """
    if experiment.t.size < 8:
        raise ValueError("At least eight spectra are required to estimate HSS- t0")
    if binding_fit_duration <= 0:
        raise ValueError("Binding fit duration must be positive")
    if completion_time_constants <= 0:
        raise ValueError("Completion time constants must be positive")

    _wavelength_mask(
        experiment,
        baseline_region[0],
        baseline_region[1],
        "Baseline correction",
    )
    diagnostic_mask = _wavelength_mask(
        experiment,
        diagnostic_region[0],
        diagnostic_region[1],
        "t0 diagnostic",
    )
    if not (
        experiment.wavelength.min()
        <= diagnostic_wavelength
        <= experiment.wavelength.max()
    ):
        raise ValueError(
            f"HSS- t0 diagnostic requires coverage of {diagnostic_wavelength:g} nm"
        )

    corrected_absorbance = baseline_correct_region(
        experiment,
        baseline_region[0],
        baseline_region[1],
    )
    corrected = Experiment(
        t=experiment.t.copy(),
        wavelength=experiment.wavelength.copy(),
        absorbance=corrected_absorbance,
    )

    spectra = corrected_absorbance[diagnostic_mask, :].T
    steps = np.linalg.norm(np.diff(spectra, axis=0), axis=1)
    search_limit = experiment.t[0] + addition_search_duration
    searchable = np.where(experiment.t[1:] <= search_limit)[0]
    if searchable.size == 0:
        raise ValueError("No early spectra are available to locate HSS- addition")
    step_index = int(searchable[np.argmax(steps[searchable])])
    addition_index = step_index + 1
    addition_time = float(experiment.t[addition_index])

    wavelength_index = int(
        np.argmin(np.abs(experiment.wavelength - diagnostic_wavelength))
    )
    observed_trace = corrected_absorbance[wavelength_index]
    local_start = max(0, addition_index - 2)
    local_end = min(experiment.t.size, addition_index + 4)
    binding_start_index = int(
        local_start + np.argmax(observed_trace[local_start:local_end])
    )
    binding_start_time = float(experiment.t[binding_start_index])

    fit_mask = (
        (np.arange(experiment.t.size) >= binding_start_index)
        & (experiment.t <= binding_start_time + binding_fit_duration)
    )
    if np.count_nonzero(fit_mask) < 6:
        raise ValueError(
            "Not enough spectra after HSS- addition to fit the rapid binding decay"
        )
    fit_times = experiment.t[fit_mask] - binding_start_time
    fit_values = observed_trace[fit_mask]
    trace_scale = max(float(np.ptp(fit_values)), 1e-6)

    def residual(parameters: np.ndarray) -> np.ndarray:
        offset, amplitude, rate, slope = parameters
        calculated = (
            offset
            + amplitude * np.exp(-rate * fit_times)
            + slope * fit_times
        )
        return (calculated - fit_values) / trace_scale

    initial_offset = float(fit_values[-1])
    initial_amplitude = max(float(fit_values[0] - initial_offset), trace_scale)
    fit = least_squares(
        residual,
        (initial_offset, initial_amplitude, 0.5, 0.0),
        bounds=(
            (-np.inf, 0.0, 1e-3, -np.inf),
            (np.inf, np.inf, 10.0, np.inf),
        ),
    )
    if not fit.success:
        raise ValueError(f"Rapid HSS- binding fit failed: {fit.message}")

    offset, amplitude, binding_rate, slope = fit.x
    if amplitude < 0.05 * trace_scale:
        raise ValueError("No resolvable rapid decay was found in the 409-nm trace")
    fitted_trace = (
        offset
        + amplitude * np.exp(-binding_rate * (experiment.t - binding_start_time))
        + slope * (experiment.t - binding_start_time)
    )
    binding_end_time = float(
        binding_start_time + completion_time_constants / binding_rate
    )
    eligible = np.where(
        (np.arange(experiment.t.size) >= binding_start_index)
        & (experiment.t >= binding_end_time)
    )[0]
    if eligible.size == 0:
        raise ValueError("The estimated end of HSS- binding is outside the experiment")
    recommended_index = int(eligible[0])

    return HssNoBindingT0Estimate(
        recommended_time=float(experiment.t[recommended_index]),
        recommended_index=recommended_index,
        addition_time=addition_time,
        addition_index=addition_index,
        binding_start_time=binding_start_time,
        binding_start_index=binding_start_index,
        binding_rate=float(binding_rate),
        binding_end_time=binding_end_time,
        completion_time_constants=completion_time_constants,
        fit_region=(binding_start_time, float(experiment.t[fit_mask][-1])),
        diagnostic_wavelength=float(experiment.wavelength[wavelength_index]),
        observed_trace=observed_trace,
        fitted_trace=fitted_trace,
        corrected=corrected,
        baseline_region=baseline_region,
    )
