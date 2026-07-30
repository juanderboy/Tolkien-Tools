#!/usr/bin/env python3
"""Estimate the start of MbFeIII-HS reduction after rapid sulfide binding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
