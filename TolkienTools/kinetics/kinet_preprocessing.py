#!/usr/bin/env python3
"""Preprocessing and interactive preprocessing choices for TolKinet."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import hashlib

import numpy as np

from kinet_common import Experiment
from kinet_known_spectra import FixedSpectrumFile, read_fixed_spectrum_file


def baseline_correct(absorbance: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Subtract the mean of the last n_points from each spectrum.

    Each column is one spectrum at one time. The correction is therefore
    column-wise, matching the original MATLAB loop.
    """
    baseline = absorbance[-n_points:, :].mean(axis=0, keepdims=True)
    return absorbance - baseline



def baseline_correct_region(experiment: Experiment, wavelength_min: float, wavelength_max: float) -> np.ndarray:
    """Subtract the mean absorbance in a selected wavelength interval."""
    mask = (experiment.wavelength >= wavelength_min) & (
        experiment.wavelength <= wavelength_max
    )
    if not np.any(mask):
        raise ValueError(
            f"Baseline region {wavelength_min:g}-{wavelength_max:g} has no data points"
        )
    baseline = experiment.absorbance[mask, :].mean(axis=0, keepdims=True)
    return experiment.absorbance - baseline



def suggest_auto_baseline_region(
    experiment: Experiment,
    window_width: float = 40.0,
    min_points: int = 5,
) -> tuple[float, float]:
    """Suggest a flat, low-activity wavelength interval for baseline correction."""
    wavelength = experiment.wavelength
    candidates: list[tuple[float, float, float, float, float]] = []

    for start_index, start in enumerate(wavelength):
        end = start + window_width
        mask = (wavelength >= start) & (wavelength <= end)
        if np.count_nonzero(mask) < min_points:
            continue

        sub = experiment.absorbance[mask, :]
        temporal_activity = float(np.mean(np.std(sub, axis=1)))
        mean_abs_signal = float(np.mean(np.abs(sub)))
        spectral_slope = float(np.mean(np.abs(np.diff(sub, axis=0)))) if sub.shape[0] > 1 else 0.0
        candidates.append((temporal_activity, mean_abs_signal, spectral_slope, start, wavelength[mask][-1]))

    if not candidates:
        raise ValueError("Could not find an automatic baseline region")

    metrics = np.array([[row[0], row[1], row[2]] for row in candidates])
    scale = np.median(metrics, axis=0)
    scale[scale == 0] = 1.0
    normalized = metrics / scale
    scores = 0.6 * normalized[:, 0] + 0.25 * normalized[:, 1] + 0.15 * normalized[:, 2]
    best_index = int(np.argmin(scores))
    return candidates[best_index][3], candidates[best_index][4]



def drop_spectra(experiment: Experiment, indices: list[int]) -> Experiment:
    """Remove selected spectra by zero-based column indices."""
    if not indices:
        return experiment

    mask = np.ones(experiment.t.size, dtype=bool)
    mask[indices] = False
    return Experiment(
        t=experiment.t[mask],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, mask],
    )


def zero_time_at_first_spectrum(experiment: Experiment) -> Experiment:
    """Shift times so the first retained spectrum is corrected time zero."""
    if experiment.t.size == 0:
        return experiment
    return Experiment(
        t=experiment.t - experiment.t[0],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance,
    )



def start_reaction_at_time(
    experiment: Experiment,
    reaction_start_time: float | None,
) -> Experiment:
    """Keep spectra with t >= reaction_start_time."""
    if reaction_start_time is None:
        return experiment
    if not np.isfinite(reaction_start_time):
        raise ValueError("Reaction start time must be finite")

    mask = experiment.t >= reaction_start_time
    if not np.any(mask):
        raise ValueError(
            f"Reaction start time {reaction_start_time:g} leaves no spectra with t >= t_start"
        )
    return Experiment(
        t=experiment.t[mask],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, mask],
    )


def end_reaction_at_time(
    experiment: Experiment,
    reaction_end_time: float | None,
) -> Experiment:
    """Keep spectra up to an optional end time in the original time units."""
    if reaction_end_time is None:
        return experiment
    if not np.isfinite(reaction_end_time):
        raise ValueError("Reaction end time must be finite")

    mask = experiment.t <= reaction_end_time
    if not np.any(mask):
        raise ValueError(
            f"Reaction end time {reaction_end_time:g} leaves no spectra with t <= t_end"
        )

    return Experiment(
        t=experiment.t[mask],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, mask],
    )



def parse_spectrum_selection(selection: str, n_spectra: int) -> list[int]:
    """Parse 1-based spectrum indices such as '1,3-5'."""
    if not selection.strip():
        return []

    indices: set[int] = set()
    for part in selection.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = [int(part)]

        for value in values:
            if value < 1 or value > n_spectra:
                raise ValueError(
                    f"Spectrum index {value} is outside the valid range 1-{n_spectra}"
                )
            indices.add(value - 1)

    return sorted(indices)



def parse_wavelength_range(text: str) -> tuple[float, float]:
    """Parse wavelength ranges like '800-820', '800 820' or '800,820'."""
    normalized = text.strip().replace(",", " ")
    if "-" in normalized:
        parts = [part.strip() for part in normalized.split("-", maxsplit=1)]
    else:
        parts = normalized.split()

    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Expected wavelength range like 800-820 or 800 820")

    lower, upper = sorted((float(parts[0]), float(parts[1])))
    return lower, upper



def parse_baseline_choice(
    text: str,
    auto_region: tuple[float, float],
    default_points: int,
) -> tuple[str, tuple[float, float] | None, int]:
    """Parse the interactive baseline correction choice."""
    choice = text.strip().lower()
    if not choice or choice in {"auto", "automatic", "a", "1"}:
        return "region", auto_region, default_points

    if choice in {"manual", "m", "2"}:
        region_text = input("Manual baseline region in nm, e.g. 760-820: ").strip()
        return "region", parse_wavelength_range(region_text), default_points

    if choice in {"none", "no", "n", "4"}:
        return "none", None, default_points

    if choice in {"points", "point", "p", "3"}:
        return "points", None, default_points

    point_prefixes = ("points ", "point ", "p ")
    if choice.startswith(point_prefixes):
        _, value_text = choice.split(maxsplit=1)
        n_points = int(value_text)
        if n_points <= 0:
            raise ValueError("Number of baseline points must be positive")
        return "points", None, n_points

    # A bare range such as 760-820 is treated as a manual baseline region.
    return "region", parse_wavelength_range(text), default_points



def parse_reaction_start_time_choice(
    text: str,
    default_reaction_start_time: float | None,
) -> float | None:
    """Parse the interactive reaction-start time choice."""
    choice = text.strip().lower()
    if not choice:
        return default_reaction_start_time
    if choice in {"none", "no", "n"}:
        return None

    reaction_start_time = float(choice)
    if not np.isfinite(reaction_start_time):
        raise ValueError("Reaction start time must be finite")
    return reaction_start_time


def parse_reaction_end_time_choice(
    text: str,
    default_reaction_end_time: float | None,
) -> float | None:
    """Parse the interactive reaction-end time choice."""
    choice = text.strip().lower()
    if not choice:
        return default_reaction_end_time
    if choice in {"none", "no", "n"}:
        return None

    reaction_end_time = float(choice)
    if not np.isfinite(reaction_end_time):
        raise ValueError("Reaction end time must be finite")
    return reaction_end_time



def crop_wavelengths(
    experiment: Experiment,
    wavelength_min: float,
    wavelength_max: float,
) -> Experiment:
    """Keep only wavelengths inside the selected interval."""
    mask = (experiment.wavelength >= wavelength_min) & (
        experiment.wavelength <= wavelength_max
    )
    if not np.any(mask):
        raise ValueError(
            f"Wavelength range {wavelength_min:g}-{wavelength_max:g} has no data points"
        )
    return Experiment(
        t=experiment.t,
        wavelength=experiment.wavelength[mask],
        absorbance=experiment.absorbance[mask, :],
    )



def corrected_absorbance_from_baseline_choice(
    experiment: Experiment,
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    baseline_window: float,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Apply the selected baseline correction and return the region used."""
    if baseline_mode == "none":
        return experiment.absorbance.copy(), baseline_region
    if baseline_mode == "points":
        return baseline_correct(experiment.absorbance, baseline_points), baseline_region
    if baseline_mode == "auto-flat":
        baseline_region = suggest_auto_baseline_region(
            experiment,
            window_width=baseline_window,
        )
        print(
            "Automatic baseline region: "
            f"{baseline_region[0]:g}-{baseline_region[1]:g} nm"
        )
        return (
            baseline_correct_region(
                experiment,
                baseline_region[0],
                baseline_region[1],
            ),
            baseline_region,
        )
    if baseline_mode == "region":
        if baseline_region is None:
            raise ValueError("Baseline mode 'region' requires a baseline region")
        return (
            baseline_correct_region(
                experiment,
                baseline_region[0],
                baseline_region[1],
            ),
            baseline_region,
        )
    raise ValueError(f"Unknown baseline mode: {baseline_mode}")



def apply_preprocessing_choices(
    args: argparse.Namespace,
    experiment: Experiment,
    drop_indices: list[int],
    reaction_start_time: float | None,
    reaction_end_time: float | None,
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    work_range: tuple[float, float],
    initial_spectrum_index: int | None = None,
) -> tuple[Experiment, Experiment, tuple[float, float] | None]:
    """Apply spectrum/time pruning, baseline correction and wavelength crop."""
    pruned = drop_spectra(experiment, drop_indices)
    pruned = end_reaction_at_time(pruned, reaction_end_time)
    pruned = start_reaction_at_time(pruned, reaction_start_time)
    if reaction_start_time is not None:
        # The entered t0 is exact, even between measured frames or after drops.
        origin = float(reaction_start_time)
    elif initial_spectrum_index is not None:
        # Keep the legacy CLI reference origin only when no explicit t0 exists.
        if initial_spectrum_index < 0 or initial_spectrum_index >= experiment.t.size:
            raise ValueError("Initial reference spectrum index is outside the experiment")
        origin = float(experiment.t[initial_spectrum_index])
    else:
        origin = float(pruned.t[0])
    args.reaction_time_origin = origin
    pruned = Experiment(t=pruned.t - origin, wavelength=pruned.wavelength,
                        absorbance=pruned.absorbance)

    if pruned.t.size < 2:
        raise ValueError("Time selection must leave at least two spectra")
    corrected_absorbance, used_region = corrected_absorbance_from_baseline_choice(
        pruned,
        baseline_mode,
        baseline_region,
        baseline_points,
        args.baseline_window,
    )
    corrected = Experiment(
        t=pruned.t,
        wavelength=pruned.wavelength,
        absorbance=corrected_absorbance,
    )
    cropped = crop_wavelengths(corrected, work_range[0], work_range[1])
    return corrected, cropped, used_region


def prepare_initial_reference_spectrum(
    experiment: Experiment,
    spectrum_index: int | tuple[int, ...] | FixedSpectrumFile | None,
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    baseline_window: float,
    work_range: tuple[float, float],
    used_region: tuple[float, float] | None,
    c0: float = 1.0,
) -> np.ndarray | None:
    """Correct selected original spectra, then average without changing fit times."""
    if spectrum_index is None:
        return None
    if isinstance(spectrum_index, FixedSpectrumFile):
        wavelength, values = read_fixed_spectrum_file(spectrum_index)
        target = experiment.wavelength[(experiment.wavelength >= work_range[0]) &
                                      (experiment.wavelength <= work_range[1])]
        if not len(target) or target[0] < wavelength[0] or target[-1] > wavelength[-1]:
            raise ValueError(f"{spectrum_index.path}: reference does not cover the fitted wavelengths")
        # Imported values are already baseline-corrected molar spectra; do not
        # apply the experimental baseline or normalize them a second time.
        interpolated = np.interp(target, wavelength, values)
        return interpolated * c0
    indices = [spectrum_index] if isinstance(spectrum_index, int) else list(spectrum_index)
    if not indices or min(indices) < 0 or max(indices) >= experiment.t.size:
        raise ValueError("Reference spectrum indices are outside the experiment")
    one = Experiment(
        t=experiment.t[indices],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, indices],
    )
    effective_mode = "region" if used_region is not None else baseline_mode
    corrected, _ = corrected_absorbance_from_baseline_choice(
        one,
        effective_mode,
        used_region if used_region is not None else baseline_region,
        baseline_points,
        baseline_window,
    )
    corrected_one = Experiment(t=one.t, wavelength=one.wavelength, absorbance=corrected)
    return crop_wavelengths(corrected_one, work_range[0], work_range[1]).absorbance.mean(axis=1)



def parse_endpoint_reference(text, n_spectra, dropped_indices):
    """Return original zero-based indices for one spectrum or an inclusive range."""
    if not re.fullmatch(r"\d+(?:\s*-\s*\d+)?", text.strip()):
        raise ValueError("Use yes, no, an original index (23), or an inclusive range (23-25)")
    parts = [int(x.strip()) for x in text.split('-')]
    first, last = parts[0], parts[-1]
    if first < 1 or last > n_spectra or first > last:
        raise ValueError(f"Reference indices must be increasing and within 1-{n_spectra}")
    indices = tuple(range(first-1, last))
    if set(indices).intersection(dropped_indices):
        raise ValueError("A reference selection contains a dropped spectrum")
    return indices


def ask_endpoint_reference(label, endpoint, original_time, n_spectra, dropped_indices,
                           default_fixed=False, default_indices=None, alias=None):
    """Read a common initial/product reference choice."""
    if isinstance(default_indices, FixedSpectrumFile):
        default = str(default_indices.path)
    elif isinstance(default_indices, int):
        default = str(default_indices + 1)
    elif default_indices:
        first, last = default_indices[0]+1, default_indices[-1]+1
        default = str(first) if first == last else f"{first}-{last}"
    else:
        default = 'yes' if default_fixed else 'no'
    endpoint_help = (f"yes = {endpoint} retained spectrum (original t = {original_time:g}); "
                     if endpoint in {"first", "last"} else "")
    print(f"{label}: {endpoint_help}no = fit freely; index/range = original "
          "spectrum/mean; file.dat = molar spectrum (amplitude choice follows).")
    text = input(f"Fixed spectrum for {label}? Enter = {default}: ").strip() or default
    choice = text.lower()
    if choice in {"s", "si", "sí", "y", "yes"}:
        if endpoint not in {"first", "last"}:
            raise ValueError("For an intermediate, specify an index, range or spectrum file")
        return True, None
    if choice in {"n", "no"}:
        return False, None
    if re.fullmatch(r"\d+(?:\s*-\s*\d+)?", text):
        return False, parse_endpoint_reference(text, n_spectra, dropped_indices)
    path = Path(text.strip('"').strip("'")).expanduser()
    if not path.is_file():
        raise ValueError(f"Reference file not found: {path}")
    reference = FixedSpectrumFile(path.resolve(), alias)
    read_fixed_spectrum_file(reference)
    previous = (default_indices if isinstance(default_indices, FixedSpectrumFile)
                and default_indices.path.resolve() == reference.path else None)
    print(f"Reference file: {reference.path}")
    print("Expected values: molar absorptivity epsilon, in M^-1 cm^-1, "
          "already baseline-corrected (1 cm optical path).")
    default_adjust = (isinstance(default_indices, FixedSpectrumFile)
                      and default_indices.path.resolve() == reference.path
                      and default_indices.fit_amplitude)
    default_text = "yes" if default_adjust else "no"
    while True:
        amplitude = input(
            f"Adjust the amplitude of {label} during fitting? "
            f"yes = fit a multiplier from 0.7 to 1.3; no = keep multiplier 1. "
            f"Enter = {default_text}: ").strip().lower() or default_text
        if amplitude in {"s", "si", "sí", "y", "yes", "n", "no"}:
            break
        print("Amplitude choice must be yes or no.")
    fit_amplitude = amplitude in {"s", "si", "sí", "y", "yes"}
    return False, FixedSpectrumFile(reference.path, alias, fit_amplitude)


def ask_preprocessing_choices(
    experiment: Experiment,
    baseline_points: int,
    default_work_range: tuple[float, float],
    default_c0: float,
    baseline_window: float,
    default_reaction_start_time: float | None,
    default_reaction_end_time: float | None,
    allowed_work_range: tuple[float, float] | None = None,
    initial_spectrum_label: str | None = None,
    default_fix_initial_spectrum: bool = False,
    initial_spectrum_unavailable_reason: str | None = None,
    default_initial_spectrum_index: int | tuple[int, ...] | FixedSpectrumFile | None = None,
    final_spectrum_label: str | None = None,
    default_fix_final_spectrum: bool = False,
    final_spectrum_unavailable_reason: str | None = None,
    overview_auto_region: tuple[float, float] | None = None,
    preferred_baseline_region: tuple[float, float] | None = None,
    default_final_spectrum_indices: tuple[int, ...] | None = None,
    species_labels: tuple[str, ...] | None = None,
    default_intermediate_references: dict | None = None,
    supplied_species: set[str] | None = None,
) -> tuple[
    list[int],
    float | None,
    float | None,
    bool,
    int | tuple[int, ...] | FixedSpectrumFile | None,
    bool,
    str,
    tuple[float, float] | None,
    int,
    tuple[float, float],
    float,
    tuple[int, ...] | FixedSpectrumFile | None,
    dict,
]:
    """Ask which spectra/baseline region/wavelength range to use."""
    print()
    print(f"The experiment contains {experiment.t.size} spectra.")
    print("Spectrum indices are 1-based: the first spectrum is 1.")
    drop_text = input(
        "Spectra to remove? E.g. 1 or 1,4-6. Enter = none: "
    )
    drop_indices = parse_spectrum_selection(drop_text, experiment.t.size)

    dropped = drop_spectra(experiment, drop_indices)
    if default_reaction_start_time is None:
        start_default_text = "first retained spectrum (no cutoff)"
    else:
        start_default_text = f"{default_reaction_start_time:g}"
    start_text = input(
        "Reaction time zero t0 (original time units)? "
        "Keep spectra with t_original >= t0; use exactly t = t_original - t0, "
        "even if no spectrum was measured at t0. "
        f"Enter = {start_default_text}: "
    )
    reaction_start_time = parse_reaction_start_time_choice(
        start_text,
        default_reaction_start_time,
    )
    if default_reaction_end_time is None:
        end_default_text = "no cutoff"
    else:
        end_default_text = f"{default_reaction_end_time:g}"
    end_text = input(
        "Final time for fitting? "
        "Use this option to isolate an initial phase. "
        f"Only use spectra with t <= t_end. Enter = {end_default_text}: "
    )
    reaction_end_time = parse_reaction_end_time_choice(
        end_text,
        default_reaction_end_time,
    )
    dropped = end_reaction_at_time(dropped, reaction_end_time)
    dropped = start_reaction_at_time(dropped, reaction_start_time)
    first_kept_original_time = float(dropped.t[0]) if dropped.t.size else np.nan
    dropped = zero_time_at_first_spectrum(dropped)
    if dropped.t.size < 2:
        raise ValueError("Time selection must leave at least two spectra")

    fix_initial_spectrum = default_fix_initial_spectrum
    initial_spectrum_index = default_initial_spectrum_index
    fix_final_spectrum = default_fix_final_spectrum
    final_spectrum_indices = default_final_spectrum_indices
    if initial_spectrum_label is not None:
        if initial_spectrum_unavailable_reason is not None:
            print(initial_spectrum_unavailable_reason)
            fix_initial_spectrum, initial_spectrum_index = False, None
        else:
            fix_initial_spectrum, initial_spectrum_index = ask_endpoint_reference(
                initial_spectrum_label, "first", first_kept_original_time,
                experiment.t.size, drop_indices, default_fix_initial_spectrum,
                default_initial_spectrum_index, alias="A")
    intermediate_references = {}
    for column, label in enumerate(species_labels or ()):
        if column in {0, len(species_labels)-1} or label in (supplied_species or set()):
            continue
        _, selection = ask_endpoint_reference(label, "intermediate", 0,
            experiment.t.size, drop_indices,
            default_indices=(default_intermediate_references or {}).get(column),
            alias="ABC"[column])
        if selection is not None:
            intermediate_references[column] = selection
    if final_spectrum_label is not None:
        if final_spectrum_unavailable_reason is not None:
            print(final_spectrum_unavailable_reason)
            fix_final_spectrum, final_spectrum_indices = False, None
        else:
            last_time = float(dropped.t[-1] + first_kept_original_time)
            fix_final_spectrum, final_spectrum_indices = ask_endpoint_reference(
                final_spectrum_label, "last", last_time, experiment.t.size,
                drop_indices, default_fix_final_spectrum, default_final_spectrum_indices,
                alias="ABC"[len(species_labels)-1] if species_labels else "B")

    selections = [initial_spectrum_index, final_spectrum_indices, *intermediate_references.values()]
    for selection in selections:
        if isinstance(selection, FixedSpectrumFile):
            wavelength, _ = read_fixed_spectrum_file(selection)
            lo = max(float(experiment.wavelength[0]), float(wavelength[0]))
            hi = min(float(experiment.wavelength[-1]), float(wavelength[-1]))
            if allowed_work_range is not None:
                lo, hi = max(lo, allowed_work_range[0]), min(hi, allowed_work_range[1])
            if lo >= hi:
                raise ValueError("Reference files and experiment have no common wavelength range")
            allowed_work_range = (lo, hi)
    if allowed_work_range is not None:
        default_work_range = (max(default_work_range[0], allowed_work_range[0]),
                              min(default_work_range[1], allowed_work_range[1]))
        if default_work_range[0] >= default_work_range[1]:
            default_work_range = allowed_work_range

    if preferred_baseline_region is None:
        auto_region = suggest_auto_baseline_region(
            dropped,
            window_width=baseline_window,
        )
    else:
        auto_region = preferred_baseline_region
    print()
    if preferred_baseline_region is not None:
        print(
            "Model-specific baseline suggestion: "
            f"{auto_region[0]:g}-{auto_region[1]:g} nm"
        )
    elif overview_auto_region is not None and not np.allclose(
        auto_region,
        overview_auto_region,
    ):
        print(
            "The automatic region changed because it now uses only "
            "the spectra retained after time trimming:"
        )
        print(
            "  Initial panel (all spectra): "
            f"{overview_auto_region[0]:g}-{overview_auto_region[1]:g} nm"
        )
        print(
            "  Current suggestion (retained spectra): "
            f"{auto_region[0]:g}-{auto_region[1]:g} nm"
        )
    else:
        print(
            "Automatic baseline suggestion: "
            f"{auto_region[0]:g}-{auto_region[1]:g} nm"
        )
    print("Baseline correction options:")
    print(f"  1. Accept automatic region {auto_region[0]:g}-{auto_region[1]:g} nm")
    print("  2. Type a manual wavelength range, e.g. 760-820")
    print(f"  3. Use the last wavelength points, e.g. points 30 (default = {baseline_points})")
    print("  4. Do not correct baseline")
    baseline_text = input(
        "Choice [Enter/1 automatic, range manual, points N, 4 none]: "
    )
    baseline_mode, baseline_region, baseline_points = parse_baseline_choice(
        baseline_text,
        auto_region,
        baseline_points,
    )

    while True:
        if allowed_work_range is None:
            range_prompt = (
                "Wavelength range for fitting? E.g. 320-820. "
                f"Enter = {default_work_range[0]:g}-{default_work_range[1]:g}: "
            )
        else:
            range_prompt = (
                "Wavelength range for fitting? "
                f"Must be within {allowed_work_range[0]:g}-"
                f"{allowed_work_range[1]:g} nm. "
                f"Enter = {default_work_range[0]:g}-{default_work_range[1]:g}: "
            )
        work_text = input(
            range_prompt
        ).strip()
        if not work_text:
            work_range = default_work_range
        else:
            work_range = parse_wavelength_range(work_text)
        if allowed_work_range is None:
            break
        if (
            work_range[0] >= allowed_work_range[0]
            and work_range[1] <= allowed_work_range[1]
        ):
            break
        print(
            "Invalid range: known spectra restrict the range to "
            f"{allowed_work_range[0]:g}-{allowed_work_range[1]:g} nm."
        )

    c0_text = input(
        f"Initial concentration c0 in molar units? Enter = {default_c0:g} M: "
    ).strip()
    if not c0_text:
        c0 = default_c0
    else:
        c0 = float(c0_text)
        if c0 <= 0:
            raise ValueError("c0 must be positive")

    return (
        drop_indices,
        reaction_start_time,
        reaction_end_time,
        fix_initial_spectrum,
        initial_spectrum_index,
        fix_final_spectrum,
        baseline_mode,
        baseline_region,
        baseline_points,
        work_range,
        c0,
        final_spectrum_indices,
        intermediate_references,
    )



def preprocess_experiment(
    args: argparse.Namespace,
    experiment: Experiment,
    allowed_work_range: tuple[float, float] | None = None,
    initial_spectrum_label: str | None = None,
    default_fix_initial_spectrum: bool = False,
    initial_spectrum_unavailable_reason: str | None = None,
    final_spectrum_label: str | None = None,
    default_fix_final_spectrum: bool = False,
    final_spectrum_unavailable_reason: str | None = None,
    overview_auto_region: tuple[float, float] | None = None,
    preferred_baseline_region: tuple[float, float] | None = None,
    species_labels: tuple[str, ...] | None = None,
    supplied_species: set[str] | None = None,
) -> tuple[
    Experiment,
    tuple[float, float],
    float,
    float | None,
    float | None,
    bool,
    bool,
    int | tuple[int, ...] | FixedSpectrumFile | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Apply optional spectrum removal and baseline correction."""
    drop_indices = parse_spectrum_selection(args.drop_spectra, experiment.t.size)
    work_range = tuple(sorted((args.lambda_min, args.lambda_max)))
    if allowed_work_range is not None:
        work_range = (
            max(work_range[0], allowed_work_range[0]),
            min(work_range[1], allowed_work_range[1]),
        )
        if work_range[0] >= work_range[1]:
            raise ValueError(
                "Requested wavelength range does not overlap the range allowed "
                "by the experiment and known spectra"
            )
    c0 = args.c0
    baseline_mode = args.baseline_mode
    baseline_points = args.baseline_points
    reaction_start_time = args.reaction_start_time
    reaction_end_time = args.reaction_end_time
    fix_initial_spectrum = default_fix_initial_spectrum
    fix_final_spectrum = default_fix_final_spectrum
    final_spectrum_indices = None
    intermediate_references = {}
    args.additional_fixed_references = {}
    args.fixed_reference_files = []
    args.intermediate_reference_indices = {}
    initial_spectrum_index = getattr(args, "initial_spectrum_index", None)
    if initial_spectrum_index is not None:
        initial_spectrum_index -= 1
        if initial_spectrum_index < 0 or initial_spectrum_index >= experiment.t.size:
            raise ValueError("--initial-spectrum-index is outside the experiment")
        if initial_spectrum_index in drop_indices:
            raise ValueError("--initial-spectrum-index refers to a dropped spectrum")
        if initial_spectrum_label is None:
            raise ValueError(
                "An initial spectrum index requires a model with initial-spectrum selection"
            )
        fix_initial_spectrum = False

    baseline_region = None
    if args.baseline_lambda_min is not None or args.baseline_lambda_max is not None:
        if args.baseline_lambda_min is None or args.baseline_lambda_max is None:
            raise ValueError(
                "Use both --baseline-lambda-min and --baseline-lambda-max, or neither"
            )
        baseline_mode = "region"
        baseline_region = tuple(
            sorted((args.baseline_lambda_min, args.baseline_lambda_max))
        )

    if args.no_plot or args.skip_preprocess_dialog:
        args.initial_reference_indices = ([initial_spectrum_index] if initial_spectrum_index is not None else None)
        args.final_reference_indices = None
        corrected, _, used_region = apply_preprocessing_choices(
            args,
            experiment,
            drop_indices,
            reaction_start_time,
            reaction_end_time,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
            initial_spectrum_index=initial_spectrum_index,
        )
        return (
            corrected,
            work_range,
            c0,
            reaction_start_time,
            reaction_end_time,
            fix_initial_spectrum,
            fix_final_spectrum,
            initial_spectrum_index,
            prepare_initial_reference_spectrum(
                experiment,
                initial_spectrum_index,
                baseline_mode,
                baseline_region,
                baseline_points,
                args.baseline_window,
                work_range,
                used_region,
            ),
            None,
        )

    while True:
        (
            drop_indices,
            reaction_start_time,
            reaction_end_time,
            fix_initial_spectrum,
            initial_spectrum_index,
            fix_final_spectrum,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
            c0,
            final_spectrum_indices,
            intermediate_references,
        ) = ask_preprocessing_choices(
            experiment,
            baseline_points,
            default_work_range=work_range,
            default_c0=c0,
            baseline_window=args.baseline_window,
            default_reaction_start_time=reaction_start_time,
            default_reaction_end_time=reaction_end_time,
            allowed_work_range=allowed_work_range,
            initial_spectrum_label=initial_spectrum_label,
            default_fix_initial_spectrum=fix_initial_spectrum,
            initial_spectrum_unavailable_reason=initial_spectrum_unavailable_reason,
            default_initial_spectrum_index=initial_spectrum_index,
            default_final_spectrum_indices=final_spectrum_indices,
            species_labels=species_labels,
            supplied_species=supplied_species,
            default_intermediate_references=intermediate_references,
            final_spectrum_label=final_spectrum_label,
            default_fix_final_spectrum=fix_final_spectrum,
            final_spectrum_unavailable_reason=final_spectrum_unavailable_reason,
            overview_auto_region=overview_auto_region,
            preferred_baseline_region=preferred_baseline_region,
        )
        corrected, cropped, used_region = apply_preprocessing_choices(
            args,
            experiment,
            drop_indices,
            reaction_start_time,
            reaction_end_time,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
            initial_spectrum_index=None,
        )
        from kinet_plotting import plot_preprocessed_experiment_preview

        plot_preprocessed_experiment_preview(
            corrected,
            cropped,
            work_range,
            baseline_mode,
            used_region,
            baseline_points,
            reaction_start_time,
            reaction_end_time,
        )
        approve = input(
            "Approve these spectra for fitting? "
            "Enter/y = yes, n = revise trimming/baseline/range: "
        ).strip().lower()
        if approve in {"", "s", "si", "sí", "y", "yes"}:
            args.initial_reference_indices = (None if isinstance(initial_spectrum_index, FixedSpectrumFile)
                else [initial_spectrum_index] if isinstance(initial_spectrum_index, int) else initial_spectrum_index)
            args.final_reference_indices = (None if isinstance(final_spectrum_indices, FixedSpectrumFile)
                                            else final_spectrum_indices)
            args.fixed_reference_files = []
            labels = species_labels or (initial_spectrum_label, final_spectrum_label)
            selections = {0: initial_spectrum_index, len(labels)-1: final_spectrum_indices,
                          **intermediate_references}
            for column, selection in selections.items():
                if isinstance(selection, FixedSpectrumFile):
                    args.fixed_reference_files.append(dict(species=labels[column],
                        path=str(selection.path), alias=selection.alias, units=selection.units,
                        amplitude_bounds=selection.amplitude_bounds,
                        fit_amplitude=selection.fit_amplitude,
                        sha256=hashlib.sha256(selection.path.read_bytes()).hexdigest()))
            args.additional_fixed_references = {
                column: prepare_initial_reference_spectrum(experiment, selection,
                    baseline_mode, baseline_region, baseline_points, args.baseline_window,
                    work_range, used_region, c0=c0) / c0
                for column, selection in intermediate_references.items()}
            args.intermediate_reference_indices = {
                labels[column]: selection for column, selection in intermediate_references.items()
                if not isinstance(selection, FixedSpectrumFile)}
            return (
                corrected,
                work_range,
                c0,
                reaction_start_time,
                reaction_end_time,
                fix_initial_spectrum,
                fix_final_spectrum,
                initial_spectrum_index,
                prepare_initial_reference_spectrum(
                    experiment,
                    initial_spectrum_index,
                    baseline_mode,
                    baseline_region,
                    baseline_points,
                    args.baseline_window,
                    work_range,
                    used_region,
                    c0=c0,
                ),
                prepare_initial_reference_spectrum(
                    experiment, final_spectrum_indices, baseline_mode,
                    baseline_region, baseline_points, args.baseline_window,
                    work_range, used_region, c0=c0),
            )
        if approve in {"n", "no"}:
            print()
            print("Returning to preprocessing options.")
            continue
        raise ValueError("Approval choice must be yes or no")
