#!/usr/bin/env python3
"""Command-line interface and orchestration for TolKinet."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from kinet_common import (
    Experiment,
    GENERAL_MODEL_LABELS,
    HSS_TRANSSULFURATION_MODELS,
    MODEL_LABELS,
    MODEL_PRESENTATIONS,
    MODEL_SPECIES,
    PARAMETER_LABELS,
    SPECIAL_MODEL_LABELS,
)
from kinet_export import export_final_fit_outputs
from kinet_fitting import fit_model_with_auto_k_max, is_near_bound
from kinet_io import read_experiment, resolve_input_file
from kinet_known_spectra import (
    KnownSpectrumSpec,
    FixedSpectrumFile,
    build_known_spectra_matrix,
    known_spectra_wavelength_range,
    parse_known_spectrum_spec,
)
from kinet_plotting import plot_experiment_overview, plot_result, print_exploration_message
from kinet_preprocessing import crop_wavelengths, drop_spectra, parse_spectrum_selection, preprocess_experiment


class FitProgress:
    """Small terminal progress indicator for optimizer objective evaluations."""

    def __init__(self, enabled: bool = True, update_interval: float = 0.5) -> None:
        self.enabled = enabled
        self.update_interval = update_interval
        self.started = time.monotonic()
        self.last_update = 0.0
        self.evaluations = 0
        self.best_error = np.inf
        self.best_params: dict[str, float] = {}

    def __call__(self, params: dict[str, float], error: float) -> None:
        if not self.enabled:
            return
        self.evaluations += 1
        if np.isfinite(error) and error < self.best_error:
            self.best_error = error
            self.best_params = params.copy()

        now = time.monotonic()
        if self.evaluations > 1 and now - self.last_update < self.update_interval:
            return
        self.last_update = now
        self._write_line(done=False)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._write_line(done=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _write_line(self, done: bool) -> None:
        elapsed = time.monotonic() - self.started
        if done:
            activity = "[done]"
        else:
            frames = ("|", "/", "-", "\\")
            activity = f"[{frames[self.evaluations % len(frames)]}]"
        best = (
            "best n/a"
            if not np.isfinite(self.best_error)
            else f"best {self.best_error:.6g}"
        )
        params = " ".join(
            f"{PARAMETER_LABELS.get(name, name)}={value:.4g}"
            for name, value in self.best_params.items()
        )
        status = "done" if done else "fitting"
        line = (
            f"\r{activity} {status}: eval {self.evaluations}  "
            f"{best}  elapsed {elapsed:.1f}s"
        )
        if params:
            line += f"  {params}"
        sys.stderr.write(line[:180].ljust(180))
        sys.stderr.flush()


def print_startup_banner() -> None:
    """Print the program title and short description."""
    width = 72
    lines = [
        "TOLKinetics",
        "Multiwavelength kinetic analysis",
        "Factor analysis and global fitting of spectral time series",
        "Personal routines by J. P. Marcolongo, 2025-2026",
    ]

    print()
    print("+" + "-" * (width - 2) + "+")
    for i, line in enumerate(lines):
        if i == 0:
            print("|" + line.center(width - 2) + "|")
            print("|" + " " * (width - 2) + "|")
        else:
            print("|" + line.center(width - 2) + "|")
    print("+" + "-" * (width - 2) + "+")
    print()



def ask_model_choice(default_model: str) -> str:
    """Ask for the kinetic model to use."""
    if default_model not in MODEL_LABELS:
        raise ValueError(f"Unknown kinetic model: {default_model}")

    print("Kinetic model:")
    general_keys = list(GENERAL_MODEL_LABELS)
    for i, key in enumerate(general_keys, start=1):
        marker = "default" if key == default_model else ""
        suffix = f" [{marker}]" if marker else ""
        print(f"  {i}. {MODEL_LABELS[key]}{suffix}")
    special_index = len(general_keys) + 1
    special_suffix = " [default]" if default_model in SPECIAL_MODEL_LABELS else ""
    print(f"  {special_index}. Special mechanisms{special_suffix}")

    choice = input("Choose a model (Enter = default): ").strip()
    if not choice:
        return default_model

    index = int(choice)
    if index < 1 or index > special_index:
        raise ValueError(f"Model choice must be between 1 and {special_index}")
    if index <= len(general_keys):
        return general_keys[index - 1]

    selected = special_study_choice(spectra_only=True)
    return default_model if selected is None else selected[1]



def print_model_presentation(model: str) -> None:
    """Print a compact description of the selected kinetic model."""
    presentation = MODEL_PRESENTATIONS[model]
    print()
    print("Model overview")
    print(f"  Model: {MODEL_LABELS[model]}")
    print(f"  Scheme: {presentation['scheme']}")
    print("  Fitted absorbing species: " + ", ".join(MODEL_SPECIES[model]))
    print("  Time evolution:")
    for line in presentation["profiles"]:
        print(f"    {line}")
    print("  Parameters:")
    for line in presentation["parameters"]:
        print(f"    {line}")
    for line in presentation["notes"]:
        print(f"  Note: {line}")
    print()


def apply_model_default_wavelength_range(
    args: argparse.Namespace,
    model: str,
) -> None:
    """Use the Soret/visible window by default for special mechanisms."""
    if model in HSS_TRANSSULFURATION_MODELS:
        default_min, default_max = 385.0, 650.0
    elif model in SPECIAL_MODEL_LABELS:
        default_min, default_max = 400.0, 650.0
    else:
        default_min, default_max = 320.0, 820.0
    if args.lambda_min is None:
        args.lambda_min = default_min
    if args.lambda_max is None:
        args.lambda_max = default_max


def endpoint_spectrum_menu_labels(
    model: str,
    interactive: bool,
    initial_spectrum_index: int | None,
) -> tuple[str | None, str | None]:
    """Return species labels exposed by the endpoint-spectrum dialog."""
    first_species = MODEL_SPECIES[model][0]
    last_species = MODEL_SPECIES[model][-1]
    initial_label = first_species if interactive or initial_spectrum_index is not None else None
    final_label = last_species if interactive else None
    return initial_label, final_label


def parse_known_spectrum_specs(texts: list[str]) -> list[KnownSpectrumSpec]:
    """Parse all known-spectrum CLI specifications."""
    return [parse_known_spectrum_spec(text) for text in texts]



def install_endpoint_references(known_spectra, known_species, species_labels,
                                initial_reference, final_reference, c0):
    """Insert fixed absorbance references without creating adjustable scales."""
    for column, label, reference in (
        (0, species_labels[0], initial_reference),
        (-1, species_labels[-1], final_reference),
    ):
        if reference is None:
            continue
        if label in known_species:
            raise ValueError(f"A reference spectrum for {label} was provided twice")
        if known_spectra is None:
            known_spectra = np.full((len(reference), len(species_labels)), np.nan)
        else:
            known_spectra = known_spectra.copy()
        known_spectra[:, column] = reference / c0
    return known_spectra


def print_known_spectra_report(known_species: tuple[str, ...], model: str,
                               fixed_references: tuple[str, ...] = ()) -> None:
    """Print which spectral shapes are known and which spectra remain fitted."""
    if not known_species and not fixed_references:
        print("Known spectra: none; all species will be fitted.")
        return
    fitted = tuple(label for label in MODEL_SPECIES[model]
                   if label not in known_species and label not in fixed_references)
    if known_species:
        print("Known spectral shapes with adjustable scale:", ", ".join(known_species))
    if fixed_references:
        print("References fixed in shape and amplitude:", ", ".join(fixed_references))
    if fitted:
        print("Spectra fitted:", ", ".join(fitted))
    else:
        print("Spectra fitted: none; all species have supplied references.")
    print()


def print_initial_spectrum_report(fix_initial_spectrum: bool, model: str) -> None:
    """Print whether the first model spectrum is fixed from the first data spectrum."""
    if not fix_initial_spectrum:
        return
    first_species = MODEL_SPECIES[model][0]
    fitted = tuple(label for label in MODEL_SPECIES[model][1:])
    print(
        "Initial spectrum fixed: "
        f"{first_species} uses the first experimental spectrum; "
        "remaining spectra fitted: "
        + (", ".join(fitted) if fitted else "none")
    )
    print()


def print_initial_reference_report(index: int | tuple[int, ...] | None, experiment: Experiment) -> None:
    """Report an initial spectrum selected independently of the fit window."""
    if index is None:
        return
    if isinstance(index, FixedSpectrumFile):
        mode = "adjustable amplitude" if index.fit_amplitude else "fixed amplitude"
        print(f"Initial reference from file ({mode}): {index.path}")
        return
    indices = [index] if isinstance(index, int) else list(index)
    print("Initial reference fixed in shape and amplitude: original spectra "
          + ", ".join(str(i+1) for i in indices)
          + (" (mean)" if len(indices) > 1 else ""))
    print()


def allowed_work_range_from_known_spectra(
    known_specs: list[KnownSpectrumSpec],
    experiment: Experiment,
) -> tuple[float, float] | None:
    """Return the wavelength interval shared by experiment and known spectra."""
    known_range = known_spectra_wavelength_range(known_specs)
    if known_range is None:
        return None
    lower = max(float(experiment.wavelength[0]), known_range[0])
    upper = min(float(experiment.wavelength[-1]), known_range[1])
    if lower >= upper:
        raise ValueError(
            "Experiment and known spectra do not share a wavelength overlap"
        )
    return lower, upper


def print_allowed_work_range_report(work_range: tuple[float, float] | None) -> None:
    """Print the wavelength range allowed by known spectra, if any."""
    if work_range is None:
        return
    print(
        "Known spectra limit the selectable fit range to: "
        f"{work_range[0]:g}-{work_range[1]:g} nm"
    )
    print()


def print_final_spectrum_report(fix_final_spectrum: bool, model: str) -> None:
    """Print whether the last model spectrum is fixed from the last data spectrum."""
    if not fix_final_spectrum:
        return
    last_species = MODEL_SPECIES[model][-1]
    fitted = tuple(label for label in MODEL_SPECIES[model][:-1])
    print(
        "Final spectrum fixed: "
        f"{last_species} uses the last experimental spectrum; "
        "remaining spectra fitted: "
        + (", ".join(fitted) if fitted else "none")
    )
    print()


def ask_hss_ratio(default_ratio: float) -> float:
    """Ask for the initial added HSS-/Mb ratio used by the transsulfuration model."""
    print()
    text = input(
        "Initial ratio R_HSS = [HSS-]_added/[Mb]_0? "
        f"Enter = {default_ratio:g}: "
    ).strip()
    if not text:
        ratio = default_ratio
    else:
        ratio = float(text)
    if ratio < 0:
        raise ValueError("R_HSS must be nonnegative")
    return ratio



def print_fit_report(
    input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    reaction_end_time: float | None,
    model: str,
) -> None:
    """Print the numerical summary for one fit."""
    k_min, k_max = final_k_bounds

    print(f"Input file: {input_path}")
    print(
        "Data matrix: "
        f"{experiment.absorbance.shape[0]} wavelengths x "
        f"{experiment.absorbance.shape[1]} times"
    )
    print(f"Wavelength range: {experiment.wavelength[0]:g} - {experiment.wavelength[-1]:g}")
    print(f"Time range: {experiment.t[0]:g} - {experiment.t[-1]:g}")
    if reaction_start_time is not None:
        print(
            f"reaction start cutoff: t >= {reaction_start_time:g} original time units"
        )
        print(f"corrected time: t_original - {reaction_start_time:g} (exact t0)")
    if reaction_end_time is not None:
        print(f"reaction end time: {reaction_end_time:g} original time units")
    print(f"c0: {c0:g} M")
    print(f"model: {MODEL_LABELS[model]}")
    if model in HSS_TRANSSULFURATION_MODELS:
        print(f"R_HSS added: {args.hss_ratio:g}")
    print(f"fit method: {result.method}")
    if result.known_species:
        print("known spectral shapes: " + ", ".join(result.known_species))
    if result.known_spectrum_scales:
        for label, scale in result.known_spectrum_scales.items():
            print(f"known spectrum scale {label}: {scale:.10g}")
    if result.fixed_initial_spectrum:
        print(
            "initial spectrum fixed from first experimental spectrum: "
            f"{result.species_labels[0]}"
        )
    if result.fixed_final_spectrum:
        print(
            "final spectrum fixed from last experimental spectrum: "
            f"{result.species_labels[-1]}"
        )
    if args.initial_spectrum_weight > 0:
        print(f"initial spectrum weight: {args.initial_spectrum_weight:g}")
    if k_max == args.k_max:
        print(f"k search range: {k_min:g} - {k_max:g}")
    else:
        print(f"k search range: {k_min:g} - {k_max:g} (expanded from {args.k_max:g})")
    if model in {"mbfe3_sulfide_irreversible_binding_autocatalytic", "femb_disulfide_irreversible"}:
        print("k_off: 0 (fixed; not fitted)")
    for name, value in result.params.items():
        print(f"{PARAMETER_LABELS.get(name, name)}: {value:.10g}")
        if is_near_bound(value, k_min):
            print(
                "  warning: "
                f"{PARAMETER_LABELS.get(name, name)} reached the lower search bound"
            )
        elif is_near_bound(value, k_max):
            print(
                "  warning: "
                f"{PARAMETER_LABELS.get(name, name)} reached the upper search bound"
            )
    if set(result.params) == {"k"}:
        fitted_k = next(iter(result.params.values()))
        print(f"half-life: {np.log(2) / fitted_k:.10g}")
    if result.selection_diagnostics is not None:
        d=result.selection_diagnostics
        print(f"FeMb selection ({d.get('selection_mode', 'automatic')}): "
              f"{d['n_candidates']} candidates; {d['n_chemical']} chemically eligible")
        print("Selected profile RMS fractions:",d['selected']['profile_rms'])
        print("Selected species peaks (nm):",d['selected']['peaks_nm'])
        print("Selected intermediate/product peak epsilon ratio:",
              d['selected']['B_peak_over_C_peak'])
        if not d['chemical_filter_passed']:
            print("The selected candidate does not pass the chemical filters; inspect species spectra.")
    print(f"fit error: {result.error:.10g}")
    print(f"minimum recovered spectrum value: {result.spectra.min():.10g}")
    print("first singular values:", " ".join(f"{x:.6g}" for x in result.singular_values[:10]))
    print()



def ask_post_fit_spectra_to_drop(experiment: Experiment) -> list[int] | None:
    """Ask whether to remove spectra after inspecting the final residuals panel."""
    while True:
        print("Use the indices shown in the residual panel tooltips.")
        choice = input(
            "Spectra to remove before refitting? "
            "E.g. 3 or 3,7-9. Enter/n = finish: "
        ).strip()
        normalized = choice.lower()
        if normalized in {"", "n", "no", "fin", "q", "quit", "exit", "salir"}:
            return None

        try:
            drop_indices = parse_spectrum_selection(choice, experiment.t.size)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")
            continue

        if not drop_indices:
            return None
        if experiment.t.size - len(drop_indices) < 2:
            print("At least 2 spectra must remain for refitting.")
            continue
        return drop_indices



def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        description="Fit multiwavelength spectrophotometric kinetic data.",
        epilog="Joint temperature/pH analysis: tolkien-tools kinetics activation --help"
    )
    parser.add_argument("input", nargs="?", default=None, help="Input text file (requested when starting an interactive analysis)")
    parser.add_argument(
        "--global-manifest",
        default=None,
        help=(
            "CSV/TSV with filename,R,c0_after for a shared-parameter "
            "transsulfuration fit"
        ),
    )
    parser.add_argument(
        "--global-root",
        default=None,
        help="Base directory for relative filenames in --global-manifest",
    )
    parser.add_argument(
        "--global-time-zero",
        type=float,
        default=47.0,
        help="Original time of HSS- addition for global fitting",
    )
    parser.add_argument(
        "--global-output-dir",
        default=None,
        help="Optional directory for compact global-fit text outputs",
    )
    parser.add_argument(
        "--global-max-starts",
        type=int,
        default=None,
        help="Optional limit on optimizer multistarts for a global fit",
    )
    parser.add_argument(
        "--global-fix-k-fast",
        type=float,
        default=None,
        help=(
            "Fix global k_fast (s^-1) to an independently measured value "
            "instead of optimizing it"
        ),
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=None,
        help=(
            "Lower fit wavelength. Defaults to 400 nm for special mechanisms "
            "(385 nm for individual transsulfuration fits) and 320 nm for general models."
        ),
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=None,
        help=(
            "Upper fit wavelength. Defaults to 650 nm for special mechanisms "
            "and 820 nm for general models."
        ),
    )
    parser.add_argument("--baseline-points", type=int, default=20)
    parser.add_argument(
        "--baseline-mode",
        choices=("auto-flat", "region", "points", "none"),
        default="auto-flat",
        help="Baseline correction mode",
    )
    parser.add_argument(
        "--baseline-window",
        type=float,
        default=40.0,
        help="Window width in nm for automatic flat baseline search",
    )
    parser.add_argument(
        "--baseline-lambda-min",
        type=float,
        default=None,
        help="Lower wavelength limit for baseline correction region",
    )
    parser.add_argument(
        "--baseline-lambda-max",
        type=float,
        default=None,
        help="Upper wavelength limit for baseline correction region",
    )
    parser.add_argument(
        "--drop-spectra",
        default="",
        help="1-based spectra to remove before fitting, e.g. '1' or '1,4-6'",
    )
    parser.add_argument(
        "--reaction-start-time",
        type=float,
        default=None,
        help=(
            "Exact reaction time zero in original time units. Keep spectra with "
            "t_original >= t0 and use t = t_original - t0, even between frames."
        ),
    )
    parser.add_argument(
        "--reaction-end-time",
        type=float,
        default=None,
        help=(
            "Optional original time where the fitted window ends. Useful for "
            "diagnostic fits restricted to an initial phase."
        ),
    )
    parser.add_argument(
        "--initial-spectrum-index",
        type=int,
        default=None,
        help=(
            "1-based original spectrum index to use as A, independently of "
            "the time window kept for fitting"
        ),
    )
    parser.add_argument(
        "--kd-lambda-start",
        type=float,
        default=None,
        help="Override first wavelength when converting KD files",
    )
    parser.add_argument(
        "--kd-lambda-step",
        type=float,
        default=None,
        help="Override wavelength step when converting KD files",
    )
    parser.add_argument("--c0", type=float, default=1.0, help="Initial concentration in M")
    parser.add_argument(
        "--hss-ratio",
        type=float,
        default=20.0,
        help=(
            "Initial added HSS-/Mb ratio for the MbFeIII-HS/HSS "
            "transsulfuration special model"
        ),
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=1e-8,
        help="Lower bound for k during log-space optimization",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=1e-1,
        help="Upper bound for k during log-space optimization",
    )
    parser.add_argument(
        "--no-auto-expand-k-max",
        action="store_false",
        dest="auto_expand_k_max",
        help="Disable automatic expansion when a fitted constant reaches --k-max",
    )
    parser.add_argument(
        "--k-max-expand-factor",
        type=float,
        default=10.0,
        help="Multiplier used when automatically expanding --k-max",
    )
    parser.add_argument(
        "--k-max-expand-steps",
        type=int,
        default=4,
        help="Maximum number of automatic --k-max expansions",
    )
    parser.add_argument(
        "--fit-method",
        choices=("nnls", "mcr_als"),
        default="nnls",
        help=(
            "Spectral fitting method. mcr_als is experimental and currently "
            "supports only A -> B -> C and A <-> B -> C."
        ),
    )
    parser.add_argument(
        "--mcr-kinetic-weight",
        type=float,
        default=1.0,
        help="Soft MCR-ALS weight keeping concentrations near the kinetic profile",
    )
    parser.add_argument(
        "--mcr-max-iter",
        type=int,
        default=200,
        help="Maximum number of MCR-ALS iterations",
    )
    parser.add_argument(
        "--known-spectrum",
        action="append",
        default=[],
        help=(
            "Fix one or more species spectra from a pure-spectra file. "
            "Examples: --known-spectrum 'A B spectra.dat' or "
            "'A,B=spectra.dat'. May be repeated."
        ),
    )
    parser.add_argument(
        "--initial-spectrum-weight",
        type=float,
        default=0.0,
        help=(
            "Softly pull the first model species toward the first measured "
            "spectrum in NNLS. "
            "0 disables it; 1 is roughly one extra time point."
        ),
    )
    parser.add_argument(
        "--fix-initial-spectrum",
        action="store_true",
        help=(
            "Fix the first species spectrum to the first experimental spectrum "
            "and fit only the remaining species spectra."
        ),
    )
    parser.add_argument(
        "--fix-final-spectrum",
        action="store_true",
        help=(
            "Fix the last species spectrum to the last experimental spectrum "
            "and fit only the remaining species spectra."
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_LABELS),
        default="a_to_b",
        help="Kinetic model to fit",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--no-fit-progress",
        action="store_true",
        help="Do not show optimizer progress while fitting",
    )
    parser.add_argument(
        "--skip-preprocess-dialog",
        action="store_true",
        help="Do not show the initial raw spectra dialog before preprocessing",
    )
    return parser




def transsulfuration_choice(include_global: bool = True) -> tuple[str, str] | None:
    """Group single-experiment and global transsulfuration analyses."""
    entries = [("spectra", model) for model in HSS_TRANSSULFURATION_MODELS]
    if include_global:
        entries += [("global", model) for model in HSS_TRANSSULFURATION_MODELS]
    while True:
        print("\nTranssulfuration reactions")
        for index, (kind, model) in enumerate(entries, 1):
            prefix = "Global fit: " if kind == "global" else ""
            print(f"  {index}. {prefix}{SPECIAL_MODEL_LABELS[model]}")
        print("  0. Back")
        choice = input("Choose a transsulfuration analysis: ").strip().lower()
        if choice in {"0", "q", "quit", "exit", "back", "salir"}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            return entries[int(choice) - 1]
        print("Choose one of the listed options.")


SPECIAL_SPECTRAL_MODELS = (
    "mbfe3_sulfide_autocatalytic",
    "mbfe3_sulfide_irreversible_binding_autocatalytic",
    "mbfe3_sulfide_binding_autocatalytic",
    "mbfe3_hss_no_binding",
    "femb_disulfide_irreversible",
    "femb_disulfide_reversible",
)


def special_study_choice(spectra_only: bool = False) -> tuple[str, str | None] | None:
    """Choose a study or mechanism, grouping transsulfuration analyses."""
    entries = [] if spectra_only else [("activation", None)]
    entries += [("spectra", model) for model in SPECIAL_SPECTRAL_MODELS]
    entries += [("transsulfuration", None)]
    while True:
        print("\nSpecial mechanisms and studies")
        for index, (kind, model) in enumerate(entries, 1):
            if kind == "activation":
                label = "Thermal analysis of disulfide acid-base speciation"
            elif kind == "transsulfuration":
                label = "Transsulfuration reactions"
            else:
                label = SPECIAL_MODEL_LABELS[model]
            print(f"  {index}. {label}")
        print("  0. Back")
        choice = input("Choose a mechanism or study: ").strip().lower()
        if choice in {"0", "q", "quit", "exit", "back", "salir"}:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            kind, model = entries[int(choice) - 1]
            if kind == "transsulfuration":
                selected = transsulfuration_choice(include_global=not spectra_only)
                if selected is None:
                    continue
                return selected
            return kind, model
        print("Choose one of the listed options.")


def kinetics_menu(args: argparse.Namespace) -> bool:
    """Choose a mechanism first; request its input only when ready to run."""
    models = list(GENERAL_MODEL_LABELS)
    while True:
        print("\nKinetics main menu")
        for index, model in enumerate(models, 1):
            print(f"  {index}. {GENERAL_MODEL_LABELS[model]}")
        print("  4. Special mechanisms and studies")
        print("  0. Back / exit")
        choice = input("Choose a mechanism or study: ").strip().lower()
        if choice in {"0", "q", "quit", "exit", "back", "salir"}:
            return False
        if choice in {"1", "2", "3"}:
            kind, model = "spectra", models[int(choice)-1]
        elif choice == "4":
            selected = special_study_choice()
            if selected is None:
                continue
            kind, model = selected
        else:
            print("Choose an option from 0 to 4.")
            continue
        if kind == "activation":
            from kinet_activation import main as activation_main
            activation_main(([args.input] if args.input else []) + ["--interactive"])
            return False
        if kind == "global":
            source = args.global_manifest or args.input
            if not source:
                source = input("Experiment table CSV/TSV (Enter = back): ").strip()
            if not source:
                continue
            args.global_manifest = source
        elif not args.input:
            source = input("Spectra file TXT/DAT/KD (Enter = back): ").strip()
            if not source:
                continue
            args.input = source
        args.model = model
        return True


def main() -> None:
    """Load data, preprocess, fit the kinetic model and report results."""
    if len(sys.argv) > 1 and sys.argv[1] == "activation":
        from kinet_activation import main as activation_main
        activation_main(sys.argv[2:])
        return
    args = build_parser().parse_args()

    use_menu = (args.global_manifest is None
                and not args.no_plot and not args.skip_preprocess_dialog)
    print_startup_banner()
    if use_menu and not kinetics_menu(args):
        return
    if args.input is None:
        args.input = "117.txt"  # Preserve the legacy noninteractive default.

    if args.global_manifest is not None:
        if args.model not in HSS_TRANSSULFURATION_MODELS:
            raise ValueError(
                "--global-manifest requires one of the two transsulfuration models"
            )
        if args.lambda_min is None:
            args.lambda_min = 410.0
        if args.lambda_max is None:
            args.lambda_max = 650.0
        if args.baseline_lambda_min is None:
            args.baseline_lambda_min = 750.0
        if args.baseline_lambda_max is None:
            args.baseline_lambda_max = 820.0
        from kinet_global import run_global_from_args

        run_global_from_args(args)
        return

    source_input_path = Path(args.input)
    input_path = resolve_input_file(args)
    experiment = read_experiment(input_path)

    model = args.model
    fit_method = args.fit_method
    known_specs = parse_known_spectrum_specs(args.known_spectrum)
    interactive = not args.no_plot and not args.skip_preprocess_dialog
    fix_initial_spectrum = args.fix_initial_spectrum
    fix_final_spectrum = args.fix_final_spectrum
    overview_auto_region = None
    preferred_baseline_region = None
    if interactive:
        print_exploration_message()
        overview_auto_region = plot_experiment_overview(
            experiment,
            baseline_window=args.baseline_window,
        )
        if not use_menu:
            model = ask_model_choice(args.model)
        print_model_presentation(model)
    else:
        print_model_presentation(model)
    apply_model_default_wavelength_range(args, model)
    if model in {"femb_disulfide_reversible", "femb_disulfide_irreversible"}:
        preferred_baseline_region = (750.,850.)
    if model in HSS_TRANSSULFURATION_MODELS and interactive:
        args.hss_ratio = ask_hss_ratio(args.hss_ratio)
    if (
        model in {"mbfe3_hss_no_binding", "mbfe3_sulfide_autocatalytic"}
        and interactive
    ):
        from kinet_plotting import (
            plot_hss_no_binding_t0_diagnostic,
            plot_sulfide_no_binding_t0_diagnostic,
        )
        from kinet_t0 import (
            estimate_hss_no_binding_t0,
            estimate_sulfide_no_binding_t0,
        )

        is_hss = model == "mbfe3_hss_no_binding"
        estimator = (
            estimate_hss_no_binding_t0 if is_hss else estimate_sulfide_no_binding_t0
        )
        reagent = "HSS-" if is_hss else "HS-"
        intermediate = "MbFeIII-HSS" if is_hss else "MbFeIII-HS"

        try:
            t0_estimate = estimator(experiment)
        except ValueError as exc:
            print(f"Automatic t0 suggestion unavailable: {exc}")
            print()
        else:
            print()
            print(f"Automatic t0 suggestion for the {reagent} model without binding")
            print(
                "  Diagnostic baseline: "
                f"{t0_estimate.baseline_region[0]:g}-"
                f"{t0_estimate.baseline_region[1]:g} nm"
            )
            print(
                f"  Detected {reagent} addition step: "
                f"t = {t0_estimate.addition_time:g}"
            )
            if is_hss:
                print(
                    "  Rapid binding decay at "
                    f"{t0_estimate.diagnostic_wavelength:g} nm: "
                    f"k_fast = {t0_estimate.binding_rate:.6g}"
                )
                print(
                    "  Estimated binding endpoint "
                    f"({t0_estimate.completion_time_constants:g} tau): "
                    f"t = {t0_estimate.binding_end_time:.6g}"
                )
                print(
                    "  Criterion: first measured spectrum after the rapid "
                    "binding phase is about 98% complete."
                )
            else:
                print(
                    "  Criterion: first spectrum within 0.5% of the early "
                    f"{intermediate} plateau."
                )
            print(f"  Suggested reaction start: t = {t0_estimate.recommended_time:g}")
            print("Close the diagnostic figure to continue.")
            print()
            if is_hss:
                plot_hss_no_binding_t0_diagnostic(experiment, t0_estimate)
            else:
                plot_sulfide_no_binding_t0_diagnostic(
                    experiment,
                    t0_estimate,
                    reagent=reagent,
                    intermediate=intermediate,
                )
            if args.reaction_start_time is None:
                args.reaction_start_time = t0_estimate.recommended_time
            overview_auto_region = t0_estimate.baseline_region
            preferred_baseline_region = t0_estimate.baseline_region

    allowed_work_range = allowed_work_range_from_known_spectra(known_specs, experiment)
    print_allowed_work_range_report(allowed_work_range)

    first_species = MODEL_SPECIES[model][0]
    last_species = MODEL_SPECIES[model][-1]
    supplied_species = {
        MODEL_SPECIES[model]["ABC".index(alias)]
        for spec in known_specs for alias in spec.species
        if "ABC".index(alias) < len(MODEL_SPECIES[model])}
    initial_spectrum_unavailable_reason = None
    if first_species in supplied_species:
        initial_spectrum_unavailable_reason = (
            "The first model spectrum was already supplied as a known spectrum "
            f"({first_species}); it cannot also be fixed from the first experimental spectrum."
        )
    final_spectrum_unavailable_reason = None
    if last_species in supplied_species:
        final_spectrum_unavailable_reason = (
            "The last model spectrum was already supplied as a known spectrum "
            f"({last_species}); it cannot also be fixed from the last experimental spectrum."
        )

    initial_spectrum_label, final_spectrum_label = endpoint_spectrum_menu_labels(
        model,
        interactive,
        args.initial_spectrum_index,
    )

    (
        corrected,
        work_range,
        c0,
        reaction_start_time,
        reaction_end_time,
        fix_initial_spectrum,
        fix_final_spectrum,
        initial_spectrum_index,
        initial_reference_spectrum,
        final_reference_spectrum,
    ) = preprocess_experiment(
        args,
        experiment,
        allowed_work_range=allowed_work_range,
        initial_spectrum_label=initial_spectrum_label,
        default_fix_initial_spectrum=fix_initial_spectrum,
        initial_spectrum_unavailable_reason=initial_spectrum_unavailable_reason,
        final_spectrum_label=final_spectrum_label,
        default_fix_final_spectrum=fix_final_spectrum,
        final_spectrum_unavailable_reason=final_spectrum_unavailable_reason,
        overview_auto_region=overview_auto_region,
        preferred_baseline_region=preferred_baseline_region,
        species_labels=MODEL_SPECIES[model],
        supplied_species=supplied_species,
    )
    if allowed_work_range is not None:
        print(f"Fit wavelength range selected: {work_range[0]:g}-{work_range[1]:g} nm")
        print()
    cropped = crop_wavelengths(corrected, work_range[0], work_range[1])
    known_spectra, known_species = build_known_spectra_matrix(
        known_specs,
        MODEL_SPECIES[model],
        cropped.wavelength,
    )
    known_spectra = install_endpoint_references(
        known_spectra, known_species, MODEL_SPECIES[model],
        initial_reference_spectrum, final_reference_spectrum, c0)
    for column, reference in args.additional_fixed_references.items():
        label = MODEL_SPECIES[model][column]
        if label in known_species:
            raise ValueError(f"Reference spectrum for {label} was provided twice")
        if known_spectra is None:
            known_spectra = np.full((cropped.wavelength.size, len(MODEL_SPECIES[model])), np.nan)
        known_spectra[:, column] = reference
    fixed_references = tuple(label for label, reference, fixed in (
        (first_species, initial_reference_spectrum, fix_initial_spectrum),
        (last_species, final_reference_spectrum, fix_final_spectrum))
        if reference is not None or fixed) + tuple(
            MODEL_SPECIES[model][i] for i in args.additional_fixed_references)
    adjustable_files = tuple(reference['species'] for reference in args.fixed_reference_files
                             if reference['fit_amplitude'])
    known_species = (*known_species, *adjustable_files)
    fixed_references = tuple(label for label in fixed_references if label not in adjustable_files)
    print_known_spectra_report(known_species, model, fixed_references)
    print_initial_spectrum_report(fix_initial_spectrum, model)
    print_initial_reference_report(initial_spectrum_index, experiment)
    print_final_spectrum_report(fix_final_spectrum, model)
    if final_reference_spectrum is not None and args.final_reference_indices is not None:
        print("Product reference fixed in shape and amplitude: original spectra "
              + ", ".join(str(i+1) for i in args.final_reference_indices))
    for reference in args.fixed_reference_files:
        mode = "adjustable amplitude" if reference["fit_amplitude"] else "fixed amplitude"
        print(f"Reference for {reference['species']} ({reference['units']}, {mode}): {reference['path']}")
    n_components = len(MODEL_SPECIES[model])

    fit_experiment = cropped
    while True:
        sys.stdout.flush()
        progress = FitProgress(enabled=not args.no_fit_progress)
        try:
            result, final_k_bounds = fit_model_with_auto_k_max(
                model,
                fit_experiment,
                c0=c0,
                n_components=n_components,
                method=fit_method,
                k_bounds=(args.k_min, args.k_max),
                auto_expand=args.auto_expand_k_max,
                expand_factor=args.k_max_expand_factor,
                max_expand_steps=args.k_max_expand_steps,
                hss_ratio=args.hss_ratio,
                initial_spectrum_weight=args.initial_spectrum_weight,
                known_spectra=known_spectra,
                known_species=known_species,
                known_spectrum_scale_bounds={r["species"]: r["amplitude_bounds"]
                    for r in args.fixed_reference_files if r["fit_amplitude"]},
                fix_initial_spectrum=fix_initial_spectrum,
                fix_final_spectrum=fix_final_spectrum,
                mcr_kinetic_weight=args.mcr_kinetic_weight,
                mcr_max_iter=args.mcr_max_iter,
                progress_callback=progress,
            )
        finally:
            progress.finish()

        if model == "femb_disulfide_reversible" and interactive:
            from kinet_candidate_explorer import explore_candidates
            result = explore_candidates(fit_experiment, result, input_filename=input_path.name)

        print_fit_report(
            input_path,
            fit_experiment,
            result,
            final_k_bounds,
            args,
            c0,
            reaction_start_time,
            reaction_end_time,
            model,
        )

        if args.no_plot:
            break

        plot_result(fit_experiment, result, input_filename=input_path.name)
        drop_indices = ask_post_fit_spectra_to_drop(fit_experiment)
        if drop_indices is None:
            break

        removed = ", ".join(
            f"{index + 1} (t={fit_experiment.t[index]:.6g})"
            for index in drop_indices
        )
        print(f"Refitting without spectra: {removed}")
        print()
        fit_experiment = drop_spectra(fit_experiment, drop_indices)

    export_final_fit_outputs(
        source_input_path,
        input_path,
        fit_experiment,
        result,
        final_k_bounds,
        args,
        c0,
        reaction_start_time,
        reaction_end_time,
        model,
        protected_input_paths=([spec.path for spec in known_specs]
                               + [Path(r["path"]) for r in args.fixed_reference_files]),
    )



if __name__ == "__main__":
    main()
