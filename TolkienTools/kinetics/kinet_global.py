#!/usr/bin/env python3
"""Global multi-R fits for MbFeIII-HS/HSS- transsulfuration series."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kd_to_txt import extract_absorbance, extract_times, find_spectrum_blocks, infer_wavelength_axis
from kinet_common import Experiment, HSS_TRANSSULFURATION_MODELS, MODEL_SPECIES
from kinet_fitting import fit_direct_spectra, optimize_kinetic_parameters
from kinet_io import read_experiment
from kinet_models import (
    concentration_profile_mbfe3_sulfide_hss_transsulfuration,
    concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto,
)
from kinet_preprocessing import baseline_correct_region, crop_wavelengths


COPASI_REFERENCE_PARAMETERS = {
    "k_slow": 9.0e-5,
    "k_ts": 2.01e3,
    "k_fast": 1.1e-2,
}


@dataclass(frozen=True)
class GlobalExperiment:
    """One preprocessed experiment and its known chemical conditions."""

    name: str
    source_path: Path
    experiment: Experiment
    hss_ratio: float
    c0: float


@dataclass(frozen=True)
class GlobalExperimentResult:
    """Reconstruction of one member of a global series."""

    name: str
    source_path: Path
    hss_ratio: float
    c0: float
    experiment: Experiment
    concentrations: np.ndarray
    absorbance_calc: np.ndarray
    residuals: np.ndarray
    rmse: float


@dataclass(frozen=True)
class GlobalFitResult:
    """Shared parameters/spectra and per-experiment reconstructions."""

    model: str
    params: dict[str, float]
    species_labels: tuple[str, ...]
    spectra: np.ndarray
    experiments: tuple[GlobalExperimentResult, ...]
    error: float
    fixed_initial_spectrum: bool
    fixed_final_spectrum: bool
    fixed_parameters: dict[str, float]


def _read_kd_in_memory(path: Path) -> Experiment:
    """Read a KD experiment without creating a converted sidecar file."""
    data = path.read_bytes()
    blocks = find_spectrum_blocks(data)
    absorbance = extract_absorbance(data, blocks)
    times = extract_times(data, blocks)
    wavelength = infer_wavelength_axis(
        data,
        first_data_start=blocks[0][0],
        n_points=absorbance.shape[0],
        lambda_start=None,
        lambda_step=None,
    )
    return Experiment(t=times, wavelength=wavelength, absorbance=absorbance)


def read_experiment_without_sidecars(path: Path) -> Experiment:
    """Read TXT-like or KD input without writing analysis artifacts."""
    if path.suffix.lower() == ".kd":
        return _read_kd_in_memory(path)
    return read_experiment(path)


def read_global_manifest(path: Path, root: Path | None = None) -> list[tuple[Path, float, float]]:
    """Read filename, R and c0_after from a CSV/TSV manifest."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = csv.DictReader(text.splitlines(), dialect=dialect)
    if rows.fieldnames is None:
        raise ValueError(f"{path}: manifest has no header")
    normalized = {name.strip().lower(): name for name in rows.fieldnames}
    required = {"filename", "r", "c0_after"}
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"{path}: missing manifest columns: {', '.join(sorted(missing))}")

    base = root if root is not None else path.parent
    parsed: list[tuple[Path, float, float]] = []
    for row_number, row in enumerate(rows, start=2):
        filename = (row.get(normalized["filename"]) or "").strip()
        if not filename:
            continue
        try:
            ratio = float((row.get(normalized["r"]) or "").strip())
            c0 = float((row.get(normalized["c0_after"]) or "").strip())
        except ValueError as exc:
            raise ValueError(f"{path}: invalid numeric value in row {row_number}") from exc
        source = Path(filename)
        if not source.is_absolute():
            source = base / source
        if ratio < 0 or c0 <= 0:
            raise ValueError(f"{path}: R must be nonnegative and c0_after positive in row {row_number}")
        if not source.is_file():
            raise FileNotFoundError(f"{path}: input file not found: {source}")
        parsed.append((source, ratio, c0))
    if len(parsed) < 2:
        raise ValueError("A global fit requires at least two experiments")
    return parsed


def preprocess_global_experiment(
    source_path: Path,
    hss_ratio: float,
    c0: float,
    time_zero: float = 47.0,
    baseline_region: tuple[float, float] = (750.0, 820.0),
    fit_region: tuple[float, float] = (410.0, 650.0),
) -> GlobalExperiment:
    """Apply the agreed baseline, time origin and wavelength window."""
    raw = read_experiment_without_sidecars(source_path)
    corrected_absorbance = baseline_correct_region(raw, *baseline_region)
    keep = raw.t >= time_zero
    if not np.any(keep):
        raise ValueError(f"{source_path}: no spectra at or after t = {time_zero:g}")
    corrected = Experiment(
        t=raw.t[keep] - time_zero,
        wavelength=raw.wavelength,
        absorbance=corrected_absorbance[:, keep],
    )
    cropped = crop_wavelengths(corrected, *fit_region)
    return GlobalExperiment(
        name=source_path.name,
        source_path=source_path,
        experiment=cropped,
        hss_ratio=hss_ratio,
        c0=c0,
    )


def _validate_global_series(series: list[GlobalExperiment]) -> None:
    if len(series) < 2:
        raise ValueError("A global fit requires at least two experiments")
    reference = series[0].experiment.wavelength
    for item in series:
        experiment = item.experiment
        if experiment.t.size < 2:
            raise ValueError(f"{item.name}: at least two spectra are required")
        if experiment.wavelength.shape != reference.shape or not np.allclose(
            experiment.wavelength,
            reference,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("All global experiments must use the same wavelength grid")
        if not np.all(np.isfinite(experiment.absorbance)):
            raise ValueError(f"{item.name}: absorbance contains non-finite values")


def _profiles_for_params(
    model: str,
    series: list[GlobalExperiment],
    params: dict[str, float],
) -> list[np.ndarray]:
    profiles: list[np.ndarray] = []
    for item in series:
        if model == "mbfe3_sulfide_hss_transsulfuration_no_auto":
            c = concentration_profile_mbfe3_sulfide_hss_transsulfuration_no_auto(
                item.experiment.t,
                params["k_slow"],
                params["k_ts"],
                params["k_fast"],
                hss_ratio=item.hss_ratio,
                c0=item.c0,
            )
        elif model == "mbfe3_sulfide_hss_transsulfuration":
            c = concentration_profile_mbfe3_sulfide_hss_transsulfuration(
                item.experiment.t,
                params["k_slow"],
                params["k_auto"],
                params["k_ts"],
                params["k_fast"],
                hss_ratio=item.hss_ratio,
                c0=item.c0,
            )
        else:
            raise ValueError("Global fitting is supported only for transsulfuration models")
        profiles.append(c)
    return profiles


def _endpoint_spectra(
    series: list[GlobalExperiment],
    fix_initial_spectrum: bool,
    fix_final_spectrum: bool,
) -> np.ndarray | None:
    """Build shared endpoint spectra by averaging concentration-normalized endpoints."""
    if not fix_initial_spectrum and not fix_final_spectrum:
        return None
    n_wavelengths = series[0].experiment.wavelength.size
    known = np.full((n_wavelengths, 2), np.nan)
    if fix_initial_spectrum:
        known[:, 0] = np.mean(
            [item.experiment.absorbance[:, 0] / item.c0 for item in series],
            axis=0,
        )
    if fix_final_spectrum:
        known[:, 1] = np.mean(
            [item.experiment.absorbance[:, -1] / item.c0 for item in series],
            axis=0,
        )
    return known


def _fit_shared_spectra(
    series: list[GlobalExperiment],
    profiles: list[np.ndarray],
    known_spectra: np.ndarray | None,
) -> np.ndarray:
    """Fit one spectral matrix while giving each experiment equal total weight."""
    scaled_absorbance = []
    scaled_profiles = []
    for item, concentrations in zip(series, profiles):
        scale = 1.0 / np.sqrt(item.experiment.t.size)
        scaled_absorbance.append(item.experiment.absorbance * scale)
        scaled_profiles.append(concentrations * scale)
    absorbance = np.concatenate(scaled_absorbance, axis=1)
    concentrations = np.concatenate(scaled_profiles, axis=1)
    return fit_direct_spectra(
        absorbance,
        concentrations,
        spectra_method="nnls",
        known_spectra=known_spectra,
    )


def _global_error(
    series: list[GlobalExperiment],
    profiles: list[np.ndarray],
    spectra: np.ndarray,
) -> float:
    """Root mean of per-experiment mean squared residuals."""
    mean_squares = []
    for item, concentrations in zip(series, profiles):
        residuals = item.experiment.absorbance - spectra @ concentrations
        mean_squares.append(float(np.mean(residuals**2)))
    return float(np.sqrt(np.mean(mean_squares)))


def fit_global_transsulfuration(
    series: list[GlobalExperiment],
    model: str = "mbfe3_sulfide_hss_transsulfuration_no_auto",
    fix_initial_spectrum: bool = True,
    fix_final_spectrum: bool = True,
    initial_parameters: dict[str, float] | None = None,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    k_ts_bounds: tuple[float, float] = (1e-2, 1e5),
    max_starts: int | None = None,
    fixed_parameters: dict[str, float] | None = None,
) -> GlobalFitResult:
    """Fit shared kinetic constants and spectra across a multi-R series."""
    _validate_global_series(series)
    if model not in HSS_TRANSSULFURATION_MODELS:
        raise ValueError("Global fitting is supported only for transsulfuration models")

    known_spectra = _endpoint_spectra(series, fix_initial_spectrum, fix_final_spectrum)
    if model == "mbfe3_sulfide_hss_transsulfuration_no_auto":
        parameter_names = ("k_slow", "k_ts", "k_fast")
    else:
        parameter_names = ("k_slow", "k_auto", "k_ts", "k_fast")
    parameter_bounds = {name: k_bounds for name in parameter_names}
    parameter_bounds["k_ts"] = k_ts_bounds
    fixed_parameters = dict(fixed_parameters or {})
    unknown_fixed = set(fixed_parameters) - set(parameter_names)
    if unknown_fixed:
        raise ValueError(
            "Fixed values provided for unknown parameters: "
            + ", ".join(sorted(unknown_fixed))
        )
    for name, value in fixed_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Fixed {name} must be positive and finite")
    optimized_names = tuple(
        name for name in parameter_names if name not in fixed_parameters
    )
    if not optimized_names:
        raise ValueError("At least one kinetic parameter must remain free")

    seeds = dict(COPASI_REFERENCE_PARAMETERS)
    if "k_auto" in parameter_names:
        # No independent COPASI/literature estimate is available for this
        # phenomenological term. Use the center of its log-space bounds rather
        # than borrowing a value from an individual spectral fit.
        seeds["k_auto"] = float(np.sqrt(k_bounds[0] * k_bounds[1]))
    if initial_parameters:
        seeds.update(initial_parameters)
    seeds = {name: value for name, value in seeds.items() if name in optimized_names}

    def objective(free_params: dict[str, float]) -> float:
        params = {**fixed_parameters, **free_params}
        profiles = _profiles_for_params(model, series, params)
        spectra = _fit_shared_spectra(series, profiles, known_spectra)
        return _global_error(series, profiles, spectra)

    optimized = optimize_kinetic_parameters(
        objective,
        optimized_names,
        k_bounds,
        parameter_bounds=parameter_bounds,
        initial_parameters=seeds,
        # With both shared endpoint spectra fixed, the global objective is a
        # smooth function of log(k) and bounded L-BFGS-B is much faster. Keep
        # Powell for profiled NNLS spectra, whose active sets introduce kinks.
        optimizer=(
            "lbfgsb"
            if fix_initial_spectrum and fix_final_spectrum
            else "powell"
        ),
        max_starts=max_starts,
    )
    params = {
        name: fixed_parameters[name] if name in fixed_parameters else optimized[name]
        for name in parameter_names
    }
    profiles = _profiles_for_params(model, series, params)
    spectra = _fit_shared_spectra(series, profiles, known_spectra)
    experiment_results = []
    for item, concentrations in zip(series, profiles):
        absorbance_calc = spectra @ concentrations
        residuals = item.experiment.absorbance - absorbance_calc
        experiment_results.append(
            GlobalExperimentResult(
                name=item.name,
                source_path=item.source_path,
                hss_ratio=item.hss_ratio,
                c0=item.c0,
                experiment=item.experiment,
                concentrations=concentrations,
                absorbance_calc=absorbance_calc,
                residuals=residuals,
                rmse=float(np.sqrt(np.mean(residuals**2))),
            )
        )
    return GlobalFitResult(
        model=model,
        params=params,
        species_labels=MODEL_SPECIES[model],
        spectra=spectra,
        experiments=tuple(experiment_results),
        error=_global_error(series, profiles, spectra),
        fixed_initial_spectrum=fix_initial_spectrum,
        fixed_final_spectrum=fix_final_spectrum,
        fixed_parameters=fixed_parameters,
    )


def write_global_outputs(result: GlobalFitResult, output_dir: Path) -> None:
    """Write compact text outputs for a completed global fit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        "# Global multi-R transsulfuration fit",
        f"model\t{result.model}",
        f"global_rmse\t{result.error:.10g}",
        f"fixed_initial_spectrum\t{result.fixed_initial_spectrum}",
        f"fixed_final_spectrum\t{result.fixed_final_spectrum}",
    ]
    for name, value in result.params.items():
        suffix = "\tfixed" if name in result.fixed_parameters else ""
        summary.append(f"{name}\t{value:.10g}{suffix}")
    summary.append("")
    summary.append("# experiment\tR\tc0_M\tn_spectra\trmse")
    for item in result.experiments:
        summary.append(
            f"{item.name}\t{item.hss_ratio:.10g}\t{item.c0:.10g}\t"
            f"{item.experiment.t.size}\t{item.rmse:.10g}"
        )
    (output_dir / "global_fit_summary.dat").write_text("\n".join(summary) + "\n", encoding="utf-8")

    wavelength = result.experiments[0].experiment.wavelength
    np.savetxt(
        output_dir / "global_pure_spectra.dat",
        np.column_stack((wavelength, result.spectra)),
        delimiter="\t",
        header="\t".join(("wavelength", *result.species_labels)),
        comments="# ",
    )
    for item in result.experiments:
        stem = item.source_path.stem
        np.savetxt(
            output_dir / f"{stem}_global_concentrations.dat",
            np.column_stack((item.experiment.t, item.concentrations.T)),
            delimiter="\t",
            header="\t".join(("time", *result.species_labels)),
            comments="# ",
        )


def plot_global_fit(
    result: GlobalFitResult,
    path: Path,
    show: bool = False,
) -> None:
    """Plot experimental/reconstructed spectra and concentrations for all R."""
    import matplotlib.pyplot as plt

    n_experiments = len(result.experiments)
    fig, axes = plt.subplots(
        n_experiments,
        2,
        figsize=(15, 3.1 * n_experiments),
        constrained_layout=True,
        squeeze=False,
    )
    params = ", ".join(f"{name}={value:.4g}" for name, value in result.params.items())
    fig.suptitle(f"Global multi-R transsulfuration fit | {params}", fontsize=13)

    for row, item in enumerate(result.experiments):
        ax_spectra, ax_concentrations = axes[row]
        ax_spectra.plot(
            item.experiment.wavelength,
            item.experiment.absorbance,
            color="black",
            linewidth=0.45,
            alpha=0.28,
        )
        ax_spectra.plot(
            item.experiment.wavelength,
            item.absorbance_calc,
            color="red",
            linewidth=0.45,
            alpha=0.28,
        )
        ax_spectra.plot([], [], color="black", label="experimental")
        ax_spectra.plot([], [], color="red", label="global fit")
        ax_spectra.set_title(f"{item.name} | R={item.hss_ratio:g} | RMSE={item.rmse:.4g}")
        ax_spectra.set_xlabel("Wavelength (nm)")
        ax_spectra.set_ylabel("Absorbance")
        ax_spectra.legend(loc="best")

        for index, label in enumerate(result.species_labels):
            ax_concentrations.plot(
                item.experiment.t,
                item.concentrations[index],
                linewidth=1.8,
                label=label,
            )
        ax_concentrations.set_title(f"Concentration profiles | R={item.hss_ratio:g}")
        ax_concentrations.set_xlabel("Time after HSS- addition (s)")
        ax_concentrations.set_ylabel("Concentration (M)")
        ax_concentrations.legend(loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def global_error_for_params(
    result: GlobalFitResult,
    series: list[GlobalExperiment],
    params: dict[str, float],
) -> float:
    """Evaluate the same profiled global objective at explicit parameters."""
    known_spectra = _endpoint_spectra(
        series,
        result.fixed_initial_spectrum,
        result.fixed_final_spectrum,
    )
    profiles = _profiles_for_params(result.model, series, params)
    spectra = _fit_shared_spectra(series, profiles, known_spectra)
    return _global_error(series, profiles, spectra)


def plot_global_error_landscape(
    result: GlobalFitResult,
    series: list[GlobalExperiment],
    path: Path,
    profile_points: int = 25,
    surface_points: int = 17,
    span_log10: float = 1.5,
    show: bool = False,
) -> None:
    """Plot fixed-parameter profiles and a k_ts/k_fast objective slice.

    These are local slices through the profiled spectral objective: parameters
    not shown on an axis remain fixed at the reported optimum. They diagnose
    flat directions and correlations but are not confidence intervals.
    """
    import matplotlib.pyplot as plt

    names = tuple(result.params)
    n_profiles = len(names)
    fig = plt.figure(figsize=(15, 4.2 + 3.2 * ((n_profiles + 1) // 2)), constrained_layout=True)
    grid = fig.add_gridspec((n_profiles + 1) // 2, 3)
    base_error = result.error

    for index, name in enumerate(names):
        row = index // 2
        column = index % 2
        ax = fig.add_subplot(grid[row, column])
        optimum = result.params[name]
        values = np.logspace(
            np.log10(optimum) - span_log10,
            np.log10(optimum) + span_log10,
            profile_points,
        )
        errors = []
        for value in values:
            trial = dict(result.params)
            trial[name] = float(value)
            errors.append(global_error_for_params(result, series, trial))
        ax.semilogx(values, np.asarray(errors) - base_error, color="tab:blue")
        ax.axvline(optimum, color="black", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("RMSE - reported RMSE")
        suffix = " (externally fixed)" if name in result.fixed_parameters else ""
        ax.set_title(f"Fixed slice for {name}{suffix}")

    ax_surface = fig.add_subplot(grid[:, 2])
    k_ts_values = np.logspace(
        np.log10(result.params["k_ts"]) - span_log10,
        np.log10(result.params["k_ts"]) + span_log10,
        surface_points,
    )
    k_fast_values = np.logspace(
        np.log10(result.params["k_fast"]) - span_log10,
        np.log10(result.params["k_fast"]) + span_log10,
        surface_points,
    )
    surface = np.empty((surface_points, surface_points))
    for row, k_fast in enumerate(k_fast_values):
        for column, k_ts in enumerate(k_ts_values):
            trial = dict(result.params)
            trial["k_ts"] = float(k_ts)
            trial["k_fast"] = float(k_fast)
            surface[row, column] = global_error_for_params(result, series, trial)
    low = float(np.min(surface))
    high = float(np.max(surface))
    levels = np.linspace(low, high, 20)
    contour = ax_surface.contourf(
        k_ts_values,
        k_fast_values,
        surface,
        levels=levels,
    )
    fig.colorbar(contour, ax=ax_surface, label="Global RMSE")
    ax_surface.scatter(
        [result.params["k_ts"]],
        [result.params["k_fast"]],
        marker="x",
        s=70,
        color="red",
        label="reported optimum",
    )
    ax_surface.set_xscale("log")
    ax_surface.set_yscale("log")
    ax_surface.set_xlabel("k_ts (M^-1 s^-1)")
    ax_surface.set_ylabel("k_fast (s^-1)")
    ax_surface.set_title("k_ts-k_fast objective slice")
    ax_surface.legend(loc="best")
    fig.suptitle(
        "Global objective landscape (other parameters fixed at optimum)",
        fontsize=13,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def run_global_from_args(args: argparse.Namespace) -> GlobalFitResult:
    """Load, preprocess, fit, report and optionally export a manifest series."""
    manifest = Path(args.global_manifest)
    root = Path(args.global_root) if args.global_root else None
    records = read_global_manifest(manifest, root=root)
    series = [
        preprocess_global_experiment(
            source,
            ratio,
            c0,
            time_zero=args.global_time_zero,
            baseline_region=(args.baseline_lambda_min, args.baseline_lambda_max),
            fit_region=(args.lambda_min, args.lambda_max),
        )
        for source, ratio, c0 in records
    ]
    result = fit_global_transsulfuration(
        series,
        model=args.model,
        fix_initial_spectrum=args.fix_initial_spectrum,
        fix_final_spectrum=args.fix_final_spectrum,
        k_bounds=(args.k_min, args.k_max),
        max_starts=args.global_max_starts,
        fixed_parameters=(
            {"k_fast": args.global_fix_k_fast}
            if args.global_fix_k_fast is not None
            else None
        ),
    )

    print("Global multi-R transsulfuration fit")
    print(f"  model: {result.model}")
    print(f"  experiments: {len(result.experiments)}")
    print(f"  wavelength range: {args.lambda_min:g}-{args.lambda_max:g} nm")
    print(f"  baseline region: {args.baseline_lambda_min:g}-{args.baseline_lambda_max:g} nm")
    print(f"  chemical time zero: {args.global_time_zero:g} s")
    for name, value in result.params.items():
        suffix = (
            " (fixed)"
            if name == "k_fast" and args.global_fix_k_fast is not None
            else ""
        )
        print(f"  {name}: {value:.10g}{suffix}")
    print(f"  global RMSE: {result.error:.10g}")
    print("  Per-experiment RMSE:")
    for item in result.experiments:
        print(f"    {item.name}: R={item.hss_ratio:g}, RMSE={item.rmse:.10g}")

    if args.global_output_dir:
        output_dir = Path(args.global_output_dir)
        write_global_outputs(result, output_dir)
        plot_global_fit(result, output_dir / "global_fit_panel.png")
        plot_global_error_landscape(
            result,
            series,
            output_dir / "global_error_landscape.png",
        )
        print(f"  outputs: {output_dir}")
    return result
