#!/usr/bin/env python3
"""Output writers for final TolKinet fits."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path

import numpy as np

from kinet_common import (
    Experiment,
    FitResult,
    HSS_TRANSSULFURATION_MODELS,
    MODEL_LABELS,
    PARAMETER_LABELS,
)
from kinet_fitting import is_near_bound
from kinet_plotting import plot_result
from kinet_linalg import projected_concentrations, concentration_point_indices


def can_write_output_directory(path: Path) -> bool:
    """Return whether an output directory can be entered and written."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".tolkin_write_probe_{uuid.uuid4().hex}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def output_prefix_from_input(path: Path) -> Path:
    """Return the file prefix used for exported fit artifacts."""
    input_dir = path.parent if str(path.parent) else Path(".")
    output_dir = input_dir / f"results_{path.stem}"
    if can_write_output_directory(output_dir):
        return output_dir / path.stem

    for suffix in range(1, 100):
        fallback_dir = input_dir / f"results_{path.stem}_{suffix}"
        if can_write_output_directory(fallback_dir):
            print(
                "Default results directory is not writable: "
                f"{output_dir}. Writing outputs to {fallback_dir} instead."
            )
            return fallback_dir / path.stem

    raise PermissionError(
        f"Could not create a writable results directory next to {path}"
    )


def ensure_writable_output_path(path: Path) -> Path:
    """Return path or a numbered alternative when an existing file is not writable."""
    if not path.exists():
        return path
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except OSError:
        for suffix in range(1, 100):
            candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            if not candidate.exists():
                return candidate
            try:
                with candidate.open("a", encoding="utf-8"):
                    pass
                return candidate
            except OSError:
                continue
    raise PermissionError(f"Could not find a writable output path for {path}")



def write_fit_summary_dat(
    path: Path,
    source_input_path: Path,
    analysis_input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    reaction_end_time: float | None,
    model: str,
) -> None:
    """Write a tab-separated summary of the final accepted fit."""
    k_min, k_max = final_k_bounds
    lines = [
        "# key\tvalue",
        f"source_file\t{source_input_path}",
        f"analysis_file\t{analysis_input_path}",
        f"model\t{MODEL_LABELS[model]}",
        f"fit_method\t{result.method}",
        f"c0_M\t{c0:.10g}",
        f"n_wavelengths\t{experiment.wavelength.size}",
        f"n_spectra\t{experiment.t.size}",
        f"wavelength_min\t{experiment.wavelength[0]:.10g}",
        f"wavelength_max\t{experiment.wavelength[-1]:.10g}",
        f"time_min\t{experiment.t[0]:.10g}",
        f"time_max\t{experiment.t[-1]:.10g}",
        f"fit_error\t{result.error:.10g}",
        f"residual_RMSE_mAU\t{1000 * np.sqrt(np.mean(result.residuals**2)):.10g}",
        "residual_definition\texperimental - calculated",
        "residual_color_limit_mAU\t20",
        f"residual_max_abs_mAU\t{1000 * np.max(np.abs(result.residuals)):.10g}",
        f"residual_saturated_fraction\t{np.mean(np.abs(result.residuals) > 0.020):.10g}",
        f"minimum_recovered_spectrum_value\t{result.spectra.min():.10g}",
        f"k_search_min\t{k_min:.10g}",
        f"k_search_max\t{k_max:.10g}",
    ]
    if reaction_start_time is not None:
        lines.append(f"reaction_start_cutoff_original_units\t{reaction_start_time:.10g}")
    if hasattr(args, "reaction_time_origin"):
        lines.append(f"reaction_time_origin_original_units\t{args.reaction_time_origin:.17g}")
        lines.append("corrected_time_formula\tt_original - reaction_time_origin")
    if reaction_end_time is not None:
        lines.append(f"reaction_end_time_original_units\t{reaction_end_time:.10g}")
    if model in HSS_TRANSSULFURATION_MODELS:
        lines.append(f"hss_ratio_added\t{args.hss_ratio:.10g}")
    if args.initial_spectrum_weight > 0:
        lines.append(f"initial_spectrum_weight\t{args.initial_spectrum_weight:.10g}")
    if result.fixed_initial_spectrum:
        lines.append(f"fixed_initial_spectrum\t{result.species_labels[0]}")
    if result.fixed_final_spectrum:
        lines.append(f"fixed_final_spectrum\t{result.species_labels[-1]}")
    for endpoint in ("initial", "final"):
        indices = getattr(args, f"{endpoint}_reference_indices", None)
        if indices is not None:
            lines.append(f"fixed_{endpoint}_reference_original_indices_1based\t"
                         + ",".join(str(i+1) for i in indices))
            lines.append(f"fixed_{endpoint}_reference_aggregation\tarithmetic_mean")
    for reference in getattr(args, "fixed_reference_files", []):
        label = reference['species']
        prefix = "adjustable" if reference['fit_amplitude'] else "fixed"
        lines.extend([f"{prefix}_reference_file_{label}\t{reference['path']}",
                      f"{prefix}_reference_sha256_{label}\t{reference['sha256']}",
                      f"{prefix}_reference_units_{label}\t{reference['units']}",
                      f"reference_amplitude_mode_{label}\t{prefix}"])
        if reference['fit_amplitude']:
            lo,hi = reference['amplitude_bounds']
            lines.extend([f"reference_amplitude_min_{label}\t{lo:.10g}",
                          f"reference_amplitude_max_{label}\t{hi:.10g}"])
        else:
            lines.append(f"reference_amplitude_multiplier_{label}\t1")
    for label, indices in getattr(args, "intermediate_reference_indices", {}).items():
        lines.append(f"fixed_reference_original_indices_{label}\t"
                     + ",".join(str(i+1) for i in indices))
    if result.known_species:
        lines.append("known_spectral_shapes\t" + ",".join(result.known_species))
    if result.known_spectrum_scales:
        for label, scale in result.known_spectrum_scales.items():
            lines.append(f"known_spectrum_scale_{label}\t{scale:.10g}")
    if model == "mbfe3_sulfide_irreversible_binding_autocatalytic":
        lines.append("fixed_k_off\t0")
    for name, value in result.params.items():
        lines.append(f"{PARAMETER_LABELS.get(name, name)}\t{value:.10g}")
    if set(result.params) == {"k"}:
        fitted_k = next(iter(result.params.values()))
        lines.append(f"half_life\t{np.log(2) / fitted_k:.10g}")
    lines.append("species\t" + ",".join(result.species_labels))
    lines.append(
        "singular_values\t"
        + ",".join(f"{value:.10g}" for value in result.singular_values)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def export_final_fit_outputs(
    source_input_path: Path,
    analysis_input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    reaction_end_time: float | None,
    model: str,
    protected_input_paths: list[Path] | None = None,
) -> None:
    """Save final fit outputs for external plotting and reporting."""
    prefix = output_prefix_from_input(source_input_path)
    concentration_path = ensure_writable_output_path(
        prefix.with_name(f"{prefix.name}_concentrations.dat")
    )
    spectra_path = ensure_writable_output_path(
        prefix.with_name(f"{prefix.name}_pure_spectra.dat")
    )
    summary_path = ensure_writable_output_path(
        prefix.with_name(f"{prefix.name}_fit_summary.dat")
    )
    panel_path = ensure_writable_output_path(
        prefix.with_name(f"{prefix.name}_fit_panel.png")
    )
    protected_resolved = {
        path.resolve()
        for path in (protected_input_paths or [])
        if path.exists()
    }
    if spectra_path.resolve() in protected_resolved:
        spectra_path = ensure_writable_output_path(
            prefix.with_name(f"{prefix.name}_fit_pure_spectra.dat")
        )
        print(
            "Known-spectrum input matches the default pure-spectra output; "
            f"writing recovered spectra to {spectra_path} instead."
        )

    def numeric_path(suffix):
        candidate = prefix.with_name(f"{prefix.name}_{suffix}")
        while candidate.resolve() in protected_resolved:
            candidate = candidate.with_name(f"{candidate.stem}_fit{candidate.suffix}")
        return ensure_writable_output_path(candidate)

    def write_numeric(path, table):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t" if path.suffix == ".dat" else ",")
            for row in table:
                writer.writerow(["" if np.isnan(value) else format(value, ".12g")
                                 for value in row])

    def spectral_matrix(values):
        table = np.empty((experiment.wavelength.size + 1, experiment.t.size + 1))
        table[0, 0] = np.nan
        table[0, 1:] = experiment.t
        table[1:, 0] = experiment.wavelength
        table[1:, 1:] = values
        return table

    experiment_path = numeric_path("experimental.csv")
    simulated_path = numeric_path("simulated.csv")
    residual_path = numeric_path("residuals_mAU.csv")
    write_numeric(experiment_path, spectral_matrix(experiment.absorbance))
    write_numeric(simulated_path, spectral_matrix(result.absorbance_calc))
    write_numeric(residual_path, spectral_matrix(1000. * result.residuals))
    write_numeric(spectra_path, np.column_stack((experiment.wavelength, result.spectra)))

    estimates = projected_concentrations(experiment, result, c0)
    point_ids = concentration_point_indices(experiment.t)
    count = len(result.species_labels)
    concentration_table = np.full((1 + 2 * count, experiment.t.size + 1), np.nan)
    concentration_table[0, 1:] = experiment.t
    concentration_table[1:, 0] = np.arange(1, 2 * count + 1)
    concentration_table[1:count + 1, 1:] = result.c
    concentration_table[count + 1:, point_ids + 1] = estimates[:, point_ids]
    write_numeric(concentration_path, concentration_table)
    estimate_path = numeric_path("spectral_concentrations.dat")
    estimate_table = np.vstack((experiment.t[point_ids], estimates[:, point_ids]))
    write_numeric(estimate_path, estimate_table)

    layout_path = numeric_path("export_layout.json")
    layout_path.write_text(json.dumps({
        "species_order": list(result.species_labels),
        "spectral_matrices": {
            "files": [experiment_path.name, simulated_path.name, residual_path.name],
            "first_row": "time, same units and origin as the fitted experiment",
            "first_column": "wavelength (nm)", "top_left": "empty",
            "absorbance_units": "AU", "residual_units": "mAU",
            "residual_definition": "experimental - simulated",
        },
        "pure_spectra": {"file": spectra_path.name,
            "first_column": "wavelength (nm)",
            "remaining_columns": list(result.species_labels), "units": "M^-1 cm^-1"},
        "concentrations": {"file": concentration_path.name,
            "first_row": "time", "first_column": "numeric row identifier",
            "model_row_ids": list(range(1, count + 1)),
            "projected_row_ids": list(range(count + 1, 2 * count + 1)),
            "species_order_within_each_block": list(result.species_labels),
            "units": "M", "empty_cells": "projected point not selected for display",
            "projection": "nonnegative, sum constrained to c0; diagnostic using fitted spectra",
            "point_selection": "nearest measured times to 0/5/15/30 and ten equally spaced targets"},
        "projected_points": {"file": estimate_path.name, "first_row": "time",
            "remaining_rows": list(result.species_labels), "units": "M"},
    }, indent=2) + "\n", encoding="utf-8")

    write_fit_summary_dat(
        summary_path,
        source_input_path,
        analysis_input_path,
        experiment,
        result,
        final_k_bounds,
        args,
        c0,
        reaction_start_time,
        reaction_end_time,
        model,
    )
    if result.selection_diagnostics is not None:
        selection_path=ensure_writable_output_path(prefix.with_name(f"{prefix.name}_selection.json"))
        selection_path.write_text(json.dumps(result.selection_diagnostics,indent=2)+"\n")
        print(f"  Selection diagnostics: {selection_path}")
    plot_result(
        experiment,
        result,
        input_filename=source_input_path.name,
        save_path=panel_path,
        show=False,
    )

    print("Exported files:")
    print(f"  {concentration_path}")
    print(f"  {spectra_path}")
    print(f"  {summary_path}")
    print(f"  {experiment_path}")
    print(f"  {simulated_path}")
    print(f"  {residual_path}")
    print(f"  {estimate_path}")
    print(f"  {layout_path}")
    print(f"  {panel_path}")
