#!/usr/bin/env python3
"""Shared data structures and labels for TolKinet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GENERAL_MODEL_LABELS = {
    "a_to_b": "A -> B irreversible first-order reaction",
    "a_to_b_to_c": "A -> B -> C consecutive irreversible reactions",
    "a_rev_b_to_c": "A <-> B -> C",
}

SPECIAL_MODEL_LABELS = {
    "mbfe3_hss_no_binding": (
        "MbFe(III) reduction by HSS- without binding in the fitted window"
    ),
    "mbfe3_sulfide_autocatalytic": "Autocatalytic MbFe(III) reduction by sulfides",
    "mbfe3_sulfide_binding_autocatalytic": (
        "Autocatalytic MbFe(III) reduction by sulfides with reversible binding"
    ),
    "mbfe3_sulfide_hss_transsulfuration": (
        "MbFe(III)-HS reduction by added HSS- with transsulfuration"
    ),
    "mbfe3_sulfide_hss_transsulfuration_no_auto": (
        "MbFe(III)-HS reduction by added HSS- with transsulfuration "
        "without autocatalysis"
    ),
    "mbfe3_sulfide_irreversible_binding_autocatalytic": (
        "Autocatalytic MbFe(III) reduction by sulfides with irreversible binding"
    ),
    "femb_disulfide_reversible": "FeMb + disulfides with reversible binding",
}

MODEL_LABELS = {
    **GENERAL_MODEL_LABELS,
    **SPECIAL_MODEL_LABELS,
}


MODEL_SPECIES = {
    "femb_disulfide_reversible": ("MbFeIII", "MbFeIII-S2", "MbFeII"),
    "a_to_b": ("A", "B"),
    "a_to_b_to_c": ("A", "B", "C"),
    "a_rev_b_to_c": ("A", "B", "C"),
    "mbfe3_hss_no_binding": ("MbFeIII-HSS", "MbFeII"),
    "mbfe3_sulfide_autocatalytic": ("MbFeIII-SH", "MbFeII"),
    "mbfe3_sulfide_binding_autocatalytic": ("MbFeIII", "MbFeIII-HS", "MbFeII"),
    "mbfe3_sulfide_irreversible_binding_autocatalytic": ("MbFeIII", "MbFeIII-HS", "MbFeII"),
    "mbfe3_sulfide_hss_transsulfuration": ("MbFeIII-Sx", "MbFeII"),
    "mbfe3_sulfide_hss_transsulfuration_no_auto": ("MbFeIII-Sx", "MbFeII"),
}

HSS_TRANSSULFURATION_MODELS = (
    "mbfe3_sulfide_hss_transsulfuration",
    "mbfe3_sulfide_hss_transsulfuration_no_auto",
)


PARAMETER_LABELS = {
    "k": "k",
    "k1": "k1",
    "k_1": "k-1",
    "k2": "k2",
    "k_on": "k_on",
    "k_off": "k_off",
    "k_slow": "k_slow,obs",
    "k_auto": "k_auto",
    "k_cat": "k_cat",
    "k_ts": "k_ts",
    "k_fast": "k_fast",
}


MODEL_PRESENTATIONS = {
    "a_to_b": {
        "scheme": "A -> B",
        "profiles": (
            "[A](t) = c0 * exp(-k * t)",
            "[B](t) = c0 - [A](t)",
        ),
        "parameters": (
            "k: first-order rate constant",
        ),
        "notes": (
            "The pure spectra of A and B are fitted simultaneously.",
        ),
    },
    "a_to_b_to_c": {
        "scheme": "A -> B -> C",
        "profiles": (
            "[A](t) = c0 * exp(-k1 * t)",
            "[B](t) = c0 * k1/(k2-k1) * (exp(-k1*t) - exp(-k2*t))",
            "[C](t) = c0 - [A](t) - [B](t)",
        ),
        "parameters": (
            "k1: rate constant for A -> B",
            "k2: rate constant for B -> C",
        ),
        "notes": (
            "If k1 and k2 coincide, the corresponding analytical limit is used.",
        ),
    },
    "a_rev_b_to_c": {
        "scheme": "A <-> B -> C",
        "profiles": (
            "d[A]/dt = -k1*[A] + k-1*[B]",
            "d[B]/dt = k1*[A] - (k-1+k2)*[B]",
            "d[C]/dt = k2*[B]",
        ),
        "parameters": (
            "k1: rate constant for A -> B",
            "k-1: rate constant for B -> A",
            "k2: rate constant for B -> C",
        ),
        "notes": (
            "Time profiles are calculated by solving the linear system.",
        ),
    },
    "mbfe3_hss_no_binding": {
        "scheme": "MbFeIII-HSS -> MbFeII",
        "profiles": (
            "[MbFeIII-HSS](t) = c0 * exp(-k * t)",
            "[MbFeII](t) = c0 - [MbFeIII-HSS](t)",
        ),
        "parameters": (
            "k: apparent first-order rate constant for MbFeIII-HSS -> MbFeII",
        ),
        "notes": (
            "Fast HSS- binding lies outside the fitted window.",
            (
                "Time zero is estimated by fitting the fast decay "
                "of MbFeIII at 409 nm; 428 and 434 nm are shown as controls."
            ),
            "No autocatalytic process is included.",
        ),
    },
    "mbfe3_sulfide_autocatalytic": {
        "scheme": "MbFeIII-SH -> MbFeII, with apparent autocatalytic acceleration",
        "profiles": (
            "x(t) = [MbFeII](t) / [Mb]total",
            "dx/dt = (k_slow + k_auto*x) * (1 - x)",
            "[MbFeIII-SH](t) = c0 * (1 - x)",
            "[MbFeII](t) = c0 * x",
        ),
        "parameters": (
            "k_slow,obs: apparent rate constant of the initial slow phase",
            "k_auto: apparent phenomenological acceleration",
        ),
        "notes": (
            "The fit uses all times and two absorbing species.",
            "x is a proxy for unobserved reactive sulfur species.",
            "The dependence of k_slow,obs on [HS-] must be determined experimentally.",
            "k_auto must not be interpreted directly as k_Red(HSS-).",
        ),
    },
    "mbfe3_sulfide_binding_autocatalytic": {
        "scheme": (
            "MbFeIII + HS- <-> MbFeIII-HS -> MbFeII, "
            "with apparent autocatalytic acceleration"
        ),
        "profiles": (
            (
                "d[MbFeIII]/dt = -k_on*[MbFeIII] "
                "+ k_off*[MbFeIII-HS]"
            ),
            "x(t) = [MbFeII](t) / [Mb]total",
            (
                "d[MbFeIII-HS]/dt = k_on*[MbFeIII] "
                "- (k_off + k_slow + k_cat*x)*[MbFeIII-HS]"
            ),
            "d[MbFeII]/dt = (k_slow + k_cat*x)*[MbFeIII-HS]",
        ),
        "parameters": (
            (
                "k_on: apparent HS- coordination rate constant "
                "under the experimental conditions"
            ),
            "k_off: MbFeIII-HS dissociation rate constant",
            "k_slow,obs: apparent slow reduction rate constant of MbFeIII-HS",
            "k_cat: apparent phenomenological autocatalytic acceleration",
        ),
        "notes": (
            "The fit uses three absorbing species: MbFeIII, MbFeIII-HS and MbFeII.",
            (
                "If [HS-] remains in excess, k_on corresponds to the bimolecular k_on "
                "multiplied by [HS-]."
            ),
            "x is a proxy for unobserved reactive sulfur species.",
            "k_cat must not be interpreted directly as k_Red(HSS-).",
        ),
    },
    "mbfe3_sulfide_hss_transsulfuration": {
        "scheme": (
            "MbFeIII-HS + HSS- -> MbFeIII-HSS -> MbFeII, "
            "alongside autocatalytic reduction by sulfides"
        ),
        "profiles": (
            "A = [MbFeIII-HS]",
            "B = [MbFeIII-HSS]",
            "S = effective free added [HSS-]",
            "x(t) = [MbFeII](t) / [Mb]total",
            "dA/dt = -(k_slow + k_auto*x)*A - k_ts*A*S",
            "dB/dt = k_ts*A*S - k_fast*B",
            "d[MbFeII]/dt = (k_slow + k_auto*x)*A + k_fast*B",
            "dS/dt = -k_ts*A*S",
            "[MbFeIII-Sx](t) = A + B",
        ),
        "parameters": (
            "k_slow,obs: apparent slow reduction rate constant with HS-",
            "k_auto: phenomenological acceleration by endogenous polysulfides",
            "k_ts: bimolecular transsulfuration by added HSS-",
            "k_fast: fast reduction of the MbFeIII-HSS intermediate",
        ),
        "notes": (
            "The fit uses two absorbing species: MbFeIII-Sx and MbFeII.",
            "MbFeIII-HS and MbFeIII-HSS are kinetically distinct but share a spectrum.",
            "The initial added HSS- concentration is fixed at R_HSS*c0.",
            "Endogenous polysulfide-like species production is represented by k_auto*x.",
        ),
    },
    "mbfe3_sulfide_hss_transsulfuration_no_auto": {
        "scheme": (
            "MbFeIII-HS + HSS- -> MbFeIII-HSS -> MbFeII, "
            "in parallel with slow MbFeIII-HS reduction"
        ),
        "profiles": (
            "A = [MbFeIII-HS]",
            "B = [MbFeIII-HSS]",
            "S = effective free added [HSS-]",
            "dA/dt = -k_slow*A - k_ts*A*S",
            "dB/dt = k_ts*A*S - k_fast*B",
            "d[MbFeII]/dt = k_slow*A + k_fast*B",
            "dS/dt = -k_ts*A*S",
            "[MbFeIII-Sx](t) = A + B",
        ),
        "parameters": (
            "k_slow,obs: apparent slow MbFeIII-HS reduction rate constant",
            "k_ts: bimolecular transsulfuration by added HSS-",
            "k_fast: fast reduction of the MbFeIII-HSS intermediate",
        ),
        "notes": (
            "The fit uses two absorbing species: MbFeIII-Sx and MbFeII.",
            "MbFeIII-HS and MbFeIII-HSS are kinetically distinct but share a spectrum.",
            "The initial added HSS- concentration is fixed at R_HSS*c0.",
            "No autocatalytic acceleration by endogenous polysulfides is included.",
        ),
    },
}


MODEL_PRESENTATIONS["mbfe3_sulfide_irreversible_binding_autocatalytic"] = {
    "scheme": "MbFeIII + HS- -> MbFeIII-HS -> MbFeII, with autocatalytic acceleration",
    "profiles": (
        "d[MbFeIII]/dt = -k_on*[MbFeIII]",
        "x(t) = [MbFeII](t) / [Mb]total",
        "d[MbFeIII-HS]/dt = k_on*[MbFeIII] - (k_slow + k_cat*x)*[MbFeIII-HS]",
        "d[MbFeII]/dt = (k_slow + k_cat*x)*[MbFeIII-HS]",
    ),
    "parameters": (
        "k_on: apparent pseudo-first-order HS- coordination rate constant",
        "k_slow,obs: apparent slow reduction rate constant",
        "k_cat: apparent phenomenological autocatalytic acceleration",
    ),
    "notes": (
        "k_off = 0 is fixed, not fitted.",
        "Uses the reversible binding integrator with dissociation disabled.",
        "k_cat must not be interpreted directly as k_Red(HSS-).",
    ),
}


MODEL_PRESENTATIONS["femb_disulfide_reversible"] = {
    "scheme": "A <-> B -> C (FeMb + disulfides), no autocatalysis",
    "profiles": ("A(0)=1; B(0)=C(0)=0; nonnegative species spectra",
                 "Explore equivalent reversible eigenvalue families"),
    "parameters": ("k_on: pseudo-first-order binding", "k_off: dissociation",
                   "k_slow: reduction (k_red)"),
    "notes": ("Select concentration-profile agreement within 0.1% of best spectral RMSE.",
              "Peak windows: A 405-414, B 423-430, C 432-438 nm.",
              "Intermediate/product Soret peak epsilon ratio: 0.50-0.95.",
              "B409/max is diagnostic only; the shoulder shape is not constrained.",
              "Uses documented FeMb rate bounds; automatic generic k-max expansion is disabled."),
}



@dataclass
class Experiment:
    """Datos experimentales en la convencion lambda x tiempo."""

    t: np.ndarray
    wavelength: np.ndarray
    absorbance: np.ndarray



@dataclass
class FitResult:
    """Resultado completo del ajuste y matrices utiles para diagnostico."""

    method: str
    model: str
    params: dict[str, float]
    species_labels: tuple[str, ...]
    c: np.ndarray
    spectra: np.ndarray
    absorbance_calc: np.ndarray
    residuals: np.ndarray
    singular_values: np.ndarray
    q: np.ndarray
    w: np.ndarray
    error: float
    known_species: tuple[str, ...] = ()
    known_spectrum_scales: dict[str, float] | None = None
    fixed_initial_spectrum: bool = False
    fixed_final_spectrum: bool = False
    selection_diagnostics: dict | None = None
    estimated_concentrations: np.ndarray | None = None
