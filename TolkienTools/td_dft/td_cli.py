#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for TD-DFT spectrum analysis.

It defines terminal options and dispatches processing to either the single-file
or folder workflow.
"""

import argparse
import math
import os
import sys

from td_batch import process_folder
from td_single import process_single_file


def prompt_positive_number(label, default):
    while True:
        value = input(f"{label} [{default}]: ").strip()
        try:
            number = float(value) if value else default
            if math.isfinite(number) and number > 0:
                return number
        except ValueError:
            pass
        print("Enter a finite number greater than zero.")


def prompt_spectrum_options(args):
    print("\nTD-DFT spectra\n  1. Single calculation\n  2. Nuclear ensemble (folder)\n  3. Explore DFT orbital transitions")
    while True:
        choice = input("Choose mode [1]: ").strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        print("Choose 1, 2 or 3.")
    if choice == "3":
        from td_transitions import main as transitions_main
        while True:
            target = os.path.expanduser(input("ORCA output file: ").strip())
            if os.path.isfile(target):
                raise SystemExit(transitions_main([target]))
            print("Enter an existing output file.")
    if choice == "1":
        while True:
            target = os.path.expanduser(input("ORCA output file: ").strip())
            if os.path.isfile(target):
                args.single_file = target
                break
            print("Enter an existing output file.")
    else:
        while True:
            target = os.path.expanduser(input("Ensemble directory [.]: ").strip() or ".")
            if os.path.isdir(target):
                args.filename = target
                break
            print("Enter an existing directory.")
    args.startx = prompt_positive_number("x0: wavelength lower bound (nm)", 200.0)
    while True:
        args.endx = prompt_positive_number("x1: wavelength upper bound (nm)", 1000.0)
        if args.endx > args.startx:
            break
        print("x1 must be greater than x0.")
    while True:
        mode = input("Plot axis: lambda / energy [lambda]: ").strip().lower() or "lambda"
        if mode in {"lambda", "energy"}:
            args.mode = mode
            break
        print("Choose lambda or energy.")
    args.linewidth_ev = prompt_positive_number("Gaussian FWHM (eV)", 0.1)
    args.nref = prompt_positive_number("Refractive index", 1.33)
    while True:
        show = input("Show Matplotlib window? [Y/n]: ").strip().lower()
        if show in {"", "y", "yes", "s", "si", "n", "no"}:
            args.show = show not in {"n", "no"}
            break
        print("Choose yes or no.")
    if choice == "1":
        print("Outputs: electronic_spectrum/spectrum.png and spectrum.dat (wavelength, epsilon).")
        print("Existing outputs with these names will be replaced.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "transitions":
        from td_transitions import main as transitions_main
        raise SystemExit(transitions_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(
        prog="tolkien-tools td",
        description=(
            "Generate molar absorptivity spectra from ORCA TD-DFT outputs.\n"
            "Single file: tolkien-tools td --file calculation.out [options]\n"
            "Folder: tolkien-tools td [directory] [options]\n"
            "The current directory is used when no target is given."
        ),
        epilog=(
            "Examples:\n"
            "  tolkien-tools td --file calculation.out --export\n"
            "  tolkien-tools td . --exclude 10,12,15 --maxlist\n"
            "  tolkien-tools td . --mode energy --no-html\n"
            "Default wavelength range: 200-1000 nm. Matplotlib display is enabled."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # filename o carpeta (opcional, default='.')
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="ORCA output (any filename) or directory containing TD_*.out (default: current directory).",
    )

    parser.add_argument(
        "--file",
        dest="single_file",
        metavar="FILE",
        help="Process exactly one ORCA output, with any filename, without ensemble averaging.",
    )

    parser.add_argument("--interactive", action="store_true", help="Ask for input mode and spectrum settings interactively.")

    # modo de representación
    parser.add_argument(
        "--mode",
        choices=["lambda", "energy"],
        default="lambda",
        help="Spectrum representation: 'lambda' (ε(λ), default) or 'energy' (ε(E)).",
    )

    # mostrar ventana
    parser.add_argument(
        "-s",
        "--show",
        default=False,
        action="store_true",
        help="Show the Matplotlib window.",
    )

    # no guardar png
    parser.add_argument(
        "-n",
        "--nosave",
        default=False,
        action="store_true",
        help="Do not save generated PNG files.",
    )

    # ancho de línea (FWHM) en eV
    parser.add_argument(
        "-wev",
        "--linewidth_ev",
        type=float,
        default=0.1,
        help="Line width (FWHM) in eV for Gaussian convolution (default: 0.1 eV).",
    )

    # rango en λ (nm)
    parser.add_argument(
        "-x0",
        "--startx",
        type=float,
        help="Lower x-axis bound (λ in nm; converted to E in 'energy' mode).",
    )
    parser.add_argument(
        "-x1",
        "--endx",
        type=float,
        help="Upper x-axis bound (λ in nm; converted to E in 'energy' mode).",
    )

    # índice de refracción
    parser.add_argument(
        "--nref",
        type=float,
        default=1.33,   # agua en el visible
        help="Refractive index of the medium (default: 1.33, water in the visible range).",
    )

    # excluir algunos TD_* por número (modo carpeta)
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="TD_*.out numbers to exclude, e.g. '10', '10,12,15' or '10-20' (folder mode only).",
    )
    # selección explícita por índice (modo carpeta)
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Select indices 1..N or ranges, e.g. '1-15' or '1,2,3,7-10' (folder mode only).",
    )
    # no marcar picos en el espectro final (modo carpeta)
    parser.add_argument(
        "--no-final-peaks",
        default=False,
        action="store_true",
        help="Do not mark/label peaks in the final cumulative spectrum (folder mode).",
    )
    # imprimir PNGs individuales en modo carpeta
    parser.add_argument(
        "--printall",
        default=False,
        action="store_true",
        help="(Folder mode) Save individual TD_*.out-nea-eps.png files.",
    )

    # lista de máximos por espectro (modo carpeta)
    parser.add_argument(
        "--maxlist",
        default=False,
        action="store_true",
        help="(Folder mode) Generate a CSV listing the peak position of each spectrum.",
    )
    parser.add_argument(
        "--maxrange",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="(Folder mode) Peak search x range (nm in lambda mode, eV in energy mode).",
    )
    parser.add_argument(
        "--maxeps",
        type=float,
        help="(Folder mode) Minimum epsilon threshold for listing peaks.",
    )
    parser.add_argument(
        "--maxonly",
        default=None,
        action="store_true",
        help="(Folder mode) List only the absolute maximum within the range.",
    )
    parser.add_argument(
        "--allpeaks",
        default=None,
        action="store_true",
        help="(Folder mode) List all local maxima within the range.",
    )

    # HTML interactivo (plotly) en modo carpeta: activado por defecto
    html_group = parser.add_mutually_exclusive_group()
    html_group.add_argument(
        "--html",
        dest="html",
        action="store_true",
        help="(Folder mode) Generate interactive HTML (default: enabled).",
    )
    html_group.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        help="(Folder mode) Do not generate interactive HTML.",
    )
    parser.set_defaults(html=True)

    # exportar datos en modo archivo único
    parser.add_argument(
        "-e",
        "--export",
        default=False,
        action="store_true",
        help="Compatibility option; single-file mode always exports electronic_spectrum/spectrum.dat.",
    )

    args = parser.parse_args()

    if args.interactive:
        if args.single_file is not None or args.filename is not None:
            parser.error("--interactive cannot be combined with an input target.")
        prompt_spectrum_options(args)

    # --- defaults "suaves" ---
    # Si no se dio -x0/-x1, usar 200–1000 nm
    if args.startx is None:
        args.startx = 200.0
    if args.endx is None:
        args.endx = 1000.0

    # Si el usuario no puso -s/--show, activamos show por defecto
    if not args.interactive and not any(a in ("-s", "--show") for a in sys.argv[1:]):
        args.show = True

    if args.linewidth_ev <= 0:
        print("Line width (--linewidth_ev) must be > 0.")
        sys.exit(1)

    if args.single_file is not None and args.filename is not None:
        parser.error("Use either --file FILE or a positional target, not both.")

    target = args.single_file if args.single_file is not None else args.filename
    if target is None:
        target = "."
    if args.single_file is not None and not os.path.isfile(target):
        parser.error(f"--file requires an existing regular file: {target}")
    if not os.path.exists(target):
        parser.error(f"Target does not exist: {target}")

    if os.path.isdir(target):
        process_folder(target, args)
    else:
        process_single_file(target, args)


if __name__ == "__main__":
    main()
