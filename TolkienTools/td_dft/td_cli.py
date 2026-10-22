#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for TD-DFT spectrum analysis.

It defines terminal options and dispatches processing to either the single-file
or folder workflow.
"""

import argparse
import os
import sys

from td_batch import process_folder
from td_single import process_single_file


def main():
    parser = argparse.ArgumentParser(
        prog="tolkien-tools td",
        description=(
            "Generate molar absorptivity spectra from ORCA TD-DFT outputs.\n"
            "Single file: tolkien-tools td TD_10.out [options]\n"
            "Folder: tolkien-tools td [directory] [options]\n"
            "The current directory is used when no target is given."
        ),
        epilog=(
            "Examples:\n"
            "  tolkien-tools td TD_10.out --export\n"
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
        default=".",
        help="ORCA output (TD_*.out) or directory containing TD_*.out (default: current directory).",
    )

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
        help="(Single-file mode) Export epsilon as espectro_*.dat (lambda) or espectroE_*.dat (energy).",
    )

    args = parser.parse_args()

    # --- defaults "suaves" ---
    # Si no se dio -x0/-x1, usar 200–1000 nm
    if args.startx is None:
        args.startx = 200.0
    if args.endx is None:
        args.endx = 1000.0

    # Si el usuario no puso -s/--show, activamos show por defecto
    if not any(a in ("-s", "--show") for a in sys.argv[1:]):
        args.show = True

    if args.linewidth_ev <= 0:
        print("Line width (--linewidth_ev) must be > 0.")
        sys.exit(1)

    target = args.filename

    if os.path.isdir(target):
        process_folder(target, args)
    else:
        process_single_file(target, args)


if __name__ == "__main__":
    main()
