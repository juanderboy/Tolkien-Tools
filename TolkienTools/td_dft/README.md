# TD-DFT NEA Spectra

Nuclear Ensemble Approximation (NEA) absorption spectra from ORCA TD-DFT
calculations. The launcher entry point is `td_analyze.py`:

```bash
tolkien-tools
tolkien-tools td --help
tolkien-tools td --file calculation.out --export
tolkien-tools td . --mode energy --no-html
```

Use `--file FILE` to explicitly process one ORCA output (any filename), without
snapshot averaging. The positional form `tolkien-tools td calculation.out` is
also supported. Do not combine `--file` with a positional target.

From the main interactive menu, choose **2. TD-DFT spectra**, then
**1. Single calculation** (or **2. Nuclear ensemble**). The prompts ask for the
input, x0/x1 wavelength limits in nm (defaults 200–1000), lambda/energy plot axis
(default lambda), Gaussian FWHM (default 0.1 eV), refractive index (default 1.33),
and window display. Press Enter to accept each default.
`tolkien-tools td --interactive` opens the same prompts directly.

Single-file mode saves `electronic_spectrum/spectrum.png` and
`electronic_spectrum/spectrum.dat` under the current working directory.
The table always contains two columns: wavelength in nm (ascending) and epsilon
in L mol^-1 cm^-1, including when the plot uses an energy axis. Existing files
with these names are replaced on subsequent runs. The PNG includes transition
sticks and peak labels. `--nosave` suppresses the PNG; the table is always saved
(`--export` remains accepted for compatibility).
Use `--mode energy` for an energy axis and `--linewidth_ev` to set the Gaussian
FWHM. `--startx` and `--endx` set wavelength limits in either representation.
HTML reports are available only in folder mode. The parser currently reads ORCA
absorption tables.

Pass a directory (or omit the target for the current directory) to process an
ensemble of `TD_*.out` files. Folder
processing supports file selection/exclusion, individual and ensemble spectra,
peak lists, and interactive HTML output. Run `--help` for all options.

## Internal organization

- `td_analyze.py`: compatible entry point used by the launcher.
- `td_cli.py`: command-line arguments and file/folder dispatch.
- `td_common.py`: physical constants, ranges, selections and HTML opening.
- `td_orca.py`: parsing of ORCA TD-DFT outputs.
- `td_spectrum.py`: energy grid, Gaussian broadening and NEA calculation.
- `td_single.py`: processing of a single `TD_*.out` file.
- `td_batch.py`: processing of directories with many `TD_*.out` files.

## Canonical DFT orbital transition explorer

Choose **3. Explore DFT orbital transitions** in the interactive TD-DFT panel,
provide an ORCA output, and select its original `STATE` number. The explorer
lists wavelength, oscillator strength, multiplicity label and printed orbital
pairs. Contributions are sorted by ORCA's printed weight, preserving signed
coefficients and alpha/beta indices. Choose `1` for the largest contribution,
`1,2` for several, or `all` for all printed contributions. ORCA's print threshold
means this is not necessarily the full expansion.

```bash
tolkien-tools td transitions calculation.out --list
tolkien-tools td transitions calculation.out
tolkien-tools td transitions calculation.out --state 6 --pairs 1,2 \
  --gbw calculation.gbw --grid 60
```

Listing needs only the output file. Generating cubes needs the corresponding
canonical DFT `.gbw` and `orca_plot` on PATH (or `--orca-plot /path/to/orca_plot`).
Interactive mode asks for both and the grid resolution. The GBW is copied into
a temporary workspace; an XYZ companion is built from the output geometry.
Original calculation files are not modified. Cubes are checked for completeness,
geometry agreement and common grid. Use the ORCA installation matching the GBW.
Matching geometry alone cannot prove that an arbitrary GBW contains the same
DFT orbitals: provide the GBW from the actual calculation.

Results go into `electronic_spectrum/transitions/state_NNN_multM/`, or under
`--output-dir`. Repeated exports receive numbered directories. Each contains
unique orbital cubes (shared orbitals are generated once), `transition.json`
with source paths and selected pairs, per-orbital ORCA logs, and `view.vmd`.
Open with `vmd -e /path/to/view.vmd`. Orbitals are separate VMD molecules; toggle
them in the molecule list. Blue/red isosurfaces show the two orbital signs at
+/-0.03; these are DFT orbital amplitudes, not excitation densities or NTOs.

The first implementation targets ORCA 6 spin-free TD-DFT/TDA output. When SOC
results are also present, only the uncorrected electric-dipole table is used;
SOC-mixed states are explicitly excluded. Original STATE numbers can differ
from local absorption labels (e.g. STATE 31 versus 1-3A). The reader matches
multiplicity and energy instead of assuming a table-row correspondence.
Concatenated/ambiguous calculations are rejected. `--min-fosc` filters the
displayed table without renumbering states. No new electronic-structure
calculation or cluster submission is performed.

For UHF/UKS references, the explorer preserves alpha/beta channels and warns
that ORCA multiplicities are estimates based on the printed `<S**2>`. They
should not be interpreted as pure-spin assignments. Printed `<S**2>` is also
recorded in the transition metadata.
