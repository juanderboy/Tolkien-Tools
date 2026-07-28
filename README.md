# Tolkien Tools

Tolkien Tools is an interactive command-line toolbox for computational
chemistry and molecular-dynamics analysis.

Its main modules cover:

- processing fragmented MD/QMMM trajectories;
- TD-DFT absorption spectra from ORCA calculations;
- charge and spin population analysis;
- multiwavelength kinetic fitting.

## Usage

The main entry point is:

```bash
./tolkien-tools
```

Modules can also be selected directly:

```bash
./tolkien-tools 1   # molecular-dynamics processing
./tolkien-tools 2   # TD-DFT spectra
./tolkien-tools 3   # charge and spin analysis
./tolkien-tools 4   # multiwavelength kinetics
```

The charge/spin module can also propose ligand groups for transition-metal
coordination complexes from a representative geometry. It reports
metal--donor contacts, denticity, composition, and uncoordinated molecular
components, and provides an interactive color-coded viewer before the groups
are accepted for statistical analysis. Single-system outputs are collected in
`charge_spin_results/` instead of being mixed with the input calculations.

Run the dependency guide with:

```bash
./tolkien-tools requirements
```

The core requirements are Python 3.10 or newer, NumPy, SciPy and Matplotlib.
The 3D viewers can additionally use py3Dmol and Plotly. Some specialized
workflows require external programs such as `cpptraj`.

More detailed documentation is available in
[`TolkienTools/README.md`](TolkienTools/README.md) and in each module folder.
