# Tolkien Tools

An interactive launcher for computational chemistry, spectroscopy and molecular
dynamics analysis. Run the main command from the directory containing your data:

```bash
tolkien-tools
```

Direct shortcuts and help:

```bash
tolkien-tools --help
tolkien-tools requirements
tolkien-tools 1   # molecular dynamics processing
tolkien-tools 2   # TD-DFT NEA spectra
tolkien-tools 3   # charge and spin analysis
tolkien-tools 4   # multiwavelength kinetics
```

Arguments after the module number or alias are forwarded to that module:

```bash
tolkien-tools 4 117.txt --fit-method nnls --no-plot
tolkien-tools md --help
tolkien-tools td --help
tolkien-tools kinetics --help
```

The launcher resolves scripts by absolute path and preserves the directory from
which you invoke it, so modules can discover inputs in the current directory.

## Installation

Clone the repository into a stable tools directory:

```bash
mkdir -p ~/Soft
cd ~/Soft
git clone git@github.com:juanderboy/Tolkien-Tools.git
cd Tolkien-Tools
```

Install the core Python dependencies and optional 3D viewer packages:

```bash
python3 -m pip install numpy scipy matplotlib py3Dmol plotly
```

Make the root executable accessible from any directory by adding it to `PATH`
in `~/.bashrc`:

```bash
chmod +x ~/Soft/Tolkien-Tools/tolkien-tools
echo 'export PATH="$HOME/Soft/Tolkien-Tools:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Replace `$HOME/Soft/Tolkien-Tools` if you cloned elsewhere. Check the installation:

```bash
tolkien-tools requirements
python3 -c "import numpy, scipy, matplotlib; print('OK')"
```

## Requirements and modules

Python 3.10 or newer, NumPy, SciPy and Matplotlib are the core requirements.
`py3Dmol` and `plotly` are optional 3D viewer backends. Ubuntu/Debian users can
also install the core packages through the system package manager:

```bash
sudo apt install python3 python3-numpy python3-scipy python3-matplotlib
```

ORCA and LIO do not need to be installed: the tools read existing outputs.
Opening TD-DFT HTML reports can use `xdg-open` on Linux or `wslview` on WSL;
reports can also be opened manually.

- [Molecular dynamics processing](md_processing/README.md): rebuild fragmented
  runs in numbered directories containing `qm.xyz`, `d_QM.in`, population files
  and `QM_*.nc`. `inspect-merge` inspects segments and merges XYZ and population
  data; `split-nc` samples `rst7` restarts and synchronized QM XYZ frames using
  external `cpptraj`. Geometry analysis uses the standard library, optional
  Matplotlib plots, and optional py3Dmol/Plotly HTML viewers.
- [TD-DFT spectra](td_dft/README.md): Nuclear Ensemble Approximation absorption
  spectra from ORCA TD-DFT outputs, usually `TD_*.out`. Uses NumPy,
  `scipy.signal.find_peaks` and Matplotlib.
- [Charge and spin analysis](charge_spin/README.md): Mulliken, CHELPG, Loewdin,
  Hirshfeld and spin populations. LIO mode reads `mq_*.dat` and optional
  `ms_*.dat`; ORCA mode reads `<prefix>_N.out` or `<prefix>_N.dat`. Uses NumPy,
  SciPy KDE/peak analysis and Matplotlib.
- [Multiwavelength kinetics](kinetics/README.md): fit experimental
  wavelength-by-time tables and convert `.KD` files. Uses NumPy,
  `scipy.optimize`, NNLS and Matplotlib.
