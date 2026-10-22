# Molecular Dynamics Processing

Rebuild and analyze molecular dynamics runs fragmented across numbered
subdirectories (`1/`, `2/`, `3/`, ...).

```bash
tolkien-tools 1
tolkien-tools md
tolkien-tools md --help
```

The expected QMMM/MD layout has a root directory with numbered segments
containing files such as `qm.xyz`, `d_QM.in`, `mulliken`, `mulliken_spin`,
`lowdin`, `lowdin_spin` and `QM_*.nc`.

## Inspecting and merging trajectories

```bash
tolkien-tools md inspect-merge
tolkien-tools md inspect-merge --merge yes --out qm_completo.xyz
tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --out qm_completo.xyz
tolkien-tools md inspect-merge --merge no
```

`inspect-merge` displays the segment panel and, in an interactive terminal,
offers to merge XYZ coordinates and charge/spin data immediately. Entire
segments can be excluded using lists and ranges such as `2,5-7`.

Time steps are read from `d_QM.in`; `nstlim` is retained as the planned run
length. Actual durations use the number of frames present in `qm.xyz`, so
truncated segments do not inflate total duration.

Individual dynamic frames can also be excluded without dropping a whole
segment. Indices are 1-based after discarding the initial coordinate block
from each `qm.xyz`, as in the normal merge. Local segment exclusions:

```bash
tolkien-tools md inspect-merge --merge yes --exclude-frames '7:754-810;8:30,35-40'
```

Global exclusions refer to the concatenated mergeable trajectory:

```bash
tolkien-tools md inspect-merge --merge yes --exclude-global-frames 7030-7080,9000
tolkien-tools md inspect-merge --merge yes --exclude-frames 5900-6288
```

Local and global exclusions can be combined with each other and `--exclude`.
To prune original segment files for later processing without repeating the
exclusions, use `--prune-originals`. This rewrites affected `qm.xyz` and
population files while keeping `.bak_frames_YYYYmmdd_HHMMSS` backups:

```bash
tolkien-tools md inspect-merge --merge yes --exclude-frames '7:754-810' --prune-originals
tolkien-tools md inspect-merge --merge yes --exclude-frames '7:754-810' --prune-originals --backup-suffix .bak_before_pruning
```

XYZ merging also consolidates `mulliken`, `mulliken_spin`, `lowdin` and
`lowdin_spin` from the same selected segments into `*_full.dat` files.
Geometry exclusions apply equally to populations. To merge only XYZ:

```bash
tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --no-pop
```

`merge-xyz` remains available for compatibility; the recommended flow starts
with `inspect-merge`. Both discard the first coordinate block of every
segment and count dynamics from the second block. Population consolidation
also discards the first block of each segment:

- `mulliken` -> `mulliken_full.dat`
- `mulliken_spin` -> `mulliken_spin_full.dat`

The resulting files can be analyzed with `tolkien-tools 3` in LIO mode.

## Sampling NetCDF restarts

```bash
tolkien-tools md split-nc
tolkien-tools md split-nc --count 100
tolkien-tools md split-nc --count 100 --skip-initial-ps 0.2
tolkien-tools md split-nc --root run --nc-pattern 'QM_*.nc' --count 100
```

`split-nc` requires `cpptraj`. It detects `.prmtop` topology in the root
and NetCDF files in numbered subdirectories. If multiple topologies exist,
it asks which to use. After counting frames and displaying the total, it
asks how many `rst7` files to generate. Snapshots are distributed over the
concatenated trajectory and saved as `restarts/QM_1.rst7`, `QM_2.rst7`, ...;
the topology is copied into the same directory.

An optional initial period can be discarded in picoseconds, interactively
after the NetCDF panel or with `--skip-initial-ps`. Frames with
`time_ps >= cutoff` are retained; global time uses each segment's `dt * ntwx`.

`restarts/qm_snapshots.xyz` contains QM coordinates synchronized to the
selected restarts. NetCDF local frame `k` maps to QM dynamic frame `k * ntwx`,
using `ntwx` from `d_QM.in`. LIO XYZ files repeat the initial geometry twice;
the first block is discarded and the remaining dynamic sequence is indexed,
matching `inspect-merge`/`merge-xyz`. XYZ comments record restart index,
global NC frame, segment, NC file, local NC frame and corresponding QM frame.

To omit QM XYZ output, or use the legacy explicit topology/pattern/selection:

```bash
tolkien-tools md split-nc --count 100 --no-qm-xyz
tolkien-tools md split-nc system.prmtop 'QM_*.nc' 250-300
```

## Geometry analysis and viewers

```bash
tolkien-tools md geom
tolkien-tools md geom qm_completo.xyz --metric dFeN:distance:9,10
```

`geom` creates a 3D HTML viewer of the first frame before requesting or
calculating metrics, and tries to open it in a browser. The default backend
is py3Dmol with spheres, sticks and element colors. Atom indices are always
shown by default to help select atoms for metrics.

```bash
tolkien-tools md geom qm_completo.xyz --viewer-labels hover
tolkien-tools md geom qm_completo.xyz --viewer-backend py3dmol
tolkien-tools md geom qm_completo.xyz --viewer-backend plotly
tolkien-tools md geom qm_completo.xyz --viewer-backend auto
tolkien-tools md geom qm_completo.xyz --no-open-viewer
```

Plotly is used only if explicitly requested with `plotly` or `auto`.

### Iron displacement from the porphyrin plane

For heme proteins, `heme_oop` measures the signed normal distance of Fe from
the best-fit plane of the four coordinated pyrrolic nitrogens, in Angstrom.
Positive displacement points toward the proximal histidine.

```bash
tolkien-tools md geom qm_completo.xyz --metric FeOOP:heme_oop:auto
tolkien-tools md geom qm_completo.xyz --metric FeOOP:heme_oop:11
tolkien-tools md geom qm_completo.xyz --metric FeOOP:heme_oop:11,2,4,6,8
tolkien-tools md geom qm_completo.xyz --metric FeOOP:heme_oop:11,2,4,6,8,10
tolkien-tools md geom qm_completo.xyz --metric FeOOP_abs:heme_oop_abs:auto
```

Specify Fe alone when multiple Fe atoms exist, Fe,N1,N2,N3,N4 to set the plane
explicitly, or append NHis to specify the proximal coordinating nitrogen.
Automatic detection chooses the nearby N4 set that is most planar and best
centered around projected Fe, avoiding the axial histidine N; the remaining
nearby N sets the proximal direction. `heme_oop_abs` reports absolute distance.

## Comparing geometry metrics across trajectories

After running `geom` in several system directories:

```bash
tolkien-tools md geom-compare
```

Interactive mode detects directories containing `*_metrics.csv` and asks
which directories, metric labels and histogram binning to use. Labels must
match the ones defined with `geom`, e.g. `dFeO`. Noninteractive examples:

```bash
tolkien-tools md geom-compare systemA systemB systemC --metric dFeO
tolkien-tools md geom-compare systemA systemB systemC --metric 'FeOOP,dFeN,dFeO'
tolkien-tools md geom-compare systemA systemB --metric FeOOP --bins 50
tolkien-tools md geom-compare systemA systemB --metric FeOOP --bins auto
tolkien-tools md geom-compare systemA systemB --metric FeOOP --kde-smooth 4.0
tolkien-tools md geom-compare systemA systemB --metric FeOOP --paper-figure yes
tolkien-tools md geom-compare systemA systemB --metric FeOOP --no-kde
tolkien-tools md geom-compare --root comparisons --metric dFeO
```

Outputs default to `geom_global/`:

- `geom_compare_<metric>_histograms.png`: overlaid histograms by system.
- `geom_compare_<metric>_histograms_paper.png`: optional clean figure without
  titles, legends, axis names or axis numbers, retaining tick marks.
- `geom_compare_<metric>_values.csv`: plotted data.
- `geom_compare_<metric>_summary.csv`: statistics by system.
- `geom_compare_<metric>_kde_modes.csv`: relevant KDE maxima.

KDE smoothing exceeds SciPy's default to suppress small, hard-to-interpret
maxima. `--kde-smooth` multiplies Scott bandwidth; higher values are smoother.
The noninteractive default is `2.5`; `4.0` emphasizes dominant populations.
`--kde-prominence` controls reported maxima (default `0.08`, excluding peaks
with prominence below 8% of each system's maximum density).
