# Charge and Spin Analysis

Analyze Mulliken, CHELPG, Loewdin, Hirshfeld and spin populations through
`charge_spin_analysis.py`, the entry point used by the launcher:

```bash
tolkien-tools
tolkien-tools charges
```

## Inputs and outputs

In single-system mode, all generated files go into `charge_spin_results/`
under the analysis working directory. This includes consolidated ORCA/LIO
files, series, averages, histograms, figures, reports and HTML viewers.
Original outputs, XYZ files and other inputs remain in the parent directory.

Global comparisons always generate `global_<analysis>_histograms.png`.
The interactive flow also offers a clean publication figure,
`global_<analysis>_histograms_paper.png`, without titles, legends, axis names
or axis numbers, retaining tick marks for external editing.

Global mode accepts individual atoms and entities defined in fragment mode.
Reusing entities reads `spin_fragment_definitions.dat` and the latest
`actor_<name>_*` series, avoiding mixtures of older atom and fragment files.
Names such as `Fe_Porphyrin`, `X1` and `X2` can also be entered manually.
Hyphens, periods and underscores are normalized, so `Fe-Porphyrin` and
`Fe_Porphyrin` can be compared across systems. New analyses are discovered
inside `charge_spin_results/`; older results in system directories remain
supported.

## Spin statistics and atom selection

The interactive flow offers:

- **Raw spin values from each snapshot** (default): analyze the population
  reported by each snapshot/output directly.
- **Spin fraction relative to the selected atoms**: normalize each snapshot
  by the summed spin of the selected atoms/entities.

Atoms can be selected manually or by automatic spin localization. Automatic
mode calculates `abs(spin_atom) / sum(abs(spin_all_atoms))` for each snapshot,
averages over the trajectory, and creates individual histograms for atoms
above the requested threshold (5% by default). Other atoms are combined into
an entity named `resto` with its own histogram.

## Molecular fragments and coordination complexes

Fragment mode analyzes regions such as Fe/porphyrin/water/histidine. Before
entering fragment compositions, choose manual definitions or an automatic
proposal for coordination complexes.

The proposal detects transition metals, infers metal-donor contacts from
distances, removes metals from the molecular graph, and assigns `L1`, `L2`,
... to the remaining connected components. Each component is described by
coordination status, atom counts/types, donor atoms and denticity. Nearby
components without a metal bond are explicitly marked as uncoordinated.

The proposal is saved in `coordination_ligand_proposal.dat` and displayed in
`coordination_ligand_viewer.html`, using a color per component and atom/group
labels. This distance-based suggestion must be reviewed before acceptance.

Fragment definitions can combine detected groups and metals, e.g. `Fe88 + L1`,
individual atoms separated by spaces or commas, ranges such as `10-18`, and
the token `remaining`. Manual mode creates
`spin_fragment_numbering_viewer.html` with all atoms numbered.

Each fragment is treated as an entity whose spin per snapshot is the sum of
its atomic spins. Definitions are saved in `spin_fragment_definitions.dat`;
main outputs use the suffix `with_fragments`. Unassigned atoms can be excluded,
grouped automatically as `resto`, or assigned by redefining fragments.
With exclusions, raw mode preserves the selected fragment populations;
fraction mode normalizes by their summed spin only.

## Geometry viewers

In ORCA mode, `spin_localization_viewer.html` uses one representative geometry:
the first sorted output containing `CARTESIAN COORDINATES (ANGSTROEM)`, even
when many `SP_*.out` files contribute to the histograms. In LIO mode, an
available XYZ (`qm_completo.xyz`, `qm.xyz` or another `*.xyz`) supplies the
geometry. Explicit connectivity inferred from conservative distance limits
avoids spurious long bonds from 3Dmol auto-bonding.

When opening a viewer on WSL, the program opens its directory with
`explorer.exe .`; double-click the HTML file to view it. On native Linux,
it tries the default browser.

## Internal organization

- `charge_spin_analysis.py`: launcher-compatible entry point.
- `charge_spin_cli.py`: interactive workflow and orchestration.
- `charge_spin_common.py`: utilities, prompts and configuration.
- `charge_spin_io.py`: consolidated files and time-series input/output.
- `charge_spin_orca.py`: population extraction from ORCA outputs.
- `charge_spin_stats.py`: KDE, histograms, normalization and spin checks.
- `charge_spin_plotting.py`: histograms and time-series figures.
- `charge_spin_viewer.py`: spin localization and geometry HTML viewer.
- `charge_spin_coordination.py`: ligand/component proposals.
- `charge_spin_global.py`: comparisons across subdirectories.
