# TD-DFT NEA Spectra

Nuclear Ensemble Approximation (NEA) absorption spectra from ORCA TD-DFT
calculations. The launcher entry point is `td_analyze.py`:

```bash
tolkien-tools
tolkien-tools td --help
tolkien-tools td TD_10.out --export
tolkien-tools td . --mode energy --no-html
```

Use a single `TD_*.out` file or a directory containing many outputs. Folder
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
