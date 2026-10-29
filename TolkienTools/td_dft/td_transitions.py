"""Explore spin-free ORCA 6 TD-DFT/TDA states and plot canonical DFT orbitals."""

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


NUMBER = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


@dataclass
class Contribution:
    source: int
    source_spin: str
    target: int
    target_spin: str
    weight: float
    coefficient: float


@dataclass
class Transition:
    state: int
    multiplicity: int
    energy_ev: float
    spin_squared: float | None = None
    multiplicity_estimated: bool = False
    label: str = ""
    wavelength_nm: float | None = None
    fosc: float | None = None
    contributions: list[Contribution] = field(default_factory=list)


def read_transitions(filename):
    """Match printed state blocks to the first *uncorrected* electric table.

    ORCA can number triplets globally in STATE blocks but locally in absorption
    labels. Match multiplicity and rounded energy, never table row number.
    Ambiguous or repeated calculation blocks are rejected rather than guessed.
    """
    text = Path(filename).read_text(errors="replace")
    header = re.compile(rf"^\s*STATE\s+(\d+):\s+E=\s*{NUMBER}\s+au\s+({NUMBER})\s+eV.*?Mult\s+(\d+)")
    pair = re.compile(rf"^\s*(\d+)([ab])\s*->\s*(\d+)([ab])\s*:\s*({NUMBER})\s*\(c=\s*({NUMBER})\)")
    states = []
    current = None
    table = []
    in_table = False
    table_done = False
    for line in text.splitlines():
        match = header.match(line)
        if match:
            current = Transition(int(match[1]), int(match[3]), number(match[2]))
            spin = re.search(rf"<S\*\*2>\s*=\s*({NUMBER})", line)
            current.spin_squared = number(spin[1]) if spin else None
            current.multiplicity_estimated = 'UHF/UKS reference:' in text
            states.append(current)
            continue
        match = pair.match(line)
        if match and current is not None:
            current.contributions.append(Contribution(int(match[1]), match[2], int(match[3]), match[4], number(match[5]), number(match[6])))
            continue
        if line.strip():
            current = None
        if line.strip() == "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" and not table_done:
            in_table = True
            continue
        if in_table:
            match = re.match(rf"\s*\d+-\S+\s+->\s+(\d+-(\d+)[A-Za-z]\S*)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})", line)
            if match:
                table.append((match[1], int(match[2]), number(match[3]), number(match[5]), number(match[6])))
            elif table and line.strip() and set(line.strip()) != {"-"}:
                in_table = False
                table_done = True
    if not states or not table:
        raise ValueError("No supported ORCA 6 spin-free STATE blocks/electric absorption table found.")
    if len({s.state for s in states}) != len(states):
        raise ValueError("Repeated STATE numbers: split concatenated calculations before exploring.")
    assigned = set()
    for label, mult, energy, wavelength, fosc in table:
        candidates = [s for s in states if s.multiplicity == mult and abs(s.energy_ev - energy) <= 0.0006]
        # The table's local root label disambiguates nearly degenerate energies.
        if len(candidates) > 1:
            manifold = [s for s in states if s.multiplicity == mult]
            root = int(label.split('-')[0])
            global_match = [s for s in candidates if s.state == root]
            candidates = global_match or [s for s in candidates if manifold.index(s) + 1 == root]
        if len(candidates) != 1 or candidates[0].state in assigned:
            raise ValueError(f"Cannot uniquely associate absorption label {label} with a STATE block.")
        state = candidates[0]
        assigned.add(state.state)
        state.label, state.energy_ev = label, energy
        state.wavelength_nm, state.fosc = wavelength, fosc
    for state in states:
        state.contributions.sort(key=lambda p: p.weight, reverse=True)
    return states, "SOC CORRECTED ABSORPTION SPECTRUM" in text


def read_geometry(filename):
    """Read the last printed Angstrom geometry without touching source files."""
    blocks = Path(filename).read_text(errors="replace").split("CARTESIAN COORDINATES (ANGSTROEM)")[1:]
    geometry = []
    for block in blocks:
        rows = []
        for line in block.splitlines():
            fields = line.split()
            if len(fields) == 4 and re.fullmatch(r"[A-Z][a-z]?", fields[0]):
                try:
                    rows.append((fields[0], *(number(v) for v in fields[1:])))
                except ValueError:
                    break
            elif rows:
                break
        if rows:
            geometry = rows
    if not geometry:
        raise ValueError("No Cartesian geometry in Angstrom found in the output.")
    return geometry


def select_pairs(spec, contributions):
    if spec.strip().lower() == "all":
        indices = list(range(len(contributions)))
    else:
        try:
            indices = sorted({int(value.strip()) - 1 for value in spec.split(',')})
        except ValueError as exc:
            raise ValueError("Choose contribution numbers separated by commas, or 'all'.") from exc
    if not indices or any(i < 0 or i >= len(contributions) for i in indices):
        raise ValueError("Contribution selection is empty or out of range.")
    return [contributions[i] for i in indices]


def show_states(states, min_fosc=0.0):
    print("\nSTATE  Label       lambda/nm    E/eV        fosc      printed pairs")
    for state in states:
        if state.fosc is not None and state.fosc < min_fosc:
            continue
        wave = f"{state.wavelength_nm:.1f}" if state.wavelength_nm is not None else "N/A"
        fosc = f"{state.fosc:.9f}" if state.fosc is not None else "N/A"
        print(f"{state.state:5d}  {state.label or 'unmatched':10s} {wave:>9s} {state.energy_ev:8.4f} {fosc:>12s} {len(state.contributions):6d}")


def show_composition(state):
    print(f"\nSTATE {state.state}, multiplicity {state.multiplicity}, label {state.label}")
    print("Pair  DFT orbital origin -> destination    printed weight    coefficient")
    for i, pair in enumerate(state.contributions, 1):
        print(f"{i:4d}  {pair.source}{pair.source_spin:1s} -> {pair.target}{pair.target_spin:1s}                 {pair.weight:12.6f} {pair.coefficient:14.8f}")
    if state.multiplicity_estimated:
        print(f"UKS/UHF: multiplicity is an estimate; printed <S**2> = {state.spin_squared}.")
    print("Weights and signed coefficients are those printed by ORCA; omitted contributions are not reconstructed.")
    print("Orbital numbers are ORCA indices (zero based); a = alpha/closed shell, b = beta.")


def validate_cube(path, geometry, reference_grid=None, expected_orbital=None):
    """Reject missing/partial cubes and inconsistent geometries or grids."""
    with path.open() as stream:
        stream.readline()
        stream.readline()
        origin = stream.readline().split()
        signed_natoms = int(origin[0])
        natoms = abs(signed_natoms)
        grid = [tuple(number(v) for v in origin[1:4])]
        counts = []
        for _ in range(3):
            row = stream.readline().split()
            counts.append(int(row[0]))
            grid.append(tuple(number(v) for v in row))
        if natoms != len(geometry) or any(n <= 0 for n in counts):
            raise ValueError("Cube has an unexpected atom count or unsupported grid.")
        for atom in geometry:
            row = stream.readline().split()
            xyz = [number(v) * 0.529177210903 for v in row[2:5]]
            if len(xyz) != 3 or any(abs(a-b) > 2e-4 for a, b in zip(xyz, atom[1:])):
                raise ValueError("GBW cube geometry differs from the ORCA output.")
        if signed_natoms < 0:
            orbital_ids = stream.readline().split()
            if len(orbital_ids) != 2 or orbital_ids[0] != '1':
                raise ValueError("Expected a single-orbital cube.")
            if expected_orbital is not None and int(orbital_ids[1]) != expected_orbital:
                raise ValueError("Cube contains a different orbital than requested.")
        value_count = 0
        for line in stream:
            for value in line.split():
                if not math.isfinite(number(value)):
                    raise ValueError("Non-finite cube value.")
                value_count += 1
        if value_count != counts[0] * counts[1] * counts[2]:
            raise ValueError("Incomplete cube data.")
    if reference_grid is not None and grid != reference_grid:
        raise ValueError("Orbital cubes have different grids.")
    return grid


def write_vmd(directory, cubes):
    # VMD's -e startup reader may evaluate lines without setting info script.
    # Record an escaped absolute fallback, while supporting normal Tcl source.
    escaped = str(directory.resolve())
    for old, new in [('\\', '\\\\'), ('"', '\\"'), ('$', '\\$'), ('[', '\\['), (']', '\\]'), ('\n', '\\n'), ('\r', '\\r')]:
        escaped = escaped.replace(old, new)
    lines = ['# Open with: vmd -e view.vmd', f'set here "{escaped}"',
             'if {[info script] ne ""} {set here [file dirname [file normalize [info script]]]}']
    for cube in cubes:
        # Generated names only; no source paths are interpolated into Tcl.
        lines += [f'mol new [file join $here {cube.name}] type cube waitfor all',
                  'mol delrep 0 top', 'mol representation Licorice 0.15 12 12',
                  'mol color Element', 'mol addrep top']
        for level, color in [('0.03', 0), ('-0.03', 1)]:
            lines += [f'mol representation Isosurface {level} 0 0 0 1 1', f'mol color ColorID {color}', 'mol material Transparent', 'mol addrep top']
        lines += ['mol off top']
    lines += ['# Orbitals are separate molecules. Toggle them in the VMD molecule list.', 'mol on top', 'display resetview']
    (directory / 'view.vmd').write_text('\n'.join(lines) + '\n')


def generate_cubes(output, gbw, state, pairs, output_dir, executable='orca_plot', grid=60):
    if not 10 <= grid <= 300:
        raise ValueError("Grid intervals must be between 10 and 300.")
    binary = shutil.which(str(executable))
    if binary is None:
        raise ValueError("orca_plot not found; supply --orca-plot /path/to/orca_plot.")
    binary = str(Path(binary).resolve())
    gbw = Path(gbw).expanduser().resolve(strict=True)
    geometry = read_geometry(output)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"state_{state.state:03d}_mult{state.multiplicity}"
    directory = root / stem
    suffix = 1
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            suffix += 1
            directory = root / f"{stem}_{suffix:02d}"
    metadata = {'output': str(Path(output).resolve()), 'gbw': str(gbw), 'orca_plot': binary,
                'grid_intervals': grid, 'state': asdict(state), 'selected_pairs': [asdict(p) for p in pairs],
                'orbital_type': 'canonical DFT; not NTO', 'status': 'started'}
    report = directory / 'transition.json'
    report.write_text(json.dumps(metadata, indent=2) + '\n')
    orbitals = sorted({(p.source, p.source_spin) for p in pairs} | {(p.target, p.target_spin) for p in pairs})
    cubes = []
    reference_grid = None
    try:
        with tempfile.TemporaryDirectory(prefix='.orca_plot_', dir=directory) as scratch:
            scratch = Path(scratch)
            shutil.copy2(gbw, scratch / 'orbitals.gbw')
            xyz = str(len(geometry)) + '\nGeometry from ORCA output (Angstrom)\n'
            xyz += '\n'.join(f'{a} {x:.8f} {y:.8f} {z:.8f}' for a, x, y, z in geometry) + '\n'
            (scratch / 'orbitals.xyz').write_text(xyz)
            for orbital, spin in orbitals:
                print(f"Generating DFT orbital {orbital}{spin} ({grid} grid intervals)...", flush=True)
                commands = f'2\n{orbital}\n3\n{0 if spin == "a" else 1}\n4\n{grid}\n11\n12\n'
                log = directory / f'orbital_{orbital}_{spin}.log'
                with log.open('w') as stream:
                    result = subprocess.run([binary, 'orbitals.gbw', '-i'], input=commands, text=True,
                                            stdout=stream, stderr=subprocess.STDOUT, cwd=scratch, timeout=600)
                raw = scratch / f'orbitals.mo{orbital}{spin}.cube'
                if result.returncode != 0 or not raw.is_file():
                    raise RuntimeError(f"orca_plot failed for {orbital}{spin}; see {log}")
                reference_grid = validate_cube(raw, geometry, reference_grid, orbital)
                destination = directory / f'orbital_{orbital}_{spin}.cube'
                shutil.move(str(raw), destination)
                cubes.append(destination)
        write_vmd(directory, cubes)
        metadata.update(status='complete', cubes=[p.name for p in cubes])
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        report.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Saved: {directory}\nVMD: vmd -e {directory / 'view.vmd'}")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', help='ORCA 6 output with spin-free states and absorption table')
    parser.add_argument('--list', action='store_true', help='List states without prompts or file writes')
    parser.add_argument('--state', type=int, help='Original STATE number (not absorption row number)')
    parser.add_argument('--pairs', default='1', help="Contribution numbers, e.g. 1,2 or all (default: largest printed weight)")
    parser.add_argument('--gbw', help='Corresponding canonical DFT GBW file')
    parser.add_argument('--grid', type=int, default=60, help='Grid intervals per axis (default: 60)')
    parser.add_argument('--orca-plot', default='orca_plot')
    parser.add_argument('--output-dir', default='electronic_spectrum/transitions')
    parser.add_argument('--min-fosc', type=float, default=0.0, help='Filter displayed table only')
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_fosc) or args.min_fosc < 0:
        parser.error('--min-fosc must be finite and non-negative.')
    if not 10 <= args.grid <= 300:
        parser.error('--grid must be between 10 and 300.')
    try:
        states, has_soc = read_transitions(args.output)
        print(f"Read {len(states)} spin-free states.")
        if has_soc:
            print("SOC-corrected states are present and excluded: this explorer uses spin-free orbital compositions.")
        if any(s.multiplicity_estimated for s in states):
            print("UKS/UHF reference: multiplicity labels are estimates based on <S**2>, not pure-spin assignments.")
        show_states(states, args.min_fosc)
        if args.list:
            return 0
        interactive = args.state is None
        while True:
            if interactive:
                choice = input('\nOriginal STATE number to inspect (q = back): ').strip()
                if choice.lower() in {'q', '0', ''}:
                    return 0
                if not choice.isdigit() or int(choice) not in {s.state for s in states}:
                    print('Choose a STATE number from the table.')
                    continue
                state_id = int(choice)
            else:
                state_id = args.state
            state = next((s for s in states if s.state == state_id), None)
            if state is None:
                raise ValueError(f'STATE {state_id} not found.')
            show_composition(state)
            if not state.contributions:
                if interactive:
                    continue
                raise ValueError('No printed orbital contributions for this state.')
            spec = args.pairs
            if interactive:
                spec = input('Pairs to plot [1], comma-separated, all, or q = back: ').strip() or '1'
            if interactive and spec.lower() == 'q':
                continue
            try:
                pairs = select_pairs(spec, state.contributions)
            except ValueError as exc:
                if not interactive:
                    raise
                print(exc)
                continue
            gbw = args.gbw
            if not gbw:
                if not interactive:
                    raise ValueError('--gbw is required when --state is supplied.')
                default = Path(args.output).expanduser().with_suffix('.gbw')
                gbw = input(f'Canonical DFT GBW [{default}]: ').strip() or str(default)
            executable = args.orca_plot
            grid = args.grid
            if interactive:
                executable = input(f'orca_plot executable [{executable}]: ').strip() or executable
                raw_grid = input(f'Grid intervals per axis [{grid}]: ').strip()
                if raw_grid:
                    try:
                        grid = int(raw_grid)
                    except ValueError:
                        print('Grid must be an integer.')
                        continue
            generate_cubes(args.output, gbw, state, pairs, args.output_dir, executable, grid)
            if not interactive:
                return 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
