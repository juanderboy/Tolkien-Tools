#!/usr/bin/env python3
"""Interactive launcher for Tolkien's analysis routines."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    script: Path
    default_args: tuple[str, ...] = ()
    ask_input_file: bool = False


TOOLS = (
    Tool(
        name="Molecular dynamics processing",
        description="Rebuild and analyze fragmented MD/QMMM runs.",
        script=ROOT / "TolkienTools" / "md_processing" / "md_process.py",
    ),
    Tool(
        name="TD-DFT spectra",
        description="Build NEA absorption spectra from ORCA TD-DFT outputs.",
        script=ROOT / "TolkienTools" / "td_dft" / "td_analyze.py",
    ),
    Tool(
        name="Charge and spin analysis",
        description="Analyze Mulliken, CHELPG, Lowdin, Hirshfeld and spin populations.",
        script=ROOT / "TolkienTools" / "charge_spin" / "charge_spin_analysis.py",
    ),
    Tool(
        name="Multilambda kinetics",
        description="Fit experimental multiwavelength kinetic spectra.",
        script=ROOT / "TolkienTools" / "kinetics" / "kinet_python.py",
        ask_input_file=True,
    ),
)

MD_SUBCOMMANDS = (
    (
        "inspect-merge",
        "Inspect segments, frames and times, then optionally merge XYZ and charge/spin data.",
    ),
    (
        "geom",
        "Create a 3D viewer and analyze distances, angles and dihedrals.",
    ),
    (
        "geom-compare",
        "Compare geometric metric histograms across trajectories.",
    ),
    (
        "split-nc",
        "Inspect fragmented QM_*.nc trajectories and generate sampled restarts.",
    ),
)


REQUIREMENTS_GUIDE = """\
Installing and running Tolkien Tools

Core requirements:
  - Python 3.10 or newer.
  - Python packages: numpy, scipy, matplotlib.
  - Optional 3D viewer packages: py3Dmol, plotly.
  - Install with:
      python3 -m pip install numpy scipy matplotlib py3Dmol plotly
    Ubuntu/Debian system packages are also available:
      sudo apt install python3 python3-numpy python3-scipy python3-matplotlib

External programs:
  - ORCA and LIO are not required: Tolkien Tools reads their existing outputs.
  - `tolkien-tools md split-nc` requires cpptraj on PATH.
  - Opening TD-DFT HTML reports can use xdg-open on Linux or wslview on WSL.
    Reports can also be opened manually.

Expected input by module:
  1. Molecular dynamics processing
     - Fragmented runs in numbered subdirectories (1, 2, 3, ...).
     - Inspect segments, merge qm.xyz and populations, analyze geometry,
       and sample rst7 restarts from QM_*.nc using cpptraj.
     - Optional 3D HTML viewers use py3Dmol or Plotly.
     - Subcommands: inspect-merge, merge-xyz, geom, geom-compare, split-nc.
  2. TD-DFT spectra
     - ORCA TD-DFT outputs, usually TD_*.out.
     - Uses numpy, scipy.signal.find_peaks and matplotlib.
     - Generates PNG, CSV/DAT and interactive HTML reports.
  3. Charge and spin analysis
     - LIO mode: mq_*.dat and optional ms_*.dat series.
     - ORCA mode: <prefix>_N.out or <prefix>_N.dat population tables.
     - Supports Mulliken, Loewdin, Hirshfeld, CHELPG and spin populations.
     - Uses numpy, scipy.stats.gaussian_kde, scipy.signal.find_peaks
       and matplotlib.
  4. Multiwavelength kinetics
     - Wavelength-by-time spectrophotometric tables in semicolon-delimited
       TXT/CSV files; also converts .KD files.
     - Uses numpy, scipy.optimize, scipy.optimize.nnls and matplotlib.

Quick check:
  python3 -c "import numpy, scipy, matplotlib; print('OK')"
"""


def print_banner() -> None:
    width = 74
    lines = [
        "Tolkien Tools",
        "Computational chemistry and spectroscopy utilities",
        "vibecoded by Tolkien, 2026",
    ]

    print()
    print("=" * width)
    for line in lines:
        print(line.center(width))
    print("=" * width)
    print()


def print_requirements_guide() -> None:
    print()
    print(REQUIREMENTS_GUIDE)


def prompt_tool_choice() -> Tool | None:
    print("What would you like to do?")
    for index, tool in enumerate(TOOLS, start=1):
        print(f"  {index}. {tool.name}")
        print(f"     {tool.description}")
    print("  i. Installation and dependency guide")
    print("  0. Exit")
    print()

    choice = input("Choose an option: ").strip()
    if choice.lower() in {"i", "info", "deps", "dependencias", "requirements", "requisitos"}:
        print_requirements_guide()
        return prompt_tool_choice()
    if choice in {"", "0", "q", "quit", "exit", "salir"}:
        return None
    if not choice.isdigit():
        raise ValueError("The option must be a number.")

    index = int(choice)
    if index < 1 or index > len(TOOLS):
        raise ValueError(f"The option must be between 1 and {len(TOOLS)}.")
    return TOOLS[index - 1]


def tool_from_cli_choice(choice: str) -> Tool:
    normalized = choice.strip().lower()
    aliases = {
        "1": 0,
        "md": 0,
        "dynamics": 0,
        "dinamica": 0,
        "dinamicas": 0,
        "procesado": 0,
        "processing": 0,
        "2": 1,
        "td": 1,
        "td-dft": 1,
        "spectra": 1,
        "espectros": 1,
        "3": 2,
        "charges": 2,
        "charge": 2,
        "spin": 2,
        "cargas": 2,
        "4": 3,
        "kinetics": 3,
        "kinetic": 3,
        "kinet": 3,
        "cinetica": 3,
        "cineticas": 3,
    }
    if normalized not in aliases:
        raise ValueError(
            "Choose 1, 2, 3 or 4 "
            "(aliases such as md, td, charges or kinetics are also accepted). "
            "For dependencies, use requirements or deps."
        )
    return TOOLS[aliases[normalized]]


def is_requirements_choice(choice: str) -> bool:
    return choice.strip().lower() in {
        "i",
        "info",
        "deps",
        "dependencias",
        "requirements",
        "requisitos",
        "--requirements",
        "--deps",
    }


def prompt_extra_args(tool: Tool) -> list[str]:
    args = list(tool.default_args)

    return args


def prompt_md_processing_choice() -> tuple[str, list[str] | None]:
    print()
    print("Molecular dynamics processing tools:")
    for index, (command, description) in enumerate(MD_SUBCOMMANDS, start=1):
        print(f"  {index}. {command}")
        print(f"     {description}")
    print("  h. General processing help")
    print("  0. Back to the main menu")
    print("  q. Exit")
    print()

    choice = input("Choose a tool (Enter = inspect-merge): ").strip().lower()
    if choice in {"h", "help", "ayuda"}:
        return "launch", ["--help"]
    if choice in {"0", "m", "menu", "principal"}:
        return "main", None
    if choice in {"q", "quit", "exit", "salir"}:
        return "exit", None
    if choice == "":
        command = "inspect-merge"
    else:
        if not choice.isdigit():
            raise ValueError("Select the processing tool by number.")
        index = int(choice)
        if index < 1 or index > len(MD_SUBCOMMANDS):
            raise ValueError(f"The tool must be between 1 and {len(MD_SUBCOMMANDS)}.")
        command, _description = MD_SUBCOMMANDS[index - 1]

    args = [command]
    if command == "split-nc":
        nc_pattern = input("NetCDF trajectory pattern (Enter = QM_*.nc): ").strip() or "QM_*.nc"
        skip_initial_ps = input("Skip initial frames before how many ps? (Enter = ask later): ").strip()
        count = input("Number of rst7 files to generate (Enter = ask after inspection): ").strip()
        if nc_pattern != "QM_*.nc":
            args.extend(["--nc-pattern", nc_pattern])
        if skip_initial_ps:
            try:
                if float(skip_initial_ps) < 0.0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("The initial time to skip must be a number >= 0.") from exc
            args.extend(["--skip-initial-ps", skip_initial_ps])
        if count:
            if not count.isdigit():
                raise ValueError("The number of rst7 files must be an integer.")
            args.extend(["--count", count])
    return "launch", args


def prompt_md_continue() -> str:
    print()
    choice = input("Enter = return to the dynamics panel; m = main menu; q = exit: ").strip().lower()
    if choice in {"q", "quit", "exit", "salir"}:
        return "exit"
    if choice in {"m", "menu", "principal", "0"}:
        return "main"
    return "md"


def run_md_processing_menu(tool: Tool, last_returncode: int = 0) -> tuple[int, bool]:
    while True:
        action, args = prompt_md_processing_choice()
        if action == "exit":
            return last_returncode, True
        if action == "main":
            return last_returncode, False
        if action == "cancel":
            continue
        last_returncode = launch_tool(tool, args or [])
        destination = prompt_md_continue()
        if destination == "exit":
            return last_returncode, True
        if destination == "main":
            return last_returncode, False


def launch_tool(tool: Tool, args: list[str]) -> int:
    if not tool.script.exists():
        raise FileNotFoundError(f"Tool script not found: {tool.script}")

    command = [sys.executable, str(tool.script), *args]
    print()
    print(f"Opening: {tool.name}")
    print(f"Command: {' '.join(command)}")
    print()
    sys.stdout.flush()
    return subprocess.run(command, cwd=Path.cwd()).returncode


def prompt_continue() -> bool:
    print()
    choice = input("Enter = return to the main menu; q = exit: ").strip().lower()
    return choice not in {"q", "quit", "exit", "salir", "0"}


def main() -> int:
    print_banner()
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] in {"-h", "--help", "help"}:
                print("Usage: tolkien-tools [TOOL [ARGS...]]")
                print("Run without arguments for the interactive menu.")
                for index, tool in enumerate(TOOLS, 1):
                    print(f"  {index}. {tool.name}: {tool.description}")
                print("Aliases: md, td, charges, kinetics")
                print("Use TOOL --help for module options; requirements for dependencies.")
                return 0
            if is_requirements_choice(sys.argv[1]):
                print_requirements_guide()
                return 0
            tool = tool_from_cli_choice(sys.argv[1])
            args = sys.argv[2:]
            return launch_tool(tool, args)
        else:
            last_returncode = 0
            while True:
                tool = prompt_tool_choice()
                if tool is None:
                    return last_returncode
                if tool.name == "Molecular dynamics processing":
                    last_returncode, should_exit = run_md_processing_menu(tool, last_returncode)
                    if should_exit:
                        return last_returncode
                    continue
                args = prompt_extra_args(tool)
                last_returncode = launch_tool(tool, args)
                if not prompt_continue():
                    return last_returncode
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
