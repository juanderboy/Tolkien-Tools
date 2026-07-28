#!/usr/bin/env python3
"""HTML molecular viewers for automatic spin-localization results.

The viewer highlights selected atoms on an XYZ or ORCA geometry and writes an
interactive 3D representation for visual inspection.
"""

import glob
import math
import os
import subprocess
import webbrowser
from pathlib import Path


def parse_first_xyz_frame(xyz_path):
    """
    Parse the first frame of a simple XYZ file.
    """
    with open(xyz_path, "r") as f:
        natoms_line = f.readline()
        if not natoms_line:
            raise ValueError("empty XYZ file")
        natoms = int(natoms_line.strip())
        comment = f.readline().rstrip("\n")
        atoms = []
        for atom_idx in range(1, natoms + 1):
            line = f.readline()
            if not line:
                raise ValueError("incomplete first XYZ frame")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"invalid XYZ atom line: {line.strip()}")
            atoms.append(
                {
                    "index": atom_idx,
                    "model_index": atom_idx - 1,
                    "element": normalize_element_symbol(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                }
            )
    return comment, atoms


def parse_orca_cartesian_coordinates(fname):
    """
    Parse the CARTESIAN COORDINATES (ANGSTROEM) block from an ORCA output.

    ORCA population tables are indexed from zero, so the returned atom index is
    zero-based to match the charge/spin analysis selections.
    """
    atoms = []
    in_block = False
    saw_separator = False

    with open(fname, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "CARTESIAN COORDINATES (ANGSTROEM)":
                in_block = True
                saw_separator = False
                atoms = []
                continue
            if not in_block:
                continue
            if stripped.startswith("---"):
                saw_separator = True
                continue
            if not saw_separator:
                continue
            if not stripped:
                break

            parts = stripped.split()
            if len(parts) < 4:
                break
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                break
            model_index = len(atoms)
            atoms.append(
                {
                    "index": model_index,
                    "model_index": model_index,
                    "element": normalize_element_symbol(parts[0]),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    if not atoms:
        raise ValueError(f"no CARTESIAN COORDINATES (ANGSTROEM) block found in {fname}")
    return f"{os.path.basename(fname)} | ORCA Cartesian coordinates | indices 0-based", atoms


def find_default_xyz_for_spin_viewer(search_dir="."):
    """
    Return a likely XYZ file for the spin-localization viewer.
    """
    search_dir = os.fspath(search_dir)
    candidates = (
        "qm_completo.xyz",
        "qm.xyz",
        "QM.xyz",
        "molecule.xyz",
        "mol.xyz",
    )
    for candidate in candidates:
        candidate_path = os.path.join(search_dir, candidate)
        if os.path.isfile(candidate_path):
            return candidate_path
    xyz_files = sorted(glob.glob(os.path.join(search_dir, "*.xyz")))
    return xyz_files[0] if xyz_files else None


def is_wsl_environment():
    """
    Return True when running inside Windows Subsystem for Linux.
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        kernel_release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False
    return "microsoft" in kernel_release or "wsl" in kernel_release


def open_html_viewer(path):
    """
    Open the viewer location in WSL or the HTML in a native Linux browser.
    """
    resolved_path = Path(path).resolve()
    if is_wsl_environment():
        try:
            subprocess.Popen(
                ["explorer.exe", "."],
                cwd=str(resolved_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[WARN] Could not open the viewer folder in Windows Explorer: {exc}")
            print(f"[INFO] Open this file manually: '{resolved_path}'")
            return
        print(
            f"[OK] Opened '{resolved_path.parent}' in Windows Explorer. "
            f"Double-click '{resolved_path.name}' to view it."
        )
        return

    try:
        opened = webbrowser.open(resolved_path.as_uri(), new=2)
    except Exception as exc:
        print(f"[WARN] Could not open viewer automatically: {exc}")
        return
    if not opened:
        print(f"[WARN] Could not open viewer automatically. Open '{path}' manually.")


def write_spin_localization_viewer(
    xyz_path,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
    label_all_atoms=False,
):
    """
    Write a py3Dmol-based HTML viewer highlighting localized-spin atoms.
    """
    comment, atoms = parse_first_xyz_frame(xyz_path)
    title = f"{os.path.basename(xyz_path)} | XYZ first frame | indices 1-based"
    write_spin_localization_viewer_from_atoms(
        title,
        atoms,
        output_path,
        highlighted_atom_ids,
        atom_types,
        avg_fraction_by_atom,
        label_all_atoms=label_all_atoms,
    )


def write_orca_spin_localization_viewer(
    orca_output_path,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
    label_all_atoms=False,
):
    """
    Write a spin-localization viewer from an ORCA output geometry block.
    """
    title, atoms = parse_orca_cartesian_coordinates(orca_output_path)
    write_spin_localization_viewer_from_atoms(
        title,
        atoms,
        output_path,
        highlighted_atom_ids,
        atom_types,
        avg_fraction_by_atom,
        label_all_atoms=label_all_atoms,
    )


def write_coordination_fragment_viewer_from_atoms(
    title,
    atoms,
    output_path,
    proposal,
):
    """
    Write an interactive viewer colored by proposed coordination component.
    """
    import py3Dmol

    palette = (
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    )
    component_by_atom = {}
    color_by_component = {}
    for idx, component in enumerate(proposal["components"]):
        color = palette[idx % len(palette)]
        color_by_component[component["id"]] = color
        for atom_id in component["atom_ids"]:
            component_by_atom[atom_id] = component

    metal_by_atom = {
        metal["atom_id"]: metal
        for metal in proposal["metals"]
    }

    view = py3Dmol.view(width=1080, height=760)
    view.addModel(atoms_to_mol_block(title, atoms), "mol")
    view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})

    for atom in atoms:
        atom_id = atom["index"]
        model_index = atom["model_index"]
        if atom_id in metal_by_atom:
            group_label = metal_by_atom[atom_id]["id"]
            color = "#7F7F7F"
            view.setStyle(
                {"index": model_index},
                {"stick": {"radius": 0.20, "color": color},
                 "sphere": {"scale": 0.55, "color": color}},
            )
        else:
            component = component_by_atom.get(atom_id)
            group_label = component["id"] if component is not None else "?"
            color = color_by_component.get(group_label, "#BDBDBD")
            view.setStyle(
                {"index": model_index},
                {"stick": {"radius": 0.16, "color": color},
                 "sphere": {"scale": 0.30, "color": color}},
            )

        view.addLabel(
            f"{atom_id} {group_label}",
            {
                "position": {"x": atom["x"], "y": atom["y"], "z": atom["z"]},
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.75,
                "fontSize": 11,
                "inFront": True,
            },
        )

    legend_lines = []
    for metal in proposal["metals"]:
        legend_lines.append(
            f"<li><span style=\"color:#7F7F7F\">&#9632;</span> "
            f"<b>{metal['id']}</b>: transition metal</li>"
        )
    for component in proposal["components"]:
        color = color_by_component[component["id"]]
        status = "coordinated" if component["coordinated"] else "not coordinated"
        composition = " ".join(
            f"{element}{count}" if count != 1 else element
            for element, count in component["elements"].items()
        )
        donors = ", ".join(str(value) for value in component["donor_atom_ids"]) or "none"
        legend_lines.append(
            f"<li><span style=\"color:{color}\">&#9632;</span> "
            f"<b>{component['id']}</b>: {status}; {component['n_atoms']} atoms; "
            f"{composition}; denticity {component['denticity']}; "
            f"donor atoms: {donors}</li>"
        )

    view.zoomTo()
    body = view._make_html()
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\"><title>Coordination fragment proposal</title></head>\n"
        "<body style=\"font-family: sans-serif;\">\n"
        f"<h3 style=\"margin: 8px 0;\">Coordination fragment proposal | {title}</h3>\n"
        "<p>Distance-based proposal. Review coordination and component boundaries "
        "before using them in the population analysis.</p>\n"
        "<ul style=\"line-height:1.5;\">\n"
        + "\n".join(legend_lines)
        + "\n</ul>\n"
        + body
        + "\n</body>\n</html>\n"
    )
    with open(output_path, "w") as out:
        out.write(html)
    print(f"[OK] Coordination-fragment viewer saved to '{output_path}'.")


def find_orca_geometry_file_for_viewer(orca_files):
    """
    Return one ORCA output containing a Cartesian coordinate block.

    The viewer only needs a representative snapshot, so prefer the first file
    in the sorted ORCA list and only scan further if that one lacks geometry.
    """
    for fname in orca_files:
        try:
            parse_orca_cartesian_coordinates(fname)
        except Exception:
            continue
        return fname
    return None


def write_spin_localization_viewer_from_atoms(
    title,
    atoms,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
    label_all_atoms=False,
):
    """
    Write a py3Dmol-based HTML viewer from parsed atoms.
    """
    import py3Dmol

    highlighted = set(highlighted_atom_ids)

    view = py3Dmol.view(width=980, height=720)
    view.addModel(atoms_to_mol_block(title, atoms), "mol")
    view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})

    for atom in atoms:
        aid = atom["index"]
        if not label_all_atoms and aid not in highlighted:
            continue
        element = atom_types.get(aid, atom["element"])
        if aid in highlighted and aid in avg_fraction_by_atom:
            frac = 100.0 * float(avg_fraction_by_atom.get(aid, 0.0))
            label_text = f"{aid} {element} {frac:.1f}%"
            font_size = 15
        else:
            label_text = f"{aid} {element}"
            font_size = 12
        view.addLabel(
            label_text,
            {
                "position": {"x": atom["x"], "y": atom["y"], "z": atom["z"]},
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.9,
                "fontSize": font_size,
                "inFront": True,
            },
        )

    view.zoomTo()
    body = view._make_html()
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\"><title>Spin localization viewer</title></head>\n"
        "<body>\n"
        f"<h3 style=\"font-family: sans-serif; margin: 8px 0;\">Spin localization viewer | {title}</h3>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
    with open(output_path, "w") as out:
        out.write(html)
    print(f"[OK] Spin-localization viewer saved to '{output_path}'.")


def atoms_to_mol_block(title, atoms):
    """
    Build a V2000 MOL block with explicit bonds for py3Dmol.
    """
    bonds = infer_bonds_from_distances(atoms)
    lines = [
        str(title)[:80],
        "TolkienTools",
        "",
        f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0            999 V2000",
    ]
    for atom in atoms:
        lines.append(
            f"{atom['x']:10.4f}{atom['y']:10.4f}{atom['z']:10.4f} "
            f"{atom['element'][:3]:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for i, j in bonds:
        lines.append(f"{i + 1:3d}{j + 1:3d}  1  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def infer_bonds_from_distances(atoms):
    """
    Infer conservative single bonds from Cartesian distances.
    """
    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "P": 1.07,
        "Fe": 1.24,
        "Ru": 1.46,
        "Cu": 1.32,
        "Zn": 1.22,
    }
    max_coordination = {
        "H": 1,
        "C": 4,
        "N": 4,
        "O": 2,
        "S": 6,
        "P": 5,
        "Fe": 6,
        "Ru": 6,
        "Cu": 5,
        "Zn": 5,
    }
    candidates = []
    for i, atom_i in enumerate(atoms):
        elem_i = normalize_element_symbol(atom_i["element"])
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            elem_j = normalize_element_symbol(atom_j["element"])
            dist = distance_between_atoms(atom_i, atom_j)
            if is_plausible_bond(elem_i, elem_j, dist, covalent_radii):
                candidates.append((dist, i, j))

    bonds = []
    degree = [0] * len(atoms)
    for _dist, i, j in sorted(candidates):
        elem_i = normalize_element_symbol(atoms[i]["element"])
        elem_j = normalize_element_symbol(atoms[j]["element"])
        if degree[i] >= max_coordination.get(elem_i, 4):
            continue
        if degree[j] >= max_coordination.get(elem_j, 4):
            continue
        bonds.append((i, j))
        degree[i] += 1
        degree[j] += 1
    return bonds


def normalize_element_symbol(element):
    """
    Normalize element capitalization for simple bonding/color rules.
    """
    element = str(element).strip()
    if not element:
        return "X"
    atomic_number_to_symbol = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        44: "Ru",
    }
    try:
        atomic_number = int(element)
    except ValueError:
        atomic_number = None
    if atomic_number is not None:
        return atomic_number_to_symbol.get(atomic_number, element)
    return element[0].upper() + element[1:].lower()


def distance_between_atoms(atom_i, atom_j):
    dx = atom_i["x"] - atom_j["x"]
    dy = atom_i["y"] - atom_j["y"]
    dz = atom_i["z"] - atom_j["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_plausible_bond(elem_i, elem_j, dist, covalent_radii):
    if dist < 0.35:
        return False
    if elem_i == "H" and elem_j == "H":
        return dist <= 0.85

    metal_elements = {"Fe", "Ru", "Cu", "Zn"}
    if elem_i in metal_elements or elem_j in metal_elements:
        ligand = elem_j if elem_i in metal_elements else elem_i
        if ligand == "H":
            return False
        return dist <= 2.65

    ri = covalent_radii.get(elem_i, 0.77)
    rj = covalent_radii.get(elem_j, 0.77)
    return dist <= 1.20 * (ri + rj) + 0.20
