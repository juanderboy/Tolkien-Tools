#!/usr/bin/env python3
"""Distance-based proposals for coordination-complex fragments."""

from collections import Counter

from charge_spin_viewer import (
    distance_between_atoms,
    infer_bonds_from_distances,
    normalize_element_symbol,
)


TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

COORDINATION_RADII = {
    "Sc": 1.44, "Ti": 1.32, "V": 1.22, "Cr": 1.18, "Mn": 1.17,
    "Fe": 1.17, "Co": 1.16, "Ni": 1.15, "Cu": 1.17, "Zn": 1.25,
    "Y": 1.62, "Zr": 1.45, "Nb": 1.34, "Mo": 1.30, "Tc": 1.27,
    "Ru": 1.25, "Rh": 1.25, "Pd": 1.28, "Ag": 1.34, "Cd": 1.48,
    "Hf": 1.44, "Ta": 1.34, "W": 1.30, "Re": 1.28, "Os": 1.26,
    "Ir": 1.27, "Pt": 1.30, "Au": 1.34, "Hg": 1.49,
}

DONOR_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "I": 1.39,
}


def is_transition_metal(element):
    return normalize_element_symbol(element) in TRANSITION_METALS


def coordination_cutoff(metal_element, donor_element):
    """Return a permissive covalent-radii cutoff for a metal--donor contact."""
    metal_radius = COORDINATION_RADII.get(normalize_element_symbol(metal_element), 1.30)
    donor_radius = DONOR_RADII.get(normalize_element_symbol(donor_element), 0.77)
    return 1.20 * (metal_radius + donor_radius) + 0.18


def infer_metal_contacts(atoms):
    """Return distance-based metal contacts as dictionaries."""
    contacts = []
    metals = [atom for atom in atoms if is_transition_metal(atom["element"])]
    for metal in metals:
        for atom in atoms:
            if atom["model_index"] == metal["model_index"]:
                continue
            if is_transition_metal(atom["element"]):
                continue
            element = normalize_element_symbol(atom["element"])
            if element == "H":
                continue
            distance = distance_between_atoms(metal, atom)
            cutoff = coordination_cutoff(metal["element"], element)
            if 0.8 < distance <= cutoff:
                contacts.append(
                    {
                        "metal_id": metal["index"],
                        "metal_element": normalize_element_symbol(metal["element"]),
                        "atom_id": atom["index"],
                        "atom_element": element,
                        "distance": distance,
                    }
                )
    return contacts


def _connected_components(atoms, bonds):
    adjacency = {atom["index"]: set() for atom in atoms}
    model_to_id = {atom["model_index"]: atom["index"] for atom in atoms}
    for model_i, model_j in bonds:
        atom_i = model_to_id[model_i]
        atom_j = model_to_id[model_j]
        adjacency[atom_i].add(atom_j)
        adjacency[atom_j].add(atom_i)

    components = []
    seen = set()
    for atom_id in sorted(adjacency):
        if atom_id in seen:
            continue
        stack = [atom_id]
        seen.add(atom_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def propose_coordination_fragments(atoms):
    """Identify metals and molecular components, annotated by coordination."""
    atoms = list(atoms)
    metals = [atom for atom in atoms if is_transition_metal(atom["element"])]
    if not metals:
        return {"metals": [], "components": [], "contacts": []}

    nonmetals = [atom for atom in atoms if not is_transition_metal(atom["element"])]
    local_nonmetal_bonds = infer_bonds_from_distances(nonmetals)
    nonmetal_bonds = [
        (nonmetals[local_i]["model_index"], nonmetals[local_j]["model_index"])
        for local_i, local_j in local_nonmetal_bonds
    ]
    components = _connected_components(nonmetals, nonmetal_bonds)
    contacts = infer_metal_contacts(atoms)
    atom_by_id = {atom["index"]: atom for atom in atoms}

    records = []
    for atom_ids in components:
        component_set = set(atom_ids)
        component_contacts = [
            contact for contact in contacts if contact["atom_id"] in component_set
        ]
        elements = Counter(
            normalize_element_symbol(atom_by_id[atom_id]["element"])
            for atom_id in atom_ids
        )
        records.append(
            {
                "atom_ids": atom_ids,
                "n_atoms": len(atom_ids),
                "elements": dict(sorted(elements.items())),
                "contacts": component_contacts,
                "donor_atom_ids": sorted({item["atom_id"] for item in component_contacts}),
                "metal_ids": sorted({item["metal_id"] for item in component_contacts}),
                "denticity": len({item["atom_id"] for item in component_contacts}),
                "coordinated": bool(component_contacts),
            }
        )

    records.sort(
        key=lambda item: (
            not item["coordinated"],
            -item["n_atoms"],
            min(item["atom_ids"]),
        )
    )
    for index, record in enumerate(records, start=1):
        record["id"] = f"L{index}"
        record["label"] = f"L{index}"

    metal_records = [
        {
            "id": f"{normalize_element_symbol(atom['element'])}{atom['index']}",
            "label": f"{normalize_element_symbol(atom['element'])}{atom['index']}",
            "atom_ids": [atom["index"]],
            "atom_id": atom["index"],
            "element": normalize_element_symbol(atom["element"]),
        }
        for atom in metals
    ]
    return {"metals": metal_records, "components": records, "contacts": contacts}


def proposal_named_groups(proposal):
    """Return case-insensitive tokens usable in fragment definitions."""
    groups = {}
    for metal in proposal["metals"]:
        groups[metal["id"].lower()] = list(metal["atom_ids"])
    for component in proposal["components"]:
        groups[component["id"].lower()] = list(component["atom_ids"])
    return groups


def format_element_composition(elements):
    return " ".join(
        f"{element}{count}" if count != 1 else element
        for element, count in elements.items()
    )


def write_coordination_proposal_report(
    proposal,
    outname="coordination_ligand_proposal.dat",
):
    """Write a human-readable and machine-friendly proposal report."""
    with open(outname, "w") as out:
        out.write("# Distance-based coordination-fragment proposal\n")
        out.write("# Review the interactive viewer before accepting these groups.\n")
        for metal in proposal["metals"]:
            out.write(
                f"METAL {metal['id']} atom={metal['atom_id']} "
                f"element={metal['element']}\n"
            )
        for component in proposal["components"]:
            status = "coordinated" if component["coordinated"] else "not_coordinated"
            donors = ",".join(str(value) for value in component["donor_atom_ids"]) or "-"
            metals = ",".join(str(value) for value in component["metal_ids"]) or "-"
            atoms = ",".join(str(value) for value in component["atom_ids"])
            out.write(
                f"COMPONENT {component['id']} status={status} "
                f"n_atoms={component['n_atoms']} denticity={component['denticity']} "
                f"metals={metals} donors={donors} "
                f"elements={format_element_composition(component['elements'])} "
                f"atoms={atoms}\n"
            )
        out.write("# metal metal_atom donor donor_atom distance_angstrom\n")
        for contact in proposal["contacts"]:
            out.write(
                f"CONTACT {contact['metal_element']} {contact['metal_id']} "
                f"{contact['atom_element']} {contact['atom_id']} "
                f"{contact['distance']:.4f}\n"
            )
    print(f"[OK] Coordination proposal saved to '{outname}'.")
