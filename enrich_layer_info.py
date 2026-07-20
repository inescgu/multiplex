#!/usr/bin/env python3
"""
enrich_layer_info.py
====================
Add neuron group categories to layer_info.json based on cell type names.
"""
import json
import argparse
from pathlib import Path
import pandas as pd

NEURON_GROUPS = {
    "Proprioceptors": [
        "PVALB+ RUNX3+ proprioceptor 1",
        "PVALB+ RUNX3+ proprioceptor 2",
        "PVALB+ RUNX3+ proprioceptor 3",
    ],
    "A-fiber LTMRs": [
        "NTRK3+ NTRK2+ A-beta LTMR subtype 1",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 2",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 3",
        "NTRK3+ NTRK2+ A-beta LTMR subtype 4",
        "A-LTMR",
    ],
    "Peptidergic nociceptors": [
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 1",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 2",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 3",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 4",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 5",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 6",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 7",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 8",
        "CALCA- TAC1- NTRK1- polymodal peptidergic subtype 9",
        "NP SST positive subtype",
    ],
    "C-fiber nociceptors": [
        "MRGPRX polymodal non-canonical peptidergic nociceptor 1",
        "MRGPRX polymodal non-canonical peptidergic nociceptor 2",
    ],
    "C-fiber LTMRs": [
        "C-LTMR subtype",
    ],
    "Thermoreceptors": [
        "Thermo sensitive population 1",
        "Thermo sensitive population 2",
    ],
    "Other": [
        "ATF3",
    ]
}

CELLTYPE_TO_GROUP = {
    cell: group for group, cells in NEURON_GROUPS.items() for cell in cells
}

def sanitize_name(name: str) -> str:
    """Convert cell type name to safe filename (matches stem_from_cell_type)."""
    return name.replace(" ", "_").replace("/", "-").replace(",", "")

def main():
    parser = argparse.ArgumentParser(description="Enrich layer_info.json with neuron group metadata")
    parser.add_argument("--wdir", required=True, help="MultiXrank working directory")
    parser.add_argument("--csv", default="layers_categories.csv", help="Cell type categories CSV")
    args = parser.parse_args()

    wdir = Path(args.wdir).resolve()
    layer_info_path = wdir / "layer_info.json"

    # Load categories CSV to get full cell type names
    csv_path = Path(args.csv) if Path(args.csv).exists() else wdir.parent / args.csv
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return

    categories_df = pd.read_csv(csv_path)
    categories_df.columns = [c.strip() for c in categories_df.columns]
    if categories_df.columns[0] in ("", "Unnamed: 0"):
        categories_df = categories_df.iloc[:, 1:]

    celltype_col = "Cell Type"
    if celltype_col not in categories_df.columns:
        print(f"❌ Expected column '{celltype_col}' not found in CSV")
        return

    # Load layer_info.json
    if not layer_info_path.exists():
        print(f"❌ Layer info not found: {layer_info_path}")
        return

    with open(layer_info_path, "r") as f:
        layer_info = json.load(f)

    # Build mapping from sanitized name to full cell type name + group
    celltype_lookup = {}
    for _, row in categories_df.iterrows():
        full_name = row[celltype_col].strip()
        safe_name = sanitize_name(full_name)
        group = CELLTYPE_TO_GROUP.get(full_name, "Other")
        celltype_lookup[safe_name] = {
            "full_name": full_name,
            "group": group,
        }

    # Enrich layer_info entries
    enriched = 0
    for layer_name, layer_data in layer_info.items():
        # Skip special layers
        if layer_data.get("type") == "special":
            continue

        # Extract cell type from layer name (assuming format: <celltype>_ppi_layer or <celltype>_inter_layer)
        if "_inter_layer" in layer_name:
            # Inter-layer: <ct1>_<ct2>_inter_layer
            # For now, skip or handle specially
            continue
        elif "_ppi_layer" in layer_name:
            safe_name = layer_name.replace("_ppi_layer", "")
        elif "_layer" in layer_name:
            safe_name = layer_name.replace("_layer", "")
        else:
            safe_name = layer_name

        # Look up the cell type
        if safe_name in celltype_lookup:
            info = celltype_lookup[safe_name]
            layer_data["cell_type_full"] = info["full_name"]
            layer_data["major_category"] = info["group"]
            layer_data["minor_category"] = info["group"]  # You can refine this if needed
            enriched += 1

    # Save enriched layer_info
    with open(layer_info_path, "w") as f:
        json.dump(layer_info, f, indent=2)

    print(f"✅ Enriched {enriched} layers with category metadata")
    print(f"   Updated: {layer_info_path}")

if __name__ == "__main__":
    main()