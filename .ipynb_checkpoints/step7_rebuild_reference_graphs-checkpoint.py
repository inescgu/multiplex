#!/usr/bin/env python3
"""
Step 7: Rebuild reference graphs based on the surviving consolidated files.
"""
import argparse
import pickle
from pathlib import Path
import json
import networkx as nx
import pandas as pd
import yaml

def load_surviving_layers(wdir: Path) -> dict[str, pd.DataFrame]:
    config_path = wdir / "config.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    layer_paths: list[str] = cfg["multiplex"]["gene_multiplex"]["layers"]
    layers: dict[str, pd.DataFrame] = {}
    print(f"Loading {len(layer_paths)} surviving layers…")
    for rel_path in sorted(layer_paths):
        abs_path = wdir / rel_path
        stem = Path(rel_path).stem
        if not abs_path.exists():
            print(f"  ⚠ Missing: {rel_path} — skipped")
            continue
        df = pd.read_csv(abs_path, sep="\t", header=None, names=["g1", "g2"], dtype=str)
        df["g1"] = df["g1"].str.strip().str.upper()
        df["g2"] = df["g2"].str.strip().str.upper()
        layers[stem] = df
    return layers

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdir", required=True)
    args = parser.parse_args()
    wdir = Path(args.wdir)

    layers = load_surviving_layers(wdir)

    # Classify
    layer_info_path = wdir / "layer_info.json"
    with open(layer_info_path) as f:
        layer_info = json.load(f)

    ppi_stems = {s for s in layers.keys() if layer_info.get(s, {}).get("edge_type") == "ppi"}
    inter_stems = {s for s in layers.keys() if layer_info.get(s, {}).get("type") in ("inter", "special")}

    # Build graphs
    ref = nx.Graph()
    ppi = nx.Graph()

    def ensure(G, a, b):
        if not G.has_edge(a, b):
            G.add_edge(a, b, cell_types=set(), inter_cell_types=set(), edge_type=None)

    for stem, df in layers.items():
        is_ppi = stem in ppi_stems
        is_inter = stem in inter_stems
        for _, row in df.iterrows():
            a, b = sorted([row["g1"], row["g2"]])
            if a == b:
                continue
            
            ensure(ref, a, b)
            if is_ppi:
                ref[a][b]["cell_types"].add(stem)
                ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "inter" else "ppi"
                ppi.add_edge(a, b)
            elif is_inter:
                ref[a][b]["inter_cell_types"].add(stem)
                ref[a][b]["edge_type"] = "both" if ref[a][b]["edge_type"] == "ppi" else "inter"

    # Save
    ref_path = wdir / "gene_reference_graph.pkl"
    ppi_path = wdir / "gene_ppi_reference_graph.pkl"

    with open(ref_path, "wb") as f:
        pickle.dump(ref, f, protocol=4)
    with open(ppi_path, "wb") as f:
        pickle.dump(ppi, f, protocol=4)

    print(f"✓ Rebuilt Reference Graphs:\n  Reference: {ref.number_of_nodes():,} nodes | {ref.number_of_edges():,} edges")
    print(f"  PPI-Only:  {ppi.number_of_nodes():,} nodes | {ppi.number_of_edges():,} edges")

if __name__ == "__main__":
    main()