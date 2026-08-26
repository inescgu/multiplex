#!/usr/bin/env python3
"""
Step 1 (replaces old step3's intra-layer half): Build per-cell-type PPI
multiplex layers for a HETEROGENEOUS MultiXrank structure.

Why this looks so much smaller than the old step3
---------------------------------------------------
The old step3 built one giant homogeneous nx.Graph where every (source,
cell_type, gene) triple got its own integer node id via a NodeRegistry —
that machinery existed *only* to fake per-cell-type node identity inside a
single graph. In a true heterogeneous MultiXrank structure, each cell type
is its own multiplex block, so it gets that separation for free: a gene
called "EGFR" in "Enteric_Neuron" and a gene called "EGFR" in "Fibroblast"
are automatically distinct nodes in two different node spaces. No registry,
no combined pickle, no annotation-merging logic needed.

You're hdrg-only, so this also drops all the tabula_sapiens loading/merging
code from the old pipeline.

What this script does
----------------------
1. Loads every *_layer.pkl in --hdrg-layers-dir (one per cell type).
2. Optionally filters/merges cell types using a categories CSV, exactly like
   the old step5 did — but *before* ever touching disk, so there's no
   write-then-rewrite step.
3. Optionally adds the full comprehensive PPI network as an extra
   pseudo-cell-type multiplex (--base-ppi), same as the old --base-ppi flag.
   This is added AFTER CSV filtering/merging (see note below) so it always
   survives as its own standalone multiplex, and step 2 can connect it to
   every other cell type via identity bipartite edges.
4. Writes multiplex/<cell_type>/ppi_layer.tsv for every surviving cell type.
5. Saves celltype_gene_sets.json (cell_type -> sorted gene list) — step 2
   needs this to know which genes live in which cell type when building
   ligand-receptor / pathway bipartite edges, and to build the identity
   bipartite edges connecting --base-ppi to every other cell type.

Output layout (under --output-wdir, i.e. the MultiXrank working dir):
    multiplex/<cell_type>/ppi_layer.tsv
    celltype_gene_sets.json
    cell_type_metadata.json

Note on --base-ppi ordering
-----------------------------
--base-ppi is intentionally added to `layers` AFTER --layers-csv
filtering/merging, not before. If it were added first, a --layers-csv run
would silently drop it (it would need to appear as its own row in the CSV's
"Cell Type" column to survive apply_csv_filter_and_merge, and if --merge-by
were set it could even get folded into some unrelated broad-category
group). Adding it after guarantees it always comes through as its own
untouched multiplex regardless of what the CSV says, since it isn't a real
hdrg cell type — it's the comprehensive backbone network step 2 wires
everything else to.
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_hdrg_layers(layers_dir: str) -> dict[str, nx.Graph]:
    layers_dir = Path(layers_dir)
    pkl_files = sorted(layers_dir.glob("*_layer.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *_layer.pkl files found in {layers_dir}")

    layers = {}
    print(f"Loading {len(pkl_files)} hdrg layer file(s) from {layers_dir} ...")
    for fpath in pkl_files:
        import pickle
        with open(fpath, "rb") as f:
            g = pickle.load(f)
        mapping = {n: str(n).strip().upper() for n in g.nodes()}
        g = nx.relabel_nodes(g, mapping)
        cell_type = fpath.stem.replace("_layer", "")
        layers[cell_type] = g
        print(f"  {cell_type}: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    print()
    return layers


def load_base_ppi(ppi_file: str) -> nx.Graph:
    import pickle
    print(f"Loading base/comprehensive PPI from {ppi_file} ...")
    with open(ppi_file, "rb") as f:
        G = pickle.load(f)
    G = nx.relabel_nodes(G, {n: str(n).strip().upper() for n in G.nodes()})
    print(f"  global_ppi: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")
    return G


# ---------------------------------------------------------------------------
# CSV-driven filtering / merging (adapted from old step5)
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace(",", "")


def load_categories(csv_path: Path, exclude_broad=None, exclude_ai=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]

    required = {"Cell Type", "Broad Category - ai", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"layers_categories.csv missing columns: {missing}. Found: {list(df.columns)}")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    before = len(df)
    if exclude_broad:
        mask = df["Broad Category - ai"].str.lower().isin([e.lower() for e in exclude_broad])
        df = df[~mask]
        print(f"Excluded {before - len(df)} rows by Broad Category ({exclude_broad})")
        before = len(df)
    if exclude_ai:
        mask = df["category"].str.lower().isin([e.lower() for e in exclude_ai])
        df = df[~mask]
        print(f"Excluded {before - len(df)} rows by ai category ({exclude_ai})")

    print(f"Retained {len(df)} cell types from CSV\n")
    return df.reset_index(drop=True)


def apply_csv_filter_and_merge(
    layers: dict[str, nx.Graph],
    csv_path: str,
    exclude_broad: list,
    exclude_ai: list,
    merge_by: str,
) -> dict[str, nx.Graph]:
    """Filters `layers` down to CSV-allowed cell types, optionally merging
    (union of edges) within Broad Category / ai category groups. Returns a
    new dict of cell_type -> nx.Graph, ready to write out."""
    df = load_categories(Path(csv_path), exclude_broad or None, exclude_ai or None)

    # map sanitized CSV cell-type name -> raw graph key (graph keys are the
    # pkl filename stems, which should already be the sanitized form)
    available = {sanitize_name(k): k for k in layers.keys()}
    allowed_sanitized = {sanitize_name(ct) for ct in df["Cell Type"]}

    not_found = allowed_sanitized - set(available)
    if not_found:
        print(f"⚠ {len(not_found)} CSV cell types not found among loaded layers (skipped):")
        for s in sorted(not_found):
            print(f"    {s}")

    out: dict[str, nx.Graph] = {}

    if merge_by == "none":
        for sanitized, raw_key in available.items():
            if sanitized in allowed_sanitized:
                out[raw_key] = layers[raw_key]
        return out

    group_col = "Broad Category - ai" if merge_by == "broad" else "category"
    print(f"Merging cell types by '{group_col}' ...")
    for group, sub in df.groupby(group_col):
        merged_name = sanitize_name(group) + "_merged"
        H = nx.Graph()
        n_sources = 0
        for ct in sub["Cell Type"]:
            raw_key = available.get(sanitize_name(ct))
            if raw_key is None:
                continue
            H.add_edges_from(layers[raw_key].edges())
            n_sources += 1
        if H.number_of_edges() == 0:
            print(f"  ⚠ {merged_name}: no edges found, skipping")
            continue
        out[merged_name] = H
        print(f"  {merged_name}: {n_sources} source cell types -> "
              f"{H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")
    print()
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_ppi_layer(G: nx.Graph, path: Path) -> int:
    edges = [(a, b) for a, b in G.edges() if a != b]
    if not edges:
        placeholder = next(iter(G.nodes())) if G.number_of_nodes() else "PLACEHOLDER"
        print(f"    ⚠ {path.name} has no edges — writing placeholder self-loop so MultiXrank doesn't choke")
        edges = [(placeholder, placeholder)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["g1", "g2"]).to_csv(path, sep="\t", index=False, header=False)
    return len(edges)


def main():
    ap = argparse.ArgumentParser(description="Build per-cell-type PPI multiplex layers (hdrg-only, heterogeneous structure)")
    ap.add_argument("--hdrg-layers-dir", required=True)
    ap.add_argument("--output-wdir", required=True, help="MultiXrank working directory root")
    ap.add_argument("--base-ppi", default=None, help="Optional comprehensive PPI pkl, added as an extra 'global_ppi' cell-type multiplex")
    ap.add_argument("--layers-csv", default=None, help="Optional cell-type selection/merge CSV (Cell Type, Broad Category - ai, category)")
    ap.add_argument("--exclude-broad", nargs="+", default=[])
    ap.add_argument("--exclude-ai", nargs="+", default=[])
    ap.add_argument("--merge-by", choices=["none", "broad", "ai"], default="none")
    args = ap.parse_args()

    out = Path(args.output_wdir)
    mul_dir = out / "multiplex"
    mul_dir.mkdir(parents=True, exist_ok=True)

    layers = load_hdrg_layers(args.hdrg_layers_dir)

    if args.layers_csv:
        layers = apply_csv_filter_and_merge(
            layers, args.layers_csv, args.exclude_broad, args.exclude_ai, args.merge_by
        )
        if not layers:
            raise SystemExit("ERROR: no cell types survived CSV filtering — check your CSV/exclude flags.")

    # Added AFTER CSV filtering/merging on purpose — see module docstring.
    # global_ppi isn't a real hdrg cell type subject to the CSV's
    # include/exclude/merge rules; it's the comprehensive backbone network
    # that step 2 connects to every surviving cell type via identity
    # bipartite edges, so it must always come through untouched.
    if args.base_ppi:
        layers["global_ppi"] = load_base_ppi(args.base_ppi)

    print("Writing per-cell-type PPI layers ...")
    gene_sets: dict[str, list] = {}
    metadata: dict[str, dict] = {}
    for cell_type, G in sorted(layers.items()):
        safe_name = sanitize_name(cell_type)
        rel_path = Path("multiplex") / safe_name / "ppi_layer.tsv"
        n_edges = write_ppi_layer(G, out / rel_path)
        genes = sorted(G.nodes())
        gene_sets[safe_name] = genes
        metadata[safe_name] = {"n_nodes": len(genes), "n_ppi_edges": n_edges, "layer_path": str(rel_path)}
        print(f"  [{safe_name}]: {len(genes):,} genes, {n_edges:,} PPI edges -> {rel_path}")
    print()

    with open(out / "celltype_gene_sets.json", "w") as f:
        json.dump(gene_sets, f)
    with open(out / "cell_type_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"Step 1 complete: {len(gene_sets)} cell-type multiplex(es) written")
    print(f"  wdir: {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()