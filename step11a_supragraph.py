#!/usr/bin/env python3
"""
Step 11a: Build the heterogeneous supra-graph from a MultiXrank working
directory (the output of your step 1 + step 2 scripts), for downstream
HiDeF-style multi-resolution clustering WITHOUT collapsing cell types
into one shared gene namespace.

Why a supra-graph and not "true" leidenalg multiplex layers
--------------------------------------------------------------
leidenalg's classic multiplex API (optimise_partition_multiplex) requires
every layer to share the SAME vertex set -- not your case, since each cell
type has its own (overlapping but distinct) gene space. leidenalg does
have a more general "slices" formulation that allows non-identical vertex
sets coupled through an explicit coupling graph, but building and
maintaining that construction adds real complexity for not much practical
gain over the approach used here.

Instead, every node is namespaced as "<cell_type>::GENE", so:
  - "EGFR" in Fibroblast and "EGFR" in Enteric_Neuron are DISTINCT nodes
    (nothing is merged/flattened across cell types)
  - step 1's intra-cell-type PPI edges become "intra" edges within a
    cell type's own node block
  - step 2's bipartite edges (flat LR, Reactome, scored LIANA/NeuronChat
    LR, global_ppi identity edges -- all already merged per cell-type-pair
    by step 2) become "bipartite" edges connecting two cell types' blocks
  - optionally, additional pure identity-coupling edges (same gene, any
    two cell types that both express it) can be added as a THIRD edge
    type, giving you a coupling-strength knob independent of the
    biological bipartite edges already present

This single weighted, edge-typed igraph.Graph is exactly the structure
that Mucha et al.'s multislice modularity (Science 2010) optimizes over --
running Leiden/Louvain on it across a resolution sweep (step 3b) and
feeding the resulting partitions into HiDeF's weaver (step 3c) reproduces
the persistence-hierarchy idea from your monoplex HiDeF run, just applied
to the full heterogeneous structure instead of one flattened PPI graph.

Output: a pickled igraph.Graph with vertex attributes 'name' ("ct::gene"),
'cell_type', 'gene', and edge attributes 'weight', 'etype' in
{'intra', 'bipartite', 'identity'}.
"""
import argparse
import json
import pickle
from pathlib import Path

import igraph as ig
import pandas as pd

def _maybe_namespace(gene: str, ct: str) -> str:
    """Namespace a gene as 'cell_type::gene', but only if it isn't already
    namespaced. Needed because this wdir may have already been through
    step 3.5 (which renames nodes in ppi_layer.tsv / bipartite/*.tsv to
    'ct::gene' in place, for MultiXrank's own benefit) before this script
    runs -- without this guard, an already-namespaced gene column gets a
    SECOND 'ct::' prefix stapled on top, e.g.
    'global_ppi::global_ppi::A1BG'."""
    gene = str(gene)
    return gene if "::" in gene else f"{ct}::{gene}"

def load_intra_layer_edges(wdir: Path):
    mul_dir = wdir / "multiplex"
    if not mul_dir.exists():
        raise FileNotFoundError(f"{mul_dir} not found -- run step 1 first.")
    cell_types = sorted([p.name for p in mul_dir.iterdir() if p.is_dir()])
    edges = []
    print(f"Loading {len(cell_types)} intra-cell-type PPI layer(s) ...")
    for ct in cell_types:
        fpath = mul_dir / ct / "ppi_layer.tsv"
        if not fpath.exists():
            print(f"  ⚠ {fpath} missing, skipping")
            continue
        df = pd.read_csv(fpath, sep="\t", header=None, names=["g1", "g2"])
        n_self = int((df["g1"] == df["g2"]).sum())
        # drop step1's placeholder self-loop (written only when a cell type
        # had zero real PPI edges, so MultiXrank wouldn't choke on an
        # empty layer file -- see step1's write_ppi_layer())
        df = df[df["g1"] != df["g2"]]
        for g1, g2 in zip(df["g1"], df["g2"]):
            edges.append((_maybe_namespace(g1, ct), _maybe_namespace(g2, ct), 1.0, "intra"))
        note = f" ({n_self} placeholder self-loop skipped)" if n_self else ""
        print(f"  {ct}: {len(df):,} intra-layer edges{note}")
    print()
    return edges, cell_types


def load_bipartite_edges(wdir: Path):
    manifest_path = wdir / "bipartite_manifest.json"
    if not manifest_path.exists():
        print("No bipartite_manifest.json found -- no cross-cell-type edges will be added "
              "(did you run step 2?).\n")
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    edges = []
    print(f"Loading {len(manifest)} bipartite edge file(s) ...")
    for entry in manifest:
        fpath = wdir / entry["file"]
        ct_a, ct_b = entry["source"], entry["target"]
        df = pd.read_csv(fpath, sep="\t", header=None, names=["g1", "g2"])
        # step 2 always orients (pair_key, edge) together so g1 belongs to
        # ct_a's gene space and g2 to ct_b's -- see step2's "FIX" note in
        # add_cross_celltype_edges()/add_scored_lr_edges().
        for g1, g2 in zip(df["g1"], df["g2"]):
            edges.append((_maybe_namespace(g1, ct_a), _maybe_namespace(g2, ct_b), 1.0, "bipartite"))
        print(f"  {ct_a} <-> {ct_b}: {len(df):,} bipartite edges")
    print()
    return edges


def load_identity_coupling_edges(wdir: Path, cell_types_present: set):
    """OPTIONAL. Connects the SAME gene across every pair of cell types that
    both express it, regardless of whether a biological bipartite edge
    (LR/Reactome/scored-LR/global_ppi identity) already links them. This is
    the generic Mucha-et-al. notion of interlayer 'coupling' -- a purely
    structural 'this is the same entity in another layer' link, separate
    from anything biological. Off by default because with --base-ppi's
    global_ppi identity edges from step 2 already wired in, most shared
    genes already have SOME cross-layer path; this adds a much denser
    additional identity layer on top, useful for a full multislice-style
    omega sweep but O(sum_gene |celltypes(gene)|^2) so can get large.
    """
    gene_sets_path = wdir / "celltype_gene_sets.json"
    with open(gene_sets_path) as f:
        gene_sets = json.load(f)
    gene_sets = {ct: set(genes) for ct, genes in gene_sets.items() if ct in cell_types_present}

    gene_to_cts: dict[str, list] = {}
    for ct, genes in gene_sets.items():
        for g in genes:
            gene_to_cts.setdefault(g, []).append(ct)

    edges = []
    for g, cts in gene_to_cts.items():
        if len(cts) < 2:
            continue
        cts_sorted = sorted(cts)
        for i in range(len(cts_sorted)):
            for j in range(i + 1, len(cts_sorted)):
                edges.append((f"{cts_sorted[i]}::{g}", f"{cts_sorted[j]}::{g}", 1.0, "identity"))
    print(f"Identity coupling: {len(edges):,} same-gene cross-cell-type edges added\n")
    return edges


def build_supra_graph(wdir: Path, add_identity_coupling: bool = False) -> ig.Graph:
    intra_edges, cell_types = load_intra_layer_edges(wdir)
    bip_edges = load_bipartite_edges(wdir)
    all_edges = intra_edges + bip_edges

    if add_identity_coupling:
        all_edges += load_identity_coupling_edges(wdir, set(cell_types))

    if not all_edges:
        raise SystemExit("ERROR: no edges loaded at all -- check --wdir points at a completed "
                          "step 1 (+ step 2) output directory.")

    nodes = sorted({u for u, v, w, e in all_edges} | {v for u, v, w, e in all_edges})
    node_idx = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes
    g.vs["cell_type"] = [n.split("::", 1)[0] for n in nodes]
    g.vs["gene"] = [n.split("::", 1)[1] for n in nodes]

    g.add_edges([(node_idx[u], node_idx[v]) for u, v, w, e in all_edges])
    g.es["weight"] = [w for u, v, w, e in all_edges]
    g.es["etype"] = [e for u, v, w, e in all_edges]

    n_intra = sum(1 for e in g.es["etype"] if e == "intra")
    n_bip = sum(1 for e in g.es["etype"] if e == "bipartite")
    n_id = sum(1 for e in g.es["etype"] if e == "identity")
    print("=" * 60)
    print(f"Supra-graph built: {g.vcount():,} nodes across {len(cell_types)} cell type(s)")
    print(f"  {g.ecount():,} edges total ({n_intra:,} intra, {n_bip:,} bipartite, {n_id:,} identity)")
    print("=" * 60)
    return g


def main():
    ap = argparse.ArgumentParser(description="Step 3a: build a heterogeneous multiplex supra-graph "
                                               "from a MultiXrank working dir")
    ap.add_argument("--wdir", required=True, help="MultiXrank working dir (output of step 1 + step 2)")
    ap.add_argument("--output", required=True, help="Path to write the supra-graph (.pkl, igraph.Graph)")
    ap.add_argument("--add-identity-coupling", action="store_true",
                     help="Also add same-gene cross-cell-type identity coupling edges (see docstring)")
    args = ap.parse_args()

    wdir = Path(args.wdir)
    g = build_supra_graph(wdir, add_identity_coupling=args.add_identity_coupling)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(g, f)
    print(f"\nSaved supra-graph -> {out_path}")


if __name__ == "__main__":
    main()