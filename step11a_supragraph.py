#!/usr/bin/env python3
"""
Step 11a: Build the heterogeneous supra-graph from a MultiXrank working
directory (the output of your step 1 + step 2 scripts), for downstream
HiDeF-style multi-resolution clustering WITHOUT collapsing cell types
into one shared gene namespace.

Every node is namespaced as "<cell_type>::GENE", so "EGFR" in Fibroblast and
"EGFR" in Enteric_Neuron are DISTINCT nodes. Edge types:
  - 'intra'     : step 1's PPI edges within one cell type's block
  - 'bipartite' : step 2's cross-cell-type edges (LR, Reactome, scored LR,
                  global_ppi identity edges)
  - 'identity'  : optional extra same-gene coupling (--add-identity-coupling)

NEW: DISEASE-MODULE RESTRICTION (--module-csv)
-----------------------------------------------
By default this builds the graph over the FULL multiplex. With --module-csv
(the disease_module.csv from step 6, which has a 'node' column of
"<celltype>::GENE" ids) the graph is restricted to the module:

  1. The full edge list is assembled exactly as before (so edge types and
     the idempotent namespacing are untouched).
  2. The node set is reduced to the module nodes, matched with the same
     normalization step 6 uses (gene part uppercased, cell-type prefix case
     preserved), so case differences between the module and the supra-graph
     don't cause silent misses.
  3. Only edges with BOTH endpoints in the node set are kept (induced
     subgraph). 'intra' / 'bipartite' / 'identity' labels are preserved.

--expand-hops N  additionally pulls in N-hop neighbors of the module nodes
                 (from the full edge list) before inducing. Vertex attribute
                 'in_module' marks which nodes are actual module nodes.
                 WARNING: hubs make this blow up quickly; start with 1.
--keep-isolates  keep module nodes that end up with no edges in the induced
                 subgraph (default: dropped, since they can only ever be
                 singleton clusters and are below --minsize anyway).

Output: a pickled igraph.Graph with vertex attributes 'name' ("ct::gene"),
'cell_type', 'gene', (and 'in_module' when --module-csv is used) and edge
attributes 'weight', 'etype' in {'intra', 'bipartite', 'identity'}.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from provenance import write_manifest
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


def _norm_key(node: str) -> str:
    """Same normalization as step 6's _norm_node: uppercase the gene part
    only, preserve the cell-type prefix's case."""
    node = str(node).strip()
    if "::" in node:
        ct, gene = node.rsplit("::", 1)
        return f"{ct}::{gene.strip().upper()}"
    return node.upper()


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
    already links them. Generic Mucha-et-al. interlayer 'coupling'. Off by
    default; O(sum_gene |celltypes(gene)|^2) so can get large on the full
    multiplex (much cheaper once restricted to a module).
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


def load_module_nodes(module_csv: str):
    df = pd.read_csv(module_csv)
    if "node" not in df.columns:
        raise SystemExit(f"ERROR: {module_csv} has no 'node' column "
                          f"(columns: {list(df.columns)}). Expected step 6's disease_module.csv.")
    nodes = {_norm_key(n) for n in df["node"].dropna()}
    print(f"Loaded {len(nodes):,} disease-module nodes from {module_csv}")
    n_ns = sum(1 for n in nodes if "::" in n)
    if n_ns < len(nodes):
        print(f"  ⚠ {len(nodes) - n_ns} module node(s) are NOT namespaced ('ct::gene') and will "
              f"never match the supra-graph")
    return nodes


def restrict_to_module(all_edges, module_norm: set, hops: int):
    """Return (kept_edges, keep_nodes, module_actual) where kept_edges is the
    induced subgraph on module (+ `hops`-hop neighbors) nodes."""
    norm_to_actual = {}
    for u, v, _, _ in all_edges:
        norm_to_actual.setdefault(_norm_key(u), u)
        norm_to_actual.setdefault(_norm_key(v), v)

    module_actual = {norm_to_actual[k] for k in module_norm if k in norm_to_actual}
    missing = sorted(module_norm - set(norm_to_actual))
    print(f"Module nodes present in supra-graph edge list: {len(module_actual):,} / {len(module_norm):,}")
    if missing:
        print(f"  ⚠ {len(missing):,} module node(s) appear in NO supra-graph edge "
              f"(e.g. {missing[:8]}). Either they're isolated in the multiplex, or the "
              f"module was built from a different reference graph than step 1/2.")
    if not module_actual:
        raise SystemExit("ERROR: none of the module nodes were found in the supra-graph. "
                          "Check namespace/case between disease_module.csv and the wdir.")

    keep = set(module_actual)
    if hops > 0:
        adj = defaultdict(set)
        for u, v, _, _ in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        frontier = set(keep)
        for h in range(1, hops + 1):
            nxt = set()
            for n in frontier:
                nxt |= adj[n]
            nxt -= keep
            keep |= nxt
            frontier = nxt
            print(f"  hop {h}: +{len(nxt):,} neighbor nodes (total {len(keep):,})")

    kept_edges = [e for e in all_edges if e[0] in keep and e[1] in keep]
    print(f"Induced subgraph: {len(kept_edges):,} / {len(all_edges):,} edges retained\n")
    return kept_edges, keep, module_actual

PRIORITY = {"intra": 0, "bipartite": 1, "identity": 2}


def dedupe_edges(edges):
    """Collapse duplicate / reversed-duplicate edges and drop self-loops.
    If the same pair appears with several etypes, keep the most 'intra' one."""
    best = {}
    for u, v, w, e in edges:
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key not in best or PRIORITY[e] < PRIORITY[best[key][3]]:
            best[key] = (key[0], key[1], w, e)
    print(f"Deduplicated edges: {len(edges):,} -> {len(best):,}\n")
    return list(best.values())

def build_supra_graph(wdir: Path, add_identity_coupling: bool = False,
                      module_csv: str = None, expand_hops: int = 0,
                      keep_isolates: bool = False) -> ig.Graph:
    intra_edges, cell_types = load_intra_layer_edges(wdir)
    bip_edges = load_bipartite_edges(wdir)
    all_edges = intra_edges + bip_edges

    if add_identity_coupling:
        all_edges += load_identity_coupling_edges(wdir, set(cell_types))

    all_edges = dedupe_edges(all_edges)

    if not all_edges:
        raise SystemExit("ERROR: no edges loaded at all -- check --wdir points at a completed "
                          "step 1 (+ step 2) output directory.")

    module_actual = None
    keep_nodes = set()
    if module_csv:
        print(f"Restricting to disease module (expand_hops={expand_hops}) ...")
        module_norm = load_module_nodes(module_csv)
        all_edges, keep_nodes, module_actual = restrict_to_module(all_edges, module_norm, expand_hops)
        if not all_edges:
            raise SystemExit("ERROR: the module induces ZERO edges in the supra-graph. "
                              "Try --expand-hops 1.")

    node_set = {u for u, v, w, e in all_edges} | {v for u, v, w, e in all_edges}
    if module_csv and keep_isolates:
        node_set |= keep_nodes
    nodes = sorted(node_set)
    node_idx = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes
    g.vs["cell_type"] = [n.split("::", 1)[0] for n in nodes]
    g.vs["gene"] = [n.split("::", 1)[1] for n in nodes]
    if module_csv:
        g.vs["in_module"] = [n in module_actual for n in nodes]

    g.add_edges([(node_idx[u], node_idx[v]) for u, v, w, e in all_edges])
    g.es["weight"] = [w for u, v, w, e in all_edges]
    g.es["etype"] = [e for u, v, w, e in all_edges]

    n_intra = sum(1 for e in g.es["etype"] if e == "intra")
    n_bip = sum(1 for e in g.es["etype"] if e == "bipartite")
    n_id = sum(1 for e in g.es["etype"] if e == "identity")
    n_iso = sum(1 for d in g.degree() if d == 0)
    comps = g.connected_components()
    print("=" * 60)
    print(f"Supra-graph built: {g.vcount():,} nodes across {len(cell_types)} cell type(s)"
          + (" [DISEASE-MODULE RESTRICTED]" if module_csv else " [FULL MULTIPLEX]"))
    print(f"  {g.ecount():,} edges total ({n_intra:,} intra, {n_bip:,} bipartite, {n_id:,} identity)")
    print(f"  {len(comps):,} connected component(s), largest = {max(comps.sizes()):,}; "
          f"{n_iso:,} isolated node(s)")
    if module_csv:
        print(f"  {sum(g.vs['in_module']):,} module nodes, "
              f"{g.vcount() - sum(g.vs['in_module']):,} expansion neighbor nodes")
        print(f"  mean degree = {2 * g.ecount() / max(g.vcount(), 1):.2f}, "
              f"density = {g.density():.2e}  (useful for choosing the CPM gamma range in 11b)")
    print("=" * 60)
    return g


def main():
    ap = argparse.ArgumentParser(description="Step 11a: build a heterogeneous multiplex supra-graph "
                                               "from a MultiXrank working dir (optionally restricted "
                                               "to a disease module)")
    ap.add_argument("--wdir", required=True, help="MultiXrank working dir (output of step 1 + step 2)")
    ap.add_argument("--output", required=True, help="Path to write the supra-graph (.pkl, igraph.Graph)")
    ap.add_argument("--add-identity-coupling", action="store_true",
                     help="Also add same-gene cross-cell-type identity coupling edges (see docstring)")
    ap.add_argument("--module-csv", default=None,
                     help="disease_module.csv from step 6 (needs a 'node' column of ct::GENE ids). "
                          "If given, the supra-graph is restricted to these nodes.")
    ap.add_argument("--expand-hops", type=int, default=0,
                     help="With --module-csv: also include N-hop neighbors of module nodes (default 0 "
                          "= module nodes only). Marked via vertex attribute 'in_module'.")
    ap.add_argument("--keep-isolates", action="store_true",
                     help="With --module-csv: keep module nodes that have no edges in the induced "
                          "subgraph (default: dropped).")
    args = ap.parse_args()
    inputs = [args.module_csv] if args.module_csv else None
    if inputs:
        write_manifest(Path(args.output).parent, args, inputs=inputs)
    else:
        write_manifest(Path(args.output).parent, args)
    wdir = Path(args.wdir)
    g = build_supra_graph(wdir, add_identity_coupling=args.add_identity_coupling,
                          module_csv=args.module_csv, expand_hops=args.expand_hops,
                          keep_isolates=args.keep_isolates)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(g, f)
    print(f"\nSaved supra-graph -> {out_path}")


if __name__ == "__main__":
    main()