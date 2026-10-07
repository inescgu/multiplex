#!/usr/bin/env python
"""
step12g_removed_genes.py

Which nodes/genes does the hub-removal attack remove (up to f_c), and what do
they have in common?  Reads f_c from <output-dir>/hub_removal/f_c_f90_observed.csv.

IMPORTANT: use the same --betweenness-k / --cascading-step as the main run,
otherwise the reconstructed removal set differs from the one that produced f_c.
(Default = exact betweenness, like the main script's default.)

Outputs (<output-dir>/hub_removal/removed_genes/):
  node_annotation_<mode>_<static|cascading>.csv  every node: removed flag, rank, seed status,
                                                  layer, degree/betweenness/core/clustering,
                                                  articulation point, community role,
                                                  seed-neighbor stats, lcc_drop_nodes/split_off (static)
  gene_table_<mode>_<tag>.csv                    collapsed to bare gene symbols
  removed_vs_kept_stats.csv                      tests (with BH q-values)
  top_cutting_nodes_<mode>_static.csv            removals that split the module most
  enrichment/                                    Enrichr (all / seed / bridge_added removed genes)
  enrichment_module_background_*.csv             with --module-background (needs gseapy)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step12_hub_removal_analysis as s12

METRICS = ["degree", "betweenness", "clustering", "core_number",
           "is_articulation_point", "frac_inter_layer", "n_seed_neighbors",
           "frac_seed_neighbors", "within_module_z", "participation_coefficient"]
CATEGORICAL = ["cell_type", "role"]


def bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if len(pv):
        o = np.argsort(pv)
        adj = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(len(pv)); out[o] = np.minimum(adj, 1)
        q[ok] = out
    return q


def static_removal(G, mode, n_rm, k, seed):
    """Top-n_rm of the fixed ranking (identical to the main attack), plus the LCC
    loss caused by each node when removed in order."""
    order = list(s12._centrality_order(G, mode, betweenness_k=k, seed=seed)[:n_rm])
    H, n0 = G.copy(), G.number_of_nodes()
    prev = s12._lcc_size_and_fraction(H, n0)[0]
    drops = {}
    for n in order:
        H.remove_node(n)
        cur = s12._lcc_size_and_fraction(H, n0)[0]
        drops[n] = prev - cur
        prev = cur
    return order, drops


def cascading_removal(G, mode, n_rm, step, k, seed):
    H, removed = G.copy(), []
    while len(removed) < n_rm and H.number_of_nodes() > 0:
        order = s12._centrality_order(H, mode, betweenness_k=k, seed=seed)
        batch = list(order[:min(step, n_rm - len(removed))])
        H.remove_nodes_from(batch)
        removed += batch
    return removed, {}


def annotate(G, removed, drops, seed_nodes, cent, roles, ann_k, seed):
    rank = {n: i + 1 for i, n in enumerate(removed)}
    deg = cent["degree"] if cent else dict(G.degree())
    btw = cent["betweenness"] if cent else nx.betweenness_centrality(
        G, k=min(ann_k, G.number_of_nodes()) if ann_k else None, seed=seed)
    clus, core = nx.clustering(G), nx.core_number(G)
    art = set(s12._articulation_points_any(G))
    rows = []
    for n in G.nodes():
        nb = list(G.neighbors(n))
        ct = s12._celltype_of(n)
        d = drops.get(n, np.nan)
        rows.append({
            "node": n, "gene": s12._bare_gene(n), "cell_type": ct,
            "removed": n in rank, "removal_rank": rank.get(n, np.nan),
            "is_seed": n in seed_nodes,
            "degree": deg[n], "betweenness": btw[n], "clustering": clus[n],
            "core_number": core[n], "is_articulation_point": n in art,
            "frac_inter_layer": np.mean([s12._celltype_of(m) != ct for m in nb]) if nb else np.nan,
            "n_seed_neighbors": sum(m in seed_nodes for m in nb),
            "frac_seed_neighbors": np.mean([m in seed_nodes for m in nb]) if nb else np.nan,
            "lcc_drop_nodes": d,
            "split_off": max(d - 1, 0) if d == d else np.nan,
        })
    df = pd.DataFrame(rows)
    if roles is not None:
        df = df.merge(roles[["node", "community_id", "within_module_z",
                             "participation_coefficient", "role"]], on="node", how="left")
    return df


def compare(df, label):
    out = []
    rm, kp = df[df.removed], df[~df.removed]
    N, K = len(df), len(rm)
    tab = [[int(rm.is_seed.sum()), int((~rm.is_seed).sum())],
           [int(kp.is_seed.sum()), int((~kp.is_seed).sum())]]
    odds, p = stats.fisher_exact(tab)
    out.append({"attack": label, "test": "seed_status_fisher", "feature": "is_seed",
                "removed_value": rm.is_seed.mean(), "kept_value": kp.is_seed.mean(),
                "effect": odds, "p": p})
    for col in [c for c in CATEGORICAL if c in df.columns]:
        for val, sub in df.dropna(subset=[col]).groupby(col):
            m, x = len(sub), int(sub.removed.sum())
            out.append({"attack": label, "test": f"{col}_hypergeom", "feature": val,
                        "removed_value": x / K if K else np.nan,
                        "kept_value": (m - x) / (N - K) if N > K else np.nan,
                        "effect": (x / K) / (m / N) if K else np.nan,   # fold enrichment
                        "p": stats.hypergeom.sf(x - 1, N, m, K)})
    for c in [c for c in METRICS if c in df.columns]:
        a, b = rm[c].astype(float).dropna(), kp[c].astype(float).dropna()
        if len(a) > 1 and len(b) > 1:
            f = np.mean if c == "is_articulation_point" else np.median
            out.append({"attack": label, "test": "mannwhitney", "feature": c,
                        "removed_value": f(a), "kept_value": f(b), "effect": np.nan,
                        "p": stats.mannwhitneyu(a, b).pvalue})
    return out


def module_background_enrichment(genes, background, name, outdir):
    try:
        import gseapy as gp
    except ImportError:
        print("  gseapy not installed; skipping module-background enrichment"); return
    libs = ["GO_Biological_Process_2023", "GO_Molecular_Function_2023",
            "GO_Cellular_Component_2023", "KEGG_2021_Human", "Reactome_2022"]
    try:
        r = gp.enrichr(gene_list=list(genes), gene_sets=libs, organism="human",
                       background=list(background), outdir=None).results
        r.to_csv(os.path.join(outdir, f"enrichment_module_background_{name}.csv"), index=False)
        print(f"  module-background enrichment saved for {name}")
    except Exception as e:
        print(f"  module-background enrichment failed for {name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-graph", required=True)
    ap.add_argument("--disease-module", required=True)
    ap.add_argument("--output-dir", default="12_hub_removal")
    ap.add_argument("--modes", default="degree", help="degree,betweenness,degree_betweenness")
    ap.add_argument("--cascading", action="store_true")
    ap.add_argument("--cascading-step", type=int, default=10, help="match the main run")
    ap.add_argument("--betweenness-k", type=int, default=None,
                    help="match the main run; default None = exact, like step12")
    ap.add_argument("--annotation-betweenness-k", type=int, default=None,
                    help="only used if centrality.pkl is missing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--module-background", action="store_true")
    ap.add_argument("--skip-enrichment", action="store_true")
    a = ap.parse_args()

    hub = os.path.join(a.output_dir, "hub_removal")
    out = s12.ensure_dir(os.path.join(hub, "removed_genes"))
    enr = s12.ensure_dir(os.path.join(out, "enrichment"))

    G = s12.load_pickle(a.module_graph)
    dm = pd.read_csv(a.disease_module)
    seed_nodes = set(dm.loc[dm["is_seed"] == True, "node"])
    fcs = pd.read_csv(os.path.join(hub, "f_c_f90_observed.csv"))
    cp = os.path.join(hub, "centrality.pkl")
    cent = s12.load_pickle(cp) if os.path.exists(cp) else None
    rp = os.path.join(hub, "node_community_roles.csv")
    roles = pd.read_csv(rp) if os.path.exists(rp) else None
    print(f"centrality.pkl: {'found' if cent else 'missing (computing)'}; "
          f"community roles: {'found' if roles is not None else 'missing'}")
    background_genes = set(s12._bare_gene(n) for n in G.nodes())
    all_stats = []

    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        for casc in ([False, True] if a.cascading else [False]):
            tag = "cascading" if casc else "static"
            row = fcs[(fcs["mode"] == mode) & (fcs["cascading"].astype(bool) == casc)]
            if row.empty or pd.isna(row.iloc[0]["f_c"]):
                print(f"[{mode} {tag}] no f_c in f_c_f90_observed.csv; skipping"); continue
            fc = float(row.iloc[0]["f_c"]); n_rm = int(round(fc * G.number_of_nodes()))
            print(f"[{mode} {tag}] f_c={fc:.4f} -> {n_rm} nodes removed", flush=True)

            if casc:
                removed, drops = cascading_removal(G, mode, n_rm, a.cascading_step, a.betweenness_k, a.seed)
            else:
                removed, drops = static_removal(G, mode, n_rm, a.betweenness_k, a.seed)
            df = annotate(G, removed, drops, seed_nodes, cent, roles, a.annotation_betweenness_k, a.seed)
            df.to_csv(os.path.join(out, f"node_annotation_{mode}_{tag}.csv"), index=False)

            gt = df.groupby("gene").agg(
                n_nodes_total=("node", "size"), n_nodes_removed=("removed", "sum"),
                any_seed=("is_seed", "any"),
                cell_types=("cell_type", lambda x: ";".join(sorted(set(x)))),
                max_degree=("degree", "max")).reset_index()
            gt.sort_values("n_nodes_removed", ascending=False).to_csv(
                os.path.join(out, f"gene_table_{mode}_{tag}.csv"), index=False)

            if not casc:
                top = df[df.removed].sort_values("split_off", ascending=False).head(50)
                top.to_csv(os.path.join(out, f"top_cutting_nodes_{mode}_static.csv"), index=False)
                print("   removals that split off the most nodes:")
                print(top[["gene", "cell_type", "is_seed", "degree", "split_off"]].head(10).to_string(index=False))

            label = f"{mode}_{tag}"
            all_stats += compare(df, label)
            rm = df[df.removed]
            print(f"   removed: {len(rm)} nodes / {rm.gene.nunique()} genes; "
                  f"{rm.is_seed.mean()*100:.1f}% seeds (module: {df.is_seed.mean()*100:.1f}%); "
                  f"{rm.is_articulation_point.mean()*100:.1f}% articulation points")

            if not a.skip_enrichment:
                sets = {"all": set(rm.gene), "seed": set(rm[rm.is_seed].gene),
                        "bridge_added": set(rm[~rm.is_seed].gene)}
                for name, genes in sets.items():
                    if len(genes) < 5:
                        continue
                    nm = f"removed_{label}_{name}"
                    s12.try_enrichment(sorted(genes), nm, enr)
                    if a.module_background:
                        module_background_enrichment(genes, background_genes, nm, out)

    res = pd.DataFrame(all_stats)
    if not res.empty:
        res["q_bh"] = res.groupby(["attack", "test"])["p"].transform(lambda s: bh(s.values))
    res.to_csv(os.path.join(out, "removed_vs_kept_stats.csv"), index=False)
    print(f"Done. Outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())