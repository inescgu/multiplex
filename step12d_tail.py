#!/usr/bin/env python
"""
step12d_tail.py  (Script D)

The two cheap stages that were sitting behind the null benchmark:

  1. Hub-bias permutation test (module collapsed to bare gene symbols vs the
     bare-gene background PPI) -> hub_bias_permutation_test.pkl / .csv
  2. Enrichment of the genes removed by the static degree attack up to f_c
     (needs f_c_f90_observed.csv from the main job; needs internet for Enrichr)
     -> enrichment/enrichment_nodes_removed_by_fc_degree_attack.csv

Usage:
    python step12d_tail.py \
        --module-graph 06_disease_module/disease_module_output_pseudobulk/module_graph.pkl \
        --background-graph ../og_gene_set/temp/combined26/full_network.pkl \
        --output-dir 12_hub_removal \
        --n-permutations 2000
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

import step12_hub_removal_analysis as s12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-graph", required=True)
    ap.add_argument("--background-graph", default=None)
    ap.add_argument("--output-dir", default="12_hub_removal")
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-permutation", action="store_true")
    ap.add_argument("--skip-enrichment", action="store_true")
    args = ap.parse_args()

    out = s12.ensure_dir(os.path.join(args.output_dir, "hub_removal"))
    G = s12.load_pickle(args.module_graph)
    print(f"Module graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    # ---- 1. permutation test ----
    if not args.skip_permutation:
        if not args.background_graph:
            print("No --background-graph given; skipping permutation test.")
        else:
            bg = s12.load_pickle(args.background_graph)
            print(f"Background graph: {bg.number_of_nodes()} nodes, {bg.number_of_edges()} edges")
            genes = s12.to_gene_symbols(G.nodes())
            print(f"Comparing {len(genes)} unique bare gene symbols", flush=True)
            res = s12.hub_bias_permutation_test(bg, genes, n_perm=args.n_permutations, seed=args.seed)
            s12.save_pickle(res, os.path.join(out, "hub_bias_permutation_test.pkl"))

            rows = []
            print(f"Observed mean degree in background: {res['observed_mean_degree']:.3f}")
            for strat in ("size_matched", "degree_matched"):
                if strat in res:
                    r = res[strat]
                    print(f"  [{strat}] null mean={r['null_mean']:.3f} std={r['null_std']:.3f} p={r['p_value']:.4g}")
                    rows.append({"strategy": strat,
                                 "observed_mean_degree": res["observed_mean_degree"],
                                 "n_module_genes_in_bg": res["n_module_genes_in_bg"],
                                 "n_permutations": args.n_permutations, **r})
            pd.DataFrame(rows).to_csv(os.path.join(out, "hub_bias_permutation_test.csv"), index=False)
            if res.get("size_matched", {}).get("p_value", 1.0) < 0.05:
                print("NOTE: module has significantly higher degree than size-matched random gene sets "
                      "(hub-bias / ascertainment caveat).")

    # ---- 2. enrichment of nodes removed by the static degree attack up to f_c ----
    if not args.skip_enrichment:
        p = os.path.join(out, "f_c_f90_observed.csv")
        if not os.path.exists(p):
            print(f"{p} not found; skipping fc-degree-attack enrichment.")
        else:
            f = pd.read_csv(p)
            row = f[(f["mode"] == "degree") & (~f["cascading"].astype(bool))]
            if row.empty or pd.isna(row.iloc[0]["f_c"]):
                print("No static-degree f_c available; skipping.")
            else:
                fc = float(row.iloc[0]["f_c"])
                n_rm = int(round(fc * G.number_of_nodes()))
                print(f"Static degree attack f_c={fc:.4f} -> {n_rm} nodes removed", flush=True)
                if n_rm > 0:
                    order = s12._centrality_order(G, "degree")
                    lost = order[:n_rm]
                    enrich_dir = s12.ensure_dir(os.path.join(out, "enrichment"))
                    s12.try_enrichment(s12.to_gene_symbols(lost),
                                       "nodes_removed_by_fc_degree_attack", enrich_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())