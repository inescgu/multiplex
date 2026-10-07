#!/usr/bin/env python
"""
step12a_make_main.py  (Script A)

Generates step12a_main.py from your existing step12_hub_removal_analysis.py by
applying a handful of exact-string patches (each is asserted to match exactly
once, so it fails loudly instead of silently producing a half-patched file).

The generated step12a_main.py is the same script plus:
  --skip-null          skip the degree-preserving null benchmark (Script B/C do it)
  --skip-ablation      skip the layer-ablation rerun
  --skip-permutation   skip the hub-bias permutation test (Script D does it)
and:
  * the ablation now bails out with a warning if none of the requested cell
    types exist in the module (your 'global_ppi' case: it removed nothing and
    reran the whole suite on an identical graph)
  * when the null is skipped, the null/comparison CSVs and attack_curves.png
    are NOT overwritten with empty versions.

Usage:
    python step12a_make_main.py \
        [--src step12_hub_removal_analysis.py] [--dst step12a_main.py]

Then, only if you need to rerun something the first job did not produce:
    python step12a_main.py <your usual args> --skip-null --skip-ablation --skip-permutation
"""
import argparse
import sys


def patch(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit(f"PATCH FAILED [{label}]: expected exactly 1 match, found {n}")
    return text.replace(old, new)


EMPTY_NULL_HELPERS = '''def _empty_null_df():
    """Empty stand-in for the null results when --skip-null is used."""
    cols = ["null_id", "mode", "cascading", "f_c", "f_90", "n_nodes", "n_edges",
            "assortativity", "n_articulation_points", "frac_articulation_points",
            "n_bridges", "frac_bridges", "max_core_number", "mean_core_number",
            "hub_neighbor_clustering", "avg_clustering", "transitivity",
            "molloy_reed_ratio", "node_connectivity", "edge_connectivity",
            "algebraic_connectivity", "global_efficiency",
            "avg_shortest_path_length", "expensive_metrics_computed"]
    return pd.DataFrame(columns=cols)


def _write_csv_unless_empty(df, path):
    """Write df to path, but never clobber an existing file with an empty
    frame (happens when --skip-null is used after Script C has already
    written real null comparison outputs)."""
    if len(df) == 0 and os.path.exists(path):
        print(f"  (empty result; keeping existing {os.path.basename(path)})")
        return
    df.to_csv(path, index=False)


def main():
'''

NEW_ARGS = '''    parser.add_argument("--skip-null", action="store_true",
                         help="Skip the degree-preserving null benchmark and everything that "
                              "depends on it (use step12b/step12c instead).")
    parser.add_argument("--skip-ablation", action="store_true",
                         help="Skip the layer-ablation attack-curve rerun.")
    parser.add_argument("--skip-permutation", action="store_true",
                         help="Skip the hub-bias permutation test (use step12d instead).")

    args = parser.parse_args()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="step12_hub_removal_analysis.py")
    ap.add_argument("--dst", default="step12a_main.py")
    a = ap.parse_args()

    with open(a.src) as f:
        t = f.read()

    t = patch(t, "def main():\n", EMPTY_NULL_HELPERS, "helpers before main()")
    t = patch(t, "    args = parser.parse_args()\n", NEW_ARGS, "new CLI flags")

    # null benchmark: swap in an empty frame when skipped (continuation-line
    # indentation inside the call parentheses doesn't matter to Python)
    t = patch(t, "        null_df = degree_preserving_null_curves(",
              "        null_df = _empty_null_df() if args.skip_null else degree_preserving_null_curves(",
              "null call")

    # don't clobber real outputs with empty ones
    for name, var in [("null_fc_f90.csv", "null_df"),
                      ("observed_vs_null_comparison.csv", "comparison_df"),
                      ("null_topology_stats.csv", "null_topo_df"),
                      ("null_topology_comparison.csv", "topo_comparison_df")]:
        t = patch(t,
                  f'{var}.to_csv(os.path.join(output_path, "{name}"), index=False)',
                  f'_write_csv_unless_empty({var}, os.path.join(output_path, "{name}"))',
                  f"csv write {name}")

    t = patch(t, "        plot_attack_curves(curves, null_df, output_path)",
              "        if not args.skip_null:\n            plot_attack_curves(curves, null_df, output_path)",
              "plot guard")

    # ablation
    t = patch(t, "if args.ablation_exclude_celltypes:",
              "if args.ablation_exclude_celltypes and not args.skip_ablation:",
              "ablation flag")
    t = patch(t, "G_ablated = G.subgraph(keep_nodes).copy()",
              "G_ablated = G.subgraph(keep_nodes).copy()\n"
              "    if len(keep_nodes) == G.number_of_nodes():\n"
              "        present = sorted({_celltype_of(n) for n in G.nodes()})\n"
              "        print(f\"  [ablation] WARNING: none of {sorted(exclude_upper)} match a cell type in \"\n"
              "              f\"this module (available: {present}). Nothing removed -- skipping rerun.\")\n"
              "        return None, None",
              "ablation no-op guard")

    # permutation test
    t = patch(t, "if background_G is not None:",
              "if background_G is not None and not args.skip_permutation:",
              "permutation flag")

    with open(a.dst, "w") as f:
        f.write(t)
    print(f"Wrote {a.dst} (7 patch groups applied cleanly).")


if __name__ == "__main__":
    main()