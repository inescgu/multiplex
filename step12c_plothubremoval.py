#!/usr/bin/env python
"""
Hub-Removal Robustness: Observed vs. Degree-Preserving Null
--------------------------------------------------------------
Builds plots from the outputs of step12_hub_removal_analysis.py.

ORIGINAL PLOTS (unchanged behavior):
  - hub_removal_robustness_static_fc.png / _cascading_fc.png : grouped bar
    chart, real module vs. null median, per attack mode, from
    observed_vs_null_comparison.csv.
  - hub_removal_percentile_summary.png : where the observed value falls in
    the null distribution, across all mode/statistic combos.

NEW PLOTS, one per mechanism-level diagnostic added to step12 (each is
OPTIONAL -- skipped with a one-line note if step12 wasn't run with the
option that produces it, e.g. --disease-module or
--ablation-exclude-celltypes):
  - topology_vs_null.png : assortativity, articulation-point count/
    fraction, and hub-neighborhood clustering, real module vs. null
    (from null_topology_comparison.csv).
  - hub_articulation_overlap.png : what fraction of each hub definition's
    members are articulation points/cut vertices of the real module
    (from hub_articulation_overlap.csv).
  - hub_layer_participation.png : mean fraction of inter-layer
    (cross-cell-type) vs. intra-layer (within-cell-type PPI) edges per
    hub definition, against the module-wide mean (from
    hub_layer_participation_summary.csv).
  - celltype_knockout.png : LCC fraction remaining after removing each
    cell type's node complement wholesale, independent of any hub
    definition (from celltype_knockout_summary.csv).
  - hub_knockout_seed_vs_bridging.png : LCC fraction remaining after
    knocking out only the seed-gene members vs. only the non-seed
    (RWR-bridging-gene) members of each hub definition (from
    hub_knockout_by_seed_status.csv; needs step12's --disease-module).
  - layer_ablation_comparison.png : f_c for the full module vs. the
    module with a layer stripped out (from ablation_f_c_f90.csv +
    f_c_f90_observed.csv; needs step12's --ablation-exclude-celltypes).

Usage:
    python plot_hub_removal_robustness.py <path/to/hub_removal/> [output_dir]

<path/to/hub_removal/> is step12's --output-dir/hub_removal folder (i.e.
pass the whole directory, not one CSV -- this version needs several of
step12's output files, not just observed_vs_null_comparison.csv).
For backward compatibility, a direct path to observed_vs_null_comparison.csv
itself is still accepted; its parent directory is used to look for the
other files.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def resolve_hub_removal_dir(path_arg):
    """Accepts either the hub_removal output directory itself, or (for
    backward compatibility with the original version of this script) a
    direct path to observed_vs_null_comparison.csv inside it -- either
    way, returns the directory to look for all the other CSVs in."""
    if not path_arg:
        print("ERROR: no path given.")
        print("Usage: python plot_hub_removal_robustness.py <path/to/hub_removal/> [output_dir]")
        sys.exit(1)
    if os.path.isdir(path_arg):
        return path_arg
    if os.path.isfile(path_arg):
        return os.path.dirname(os.path.abspath(path_arg)) or "."
    print(f"ERROR: path not found: {path_arg}")
    sys.exit(1)


def load_required_csv(hub_removal_dir, filename, required_cols):
    path = os.path.join(hub_removal_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: required file not found: {path}")
        print("Run step12_hub_removal_analysis.py first to generate it.")
        sys.exit(1)
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing expected column(s): {sorted(missing)}")
        sys.exit(1)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_optional_csv(hub_removal_dir, filename, required_cols=None, label=None):
    """Loads a CSV if present, else prints a one-line skip note and
    returns None. Used for the NEW plots, which are only produced by
    step12 under certain flags (--disease-module,
    --ablation-exclude-celltypes) so their inputs won't always exist."""
    path = os.path.join(hub_removal_dir, filename)
    if not os.path.exists(path):
        print(f"  [skip] {label or filename}: not found at {path} -- skipping the corresponding plot.")
        return None
    df = pd.read_csv(path)
    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            print(f"  [skip] {label or filename}: missing column(s) {sorted(missing)} -- skipping.")
            return None
    print(f"  Loaded {len(df)} rows from {path}")
    return df


# ---------------------------------------------------------------------------
# Original plots (unchanged)
# ---------------------------------------------------------------------------

def plot_robustness(df, statistic="f_c", cascading=False, output_path="hub_removal_robustness.png"):
    """
    Grouped bar chart: observed vs null-median, one group per attack mode,
    for a single statistic (f_c or f_90) and attack style (static/cascading).
    """
    sub = df[(df["statistic"] == statistic) & (df["cascading"] == cascading)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for statistic={statistic}, cascading={cascading}")

    # Nice display order / labels
    mode_order = ["degree", "betweenness", "degree_betweenness"]
    mode_labels = {"degree": "Degree", "betweenness": "Betweenness", "degree_betweenness": "Degree +\nBetweenness"}
    sub["mode"] = pd.Categorical(sub["mode"], categories=mode_order, ordered=True)
    sub = sub.sort_values("mode")

    modes = sub["mode"].tolist()
    observed = sub["observed"].values
    null_median = sub["null_median"].values
    null_std = sub["null_std"].values

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars_obs = ax.bar(
        x - width / 2, observed, width,
        label="Real disease module", color="#c0392b", edgecolor="black", zorder=3,
    )
    bars_null = ax.bar(
        x + width / 2, null_median, width,
        yerr=null_std, capsize=5,
        label="Degree-preserving null\n(median \u00b1 SD)", color="#95a5a6", edgecolor="black", zorder=3,
    )

    # Value labels on top of bars
    for bar, val in zip(bars_obs, observed):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c0392b")
    for bar, val in zip(bars_null, null_median):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10, color="#555555")

    stat_label = {
        "f_c": "Fraction of nodes removed before network\ncollapses to \u22645% of original size (f\u1D04)",
        "f_90": "Fraction of nodes removed before network\ndrops to \u226490% of original size (f\u2089\u2080)",
    }[statistic]
    style_label = "Cascading attack (re-ranked as nodes are removed)" if cascading else "Static attack (fixed ranking)"

    ax.set_ylabel(stat_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[m] for m in modes], fontsize=11)
    ax.set_title(
        f"Disease Module Does Not Resist Hub Removal Better Than Expected\n{style_label}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(observed.max(), null_median.max()) * 1.25)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the takeaway
    ax.text(
        0.5, -0.18,
        "Higher bars = more removal needed to fragment the network \u2192 the real module is not more \n"
        "robust than a random network with the same degree sequence, across most attack strategies tested.",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, style="italic", color="#333333",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_percentiles(df, output_path="hub_removal_percentile_summary.png"):
    """
    Compact summary panel: for every mode/cascading/statistic combo, show
    observed_percentile_in_null (or compute it if not present) as a single
    bar chart. Communicates "how extreme is the real module vs the null
    distribution" in one glance.
    """
    sub = df.copy()
    if "observed_percentile_in_null" not in sub.columns:
        raise ValueError("observed_percentile_in_null column not found; "
                          "pass the full observed_vs_null_comparison.csv for this plot.")

    sub["label"] = (
        sub["mode"].str.replace("_", "+").str.title()
        + " (" + np.where(sub["cascading"], "cascading", "static") + ", "
        + sub["statistic"] + ")"
    )
    sub = sub.sort_values("observed_percentile_in_null")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if p >= 95 else "#e67e22" if p >= 80 else "#95a5a6" for p in sub["observed_percentile_in_null"]]
    ax.barh(sub["label"], sub["observed_percentile_in_null"], color=colors, edgecolor="black")
    ax.axvline(50, color="gray", linestyle=":", alpha=0.6, label="Null median (50th pct)")
    ax.set_xlabel("Percentile of observed value within null distribution", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title("How Extreme Is the Real Module vs. Degree-Preserving Nulls?",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# NEW plots -- one per mechanism-level diagnostic added to step12
# ---------------------------------------------------------------------------

TOPOLOGY_STAT_LABELS = {
    "assortativity": "Degree assortativity\n(more negative = hubs attach to low-degree nodes)",
    "n_articulation_points": "Number of articulation points\n(cut vertices)",
    "frac_articulation_points": "Fraction of nodes that are\narticulation points",
    "n_bridges": "Number of bridges\n(cut edges)",
    "frac_bridges": "Fraction of edges that\nare bridges",
    "max_core_number": "Max k-core\n(deepest core)",
    "mean_core_number": "Mean k-core",
    "hub_neighbor_clustering": "Mean clustering coefficient of\ntop-degree hub neighborhoods",
    "avg_clustering": "Average clustering\ncoefficient (whole module)",
    "transitivity": "Transitivity\n(whole module)",
    "node_connectivity": "Global node connectivity\n(min nodes to disconnect any pair)",
    "edge_connectivity": "Global edge connectivity\n(min edges to disconnect any pair)",
    "algebraic_connectivity": "Algebraic connectivity\n(Fiedler value, LCC)",
    "global_efficiency": "Global efficiency",
    "avg_shortest_path_length": "Avg shortest path\nlength (LCC)",
}

# Two curated small-multiple panels so a single figure doesn't get
# overcrowded -- structural/cut-vertex stats vs. connectivity/redundancy
# stats. plot_topology_comparison(df, ..., stat_subset=...) picks which.
TOPOLOGY_STRUCTURAL_STATS = [
    "assortativity", "n_articulation_points", "frac_articulation_points",
    "n_bridges", "hub_neighbor_clustering",
]
TOPOLOGY_CONNECTIVITY_STATS = [
    "node_connectivity", "algebraic_connectivity", "global_efficiency",
    "avg_shortest_path_length", "avg_clustering",
]


def plot_topology_comparison(df, output_path, stat_subset=None, title=None):
    """
    null_topology_comparison.csv (or three_way_topology_comparison.csv,
    see plot_three_way_topology) -> small multiples, one panel per
    statistic, each showing the real module's value against the null
    median +/- SD. Same observed-vs-null idea as plot_robustness,
    generalized to statistics that are a single module-level number
    rather than broken out by attack mode.

    stat_subset restricts which statistics are plotted (and their order)
    -- see TOPOLOGY_STRUCTURAL_STATS / TOPOLOGY_CONNECTIVITY_STATS above
    for the two curated panels this script produces by default.
    """
    labels = TOPOLOGY_STAT_LABELS
    candidates = stat_subset if stat_subset is not None else list(labels)
    stats_present = [s for s in candidates if s in df["statistic"].values and s in labels]
    if not stats_present:
        print(f"  [skip] plot_topology_comparison ({output_path}): no recognized statistics present")
        return

    fig, axes = plt.subplots(1, len(stats_present), figsize=(4.5 * len(stats_present), 5.5))
    if len(stats_present) == 1:
        axes = [axes]

    for ax, stat in zip(axes, stats_present):
        row = df[df["statistic"] == stat].iloc[0]
        x = np.arange(2)
        vals = [row["observed"], row["null_median"]]
        errs = [0, row["null_std"]]
        colors = ["#c0392b", "#95a5a6"]
        bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="black", zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Real\nmodule", "Null\n(median \u00b1 SD)"], fontsize=9)
        ax.set_title(labels[stat], fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        pct = row.get("observed_percentile_in_null", np.nan)
        if pd.notna(pct):
            ax.text(0.5, -0.16, f"observed = {pct:.0f}th pct of null",
                     transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color="#333333")

    fig.suptitle(title or "Real Module vs. Degree-Preserving Null: Topology Diagnostics",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_hub_articulation_overlap(df, output_path):
    """
    hub_articulation_overlap.csv -> horizontal bar of what fraction of
    each hub definition's members are articulation points (cut vertices)
    of the real module. This is the most direct mechanistic plot: a hub
    set that's mostly cut vertices explains fragility structurally,
    independent of degree.
    """
    sub = df.sort_values("frac_hub_nodes_that_are_articulation_points")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(sub))))
    colors = ["#c0392b" if v >= 0.5 else "#e67e22" if v >= 0.2 else "#95a5a6"
              for v in sub["frac_hub_nodes_that_are_articulation_points"]]
    bars = ax.barh(sub["hub_definition"], sub["frac_hub_nodes_that_are_articulation_points"] * 100,
                    color=colors, edgecolor="black")
    for bar, n_artic, n_total in zip(bars, sub["n_hub_nodes_that_are_articulation_points"], sub["n_hub_nodes"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(n_artic)}/{int(n_total)}", va="center", fontsize=8.5, color="#333333")
    ax.set_xlabel("% of hub-set members that are articulation points (cut vertices)", fontsize=10)
    ax.set_xlim(0, max(105, sub["frac_hub_nodes_that_are_articulation_points"].max() * 100 + 15))
    ax.set_title("How Much of Each Hub Definition Is a Structural Cut Vertex?",
                  fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_participation(df, output_path):
    """
    hub_layer_participation_summary.csv -> bar of mean fraction of
    inter-layer (cross-cell-type identity) edges per hub definition,
    against the module-wide mean (dashed line). Bars to the right of the
    line are hub sets dominated by cross-layer connector nodes rather
    than genuine within-layer PPI hubs -- worth knowing before
    interpreting a hub biologically.
    """
    sub = df.dropna(subset=["mean_frac_inter_layer"]).copy()
    if sub.empty:
        print("  [skip] plot_layer_participation: no non-NaN rows")
        return
    module_mean = sub["module_wide_mean_frac_inter_layer"].iloc[0]
    sub = sub.sort_values("mean_frac_inter_layer")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(sub))))
    colors = ["#8e44ad" if v > module_mean else "#3498db" for v in sub["mean_frac_inter_layer"]]
    ax.barh(sub["hub_definition"], sub["mean_frac_inter_layer"], color=colors, edgecolor="black")
    ax.axvline(module_mean, color="black", linestyle="--", linewidth=1.2,
               label=f"Module-wide mean ({module_mean:.3f})")
    ax.set_xlabel("Mean fraction of a node's edges that are inter-layer\n"
                  "(cross-cell-type identity edges, vs. within-layer PPI edges)", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_title("Are Hub Sets Cross-Layer Connectors or Within-Layer PPI Hubs?",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_celltype_knockout(df, output_path):
    """
    celltype_knockout_summary.csv -> horizontal bar of LCC fraction
    remaining after removing each cell type's node complement entirely,
    independent of any hub definition. Lowest bars = the layer whose
    presence is doing the most connectivity work.
    """
    sub = df.sort_values("lcc_fraction_after")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(sub))))
    colors = ["#c0392b" if v <= 0.5 else "#e67e22" if v <= 0.8 else "#27ae60"
              for v in sub["lcc_fraction_after"]]
    bars = ax.barh(sub["cell_type"], sub["lcc_fraction_after"], color=colors, edgecolor="black")
    for bar, frac_rm in zip(bars, sub["frac_removed"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{frac_rm*100:.0f}% of nodes removed", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("LCC fraction remaining after removing this cell type's nodes entirely", fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.set_title("Which Cell Type's Removal Does the Most Damage?",
                  fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_seed_vs_bridging(df, output_path):
    """
    hub_knockout_by_seed_status.csv -> grouped bar per hub definition,
    seed-only vs. non-seed/bridging-only knockout, showing LCC fraction
    remaining. A large gap means the fragility measured for a hub
    definition is dominated by one subset -- if it's mostly the
    'nonseed_bridging_only' bars that collapse, revisit --max-bridge /
    the step6 bridging elbow before treating the fragility as biological.
    """
    pivot = df.pivot_table(index="hub_definition", columns="subset",
                            values="lcc_fraction_after", aggfunc="first")
    for col in ("seed_only", "nonseed_bridging_only"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["seed_only", "nonseed_bridging_only"]]

    x = np.arange(len(pivot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(pivot)), 6))
    ax.bar(x - width / 2, pivot["seed_only"], width, label="Seed-gene hubs removed",
           color="#2980b9", edgecolor="black", zorder=3)
    ax.bar(x + width / 2, pivot["nonseed_bridging_only"], width, label="Non-seed (bridging-gene) hubs removed",
           color="#f39c12", edgecolor="black", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("LCC fraction remaining after knockout", fontsize=10)
    ax.set_title("Fragility Driven by Seed Genes or by Bridging Genes?",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_node_origin_boxplots(node_diag_df, output_path):
    """
    node_level_diagnostics.csv -> side-by-side boxplots of degree,
    betweenness, and delta_lcc (removal impact), split by seed vs.
    bridge-added origin. The distributional comparison the advisor asked
    for (advisor #2). delta_lcc is only populated for a bounded node set
    (articulation points + hub-set members, see step12), so that panel
    uses however many non-NaN values are available per group.
    """
    if "is_seed" not in node_diag_df.columns or node_diag_df.empty:
        print("  [skip] plot_node_origin_boxplots: no is_seed column / empty data")
        return
    panels = [("degree", "Degree"), ("betweenness", "Betweenness"), ("delta_lcc", "Removal impact (delta_lcc)")]
    panels = [(c, lbl) for c, lbl in panels if c in node_diag_df.columns]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, panels):
        seed_vals = node_diag_df.loc[node_diag_df["is_seed"], col].dropna().values
        bridge_vals = node_diag_df.loc[~node_diag_df["is_seed"], col].dropna().values
        data = [v for v in [seed_vals, bridge_vals] if len(v)]
        tick_labels = []
        if len(seed_vals):
            tick_labels.append(f"Seed\n(n={len(seed_vals)})")
        if len(bridge_vals):
            tick_labels.append(f"Bridge-added\n(n={len(bridge_vals)})")
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], ["#2980b9", "#f39c12"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Seed vs. Bridge-Added Node Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_knockout_by_origin(df, output_path):
    """knockout_by_origin_summary.csv -> LCC fraction remaining after
    removing ALL seeds at once vs. ALL bridge-added genes at once
    (advisor #2's full-set, not just hub-set, knockout)."""
    if df.empty:
        print("  [skip] plot_knockout_by_origin: empty data")
        return
    label_map = {"all_seeds": "All seeds\nremoved", "all_bridge_added": "All bridge-added\ngenes removed"}
    sub = df.copy()
    sub["label"] = sub["origin_group"].map(label_map).fillna(sub["origin_group"])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#2980b9" if g == "all_seeds" else "#f39c12" for g in sub["origin_group"]]
    bars = ax.bar(sub["label"], sub["lcc_fraction_after"], color=colors, edgecolor="black", zorder=3)
    for bar, frac_rm in zip(bars, sub["frac_removed"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{frac_rm*100:.0f}% of nodes removed", ha="center", fontsize=8.5, color="#333333")
    ax.set_ylabel("LCC fraction remaining", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Full Seed-Set vs. Full Bridge-Added-Set Knockout", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_celltype_subgraph_fragility(df, output_path):
    """celltype_subgraph_attack_summary.csv -> f_c for each cell type's
    OWN induced subgraph attacked in isolation (advisor #4: is a layer
    fragile entirely on its own, independent of the rest of the module)."""
    sub = df[df["mode"] == "degree"].dropna(subset=["f_c"]).sort_values("f_c")
    if sub.empty:
        print("  [skip] plot_celltype_subgraph_fragility: no usable rows")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(sub))))
    colors = ["#c0392b" if v <= 0.1 else "#e67e22" if v <= 0.25 else "#27ae60" for v in sub["f_c"]]
    bars = ax.barh(sub["cell_type"], sub["f_c"], color=colors, edgecolor="black")
    for bar, n_nodes in zip(bars, sub["n_nodes"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={int(n_nodes)}", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("f_c of this cell type's own induced subgraph\n(degree attack, lower = more fragile alone)", fontsize=9.5)
    ax.set_title("Is Each Cell-Type Layer Fragile on Its Own?", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_celltype_hub_knockout_heatmap(df, output_path):
    """celltype_hub_knockout.csv -> heatmap of LCC fraction remaining
    after removing hub_definition's members WITHIN cell_type only.
    Shows whether a hub definition's damage is concentrated in one
    neuron population or spread across layers."""
    if df.empty:
        print("  [skip] plot_celltype_hub_knockout_heatmap: empty data")
        return
    pivot = df.pivot_table(index="hub_definition", columns="cell_type",
                            values="lcc_fraction_after", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * pivot.shape[1]), max(4, 0.5 * pivot.shape[0])))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < v < 0.8 else "white")
    fig.colorbar(im, ax=ax, label="LCC fraction remaining")
    ax.set_title("Hub Definition \u00d7 Cell Type: Knockout Damage", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_community_roles_scatter(node_roles_df, output_path):
    """
    node_community_roles.csv -> the classic Guimera-Amaral scatter:
    within-module degree z-score (y) vs. participation coefficient (x),
    colored by role, with the standard cutoff lines (z=2.5, P=0.62).
    Connector hubs (top-right quadrant) are the nodes expected to
    fragment the module if removed; provincial hubs (top-left) are more
    replaceable.
    """
    if node_roles_df.empty:
        print("  [skip] plot_community_roles_scatter: empty data")
        return
    role_colors = {
        "connector_hub": "#c0392b", "provincial_hub": "#e67e22",
        "connector_non_hub": "#2980b9", "peripheral": "#95a5a6",
    }
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for role, color in role_colors.items():
        sub = node_roles_df[node_roles_df["role"] == role]
        if sub.empty:
            continue
        ax.scatter(sub["participation_coefficient"], sub["within_module_z"],
                   s=25, alpha=0.7, color=color, edgecolor="black", linewidth=0.3, label=role, zorder=3)
    ax.axhline(2.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.62, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Participation coefficient", fontsize=10)
    ax.set_ylabel("Within-module degree z-score", fontsize=10)
    ax.set_title("Guimera-Amaral Node Roles: Connector vs. Provincial Hubs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_hub_community_role_composition(df, output_path):
    """hub_sets_community_role_summary.csv -> stacked horizontal bar of
    Guimera-Amaral role composition per hub definition."""
    if df.empty:
        print("  [skip] plot_hub_community_role_composition: empty data")
        return
    roles = ["connector_hub", "provincial_hub", "connector_non_hub", "peripheral"]
    colors = {"connector_hub": "#c0392b", "provincial_hub": "#e67e22",
              "connector_non_hub": "#2980b9", "peripheral": "#95a5a6"}
    frac_cols = [f"frac_{r}" for r in roles if f"frac_{r}" in df.columns]
    if not frac_cols:
        return
    sub = df.set_index("hub_definition")[frac_cols].fillna(0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(sub))))
    left = np.zeros(len(sub))
    for r in roles:
        col = f"frac_{r}"
        if col not in sub.columns:
            continue
        ax.barh(sub.index, sub[col], left=left, color=colors[r], edgecolor="black", label=r)
        left += sub[col].values
    ax.set_xlabel("Fraction of hub-set members in each role", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title("Connector/Provincial-Hub Composition per Hub Definition", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_seed_component_connectivity(df, output_path):
    """seed_component_connectivity.csv -> heatmap of node-disjoint paths
    (Menger's theorem) between pairs of original seed components. Low
    values mean the module relies on very few alternate routes to keep
    those seed components connected."""
    if df.empty:
        print("  [skip] plot_seed_component_connectivity: empty data")
        return
    comps = sorted(set(df["component_i"]).union(df["component_j"]))
    n = len(comps)
    mat = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = comps.index(r["component_i"]), comps.index(r["component_j"])
        mat[i, j] = r["node_disjoint_paths"]
        mat[j, i] = r["node_disjoint_paths"]
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * n), max(4, 0.8 * n)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Comp {c}" for c in comps], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Comp {c}" for c in comps], fontsize=9)
    for i in range(n):
        for j in range(n):
            if pd.notna(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Node-disjoint paths")
    ax.set_title("Node-Disjoint Paths Between Original Seed Components", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_three_way_fc(df, output_path, statistic="f_c", cascading=False):
    """
    three_way_fc_f90_comparison.csv (from step12b_layer_preserving_null.py)
    -> grouped bar of real module vs. naive null vs. layer-preserving
    null, per attack mode. This is the plot that answers the advisor's
    "essential control" question directly: if the naive-null bar makes
    the module look fragile but the layer-preserving-null bar doesn't,
    the original finding was a multiplex-construction artifact.
    """
    sub = df[(df["statistic"] == statistic) & (df["cascading"] == cascading)].copy()
    if sub.empty or "layer_preserving_null_median" not in sub.columns:
        print(f"  [skip] plot_three_way_fc: no usable rows for statistic={statistic}, cascading={cascading}")
        return
    mode_order = ["degree", "betweenness", "degree_betweenness"]
    mode_labels = {"degree": "Degree", "betweenness": "Betweenness", "degree_betweenness": "Degree +\nBetweenness"}
    sub["mode"] = pd.Categorical(sub["mode"], categories=mode_order, ordered=True)
    sub = sub.sort_values("mode").dropna(subset=["mode"])
    if sub.empty:
        return

    has_naive = "naive_null_median" in sub.columns and sub["naive_null_median"].notna().any()
    n_bars = 3 if has_naive else 2
    width = 0.8 / n_bars
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, sub["observed"], width, label="Real module", color="#c0392b", edgecolor="black", zorder=3)
    offset = 0
    if has_naive:
        ax.bar(x, sub["naive_null_median"], width, yerr=sub["naive_null_std"], capsize=4,
               label="Naive null\n(whole-graph swap)", color="#95a5a6", edgecolor="black", zorder=3)
        offset = width
    ax.bar(x - width + width + offset, sub["layer_preserving_null_median"], width,
           yerr=sub["layer_preserving_null_std"], capsize=4,
           label="Layer-preserving null\n(edge-type-stratified)", color="#16a085", edgecolor="black", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels.get(m, m) for m in sub["mode"]], fontsize=10)
    stat_label = "f_c (LCC \u2264 5%)" if statistic == "f_c" else "f_90 (LCC \u2264 90%)"
    ax.set_ylabel(f"Fraction of nodes removed ({stat_label})", fontsize=10)
    style_label = "cascading" if cascading else "static"
    ax.set_title(f"Real Module vs. Naive Null vs. Layer-Preserving Null\n({stat_label}, {style_label} attack)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_three_way_topology(df, output_path, stat_subset=None):
    """three_way_topology_comparison.csv -> same idea as
    plot_topology_comparison but with a third bar for the naive null, so
    assortativity/articulation-points/etc. can be checked against both
    nulls side by side."""
    labels = TOPOLOGY_STAT_LABELS
    candidates = stat_subset if stat_subset is not None else list(labels)
    stats_present = [s for s in candidates if s in df["statistic"].values and s in labels]
    if not stats_present:
        print(f"  [skip] plot_three_way_topology ({output_path}): no recognized statistics present")
        return

    fig, axes = plt.subplots(1, len(stats_present), figsize=(4.5 * len(stats_present), 5.5))
    if len(stats_present) == 1:
        axes = [axes]
    for ax, stat in zip(axes, stats_present):
        row = df[df["statistic"] == stat].iloc[0]
        has_naive = pd.notna(row.get("naive_null_median", np.nan))
        labels_x = ["Real\nmodule"]
        vals = [row["observed"]]
        errs = [0]
        colors = ["#c0392b"]
        if has_naive:
            labels_x.append("Naive\nnull")
            vals.append(row["naive_null_median"])
            errs.append(row.get("naive_null_std", 0))
            colors.append("#95a5a6")
        labels_x.append("Layer-\npreserving\nnull")
        vals.append(row.get("layer_preserving_null_median", np.nan))
        errs.append(row.get("layer_preserving_null_std", 0))
        colors.append("#16a085")

        x = np.arange(len(labels_x))
        bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="black", zorder=3)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                         fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=8.5)
        ax.set_title(labels[stat], fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Real Module vs. Naive Null vs. Layer-Preserving Null: Topology",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_comparison(ablation_df, observed_df, output_path, ablated_label="Ablated module"):
    """
    Compares f_c between the full module (f_c_f90_observed.csv) and the
    module with one or more cell-type layers stripped out
    (ablation_f_c_f90.csv), per attack mode/static-cascading combo
    present in both.
    """
    merged = pd.merge(
        observed_df, ablation_df, on=["mode", "cascading"], suffixes=("_full", "_ablated"),
    )
    if merged.empty:
        print("  [skip] plot_ablation_comparison: no overlapping mode/cascading rows between "
              "f_c_f90_observed.csv and ablation_f_c_f90.csv")
        return

    mode_labels = {"degree": "Degree", "betweenness": "Betweenness",
                   "degree_betweenness": "Degree +\nBetweenness", "random": "Random"}
    merged["label"] = merged.apply(
        lambda r: f"{mode_labels.get(r['mode'], r['mode'])}\n({'cascading' if r['cascading'] else 'static'})",
        axis=1,
    )

    x = np.arange(len(merged))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(merged)), 6))
    ax.bar(x - width / 2, merged["f_c_full"], width, label="Full module",
           color="#c0392b", edgecolor="black", zorder=3)
    ax.bar(x + width / 2, merged["f_c_ablated"], width, label=ablated_label,
           color="#16a085", edgecolor="black", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], fontsize=9)
    ax.set_ylabel("f_c (fraction removed before LCC \u2264 5%)", fontsize=10)
    ax.set_title("Does Removing a Layer Change How Fragile the Module Is?",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_hub_removal_robustness.py <path/to/hub_removal/> [output_dir]")
        print("  <path/to/hub_removal/> is step12's --output-dir/hub_removal folder (pass the")
        print("  whole directory -- this version reads several of step12's output files, not")
        print("  just observed_vs_null_comparison.csv). A direct path to")
        print("  observed_vs_null_comparison.csv itself is also still accepted.")
        sys.exit(1)

    path_arg = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    hub_removal_dir = resolve_hub_removal_dir(path_arg)
    print(f"Reading step12 outputs from: {hub_removal_dir}\n")

    # --- Original plots (required; unchanged behavior) ---------------------
    df = load_required_csv(
        hub_removal_dir, "observed_vs_null_comparison.csv",
        required_cols={"mode", "cascading", "statistic", "observed", "null_median", "null_std"},
    )

    plot_robustness(df, statistic="f_c", cascading=False,
                     output_path=os.path.join(out_dir, "hub_removal_robustness_static_fc.png"))
    plot_robustness(df, statistic="f_c", cascading=True,
                     output_path=os.path.join(out_dir, "hub_removal_robustness_cascading_fc.png"))
    if "observed_percentile_in_null" in df.columns:
        plot_all_percentiles(df, output_path=os.path.join(out_dir, "hub_removal_percentile_summary.png"))
    else:
        print("Skipping percentile summary plot (no observed_percentile_in_null column).")

    # --- NEW plots: each optional, skipped with a note if step12 wasn't ----
    # --- run with the flag that produces its input CSV. --------------------
    print("\n--- New diagnostic plots ---")

    topo_df = load_optional_csv(
        hub_removal_dir, "null_topology_comparison.csv",
        required_cols={"statistic", "observed", "null_median", "null_std"},
        label="topology comparison (assortativity/articulation points/connectivity/efficiency/...)",
    )
    if topo_df is not None:
        plot_topology_comparison(topo_df, os.path.join(out_dir, "topology_vs_null_structural.png"),
                                  stat_subset=TOPOLOGY_STRUCTURAL_STATS,
                                  title="Real Module vs. Null: Cut-Vertex / Structural Diagnostics")
        plot_topology_comparison(topo_df, os.path.join(out_dir, "topology_vs_null_connectivity.png"),
                                  stat_subset=TOPOLOGY_CONNECTIVITY_STATS,
                                  title="Real Module vs. Null: Connectivity / Redundancy Diagnostics")

    node_diag_df = load_optional_csv(
        hub_removal_dir, "node_level_diagnostics.csv",
        required_cols={"is_seed", "degree", "betweenness"},
        label="node-level diagnostics (seed vs. bridge-added)",
    )
    if node_diag_df is not None:
        plot_node_origin_boxplots(node_diag_df, os.path.join(out_dir, "node_origin_diagnostics.png"))

    origin_ko_df = load_optional_csv(
        hub_removal_dir, "knockout_by_origin_summary.csv",
        required_cols={"origin_group", "lcc_fraction_after"},
        label="full seed-set vs. bridge-added-set knockout",
    )
    if origin_ko_df is not None:
        plot_knockout_by_origin(origin_ko_df, os.path.join(out_dir, "knockout_by_origin.png"))

    celltype_subgraph_df = load_optional_csv(
        hub_removal_dir, "celltype_subgraph_attack_summary.csv",
        required_cols={"cell_type", "mode", "f_c"},
        label="per-cell-type induced-subgraph fragility",
    )
    if celltype_subgraph_df is not None:
        plot_celltype_subgraph_fragility(celltype_subgraph_df, os.path.join(out_dir, "celltype_subgraph_fragility.png"))

    celltype_hub_ko_df = load_optional_csv(
        hub_removal_dir, "celltype_hub_knockout.csv",
        required_cols={"hub_definition", "cell_type", "lcc_fraction_after"},
        label="cell-type x hub-definition knockout",
    )
    if celltype_hub_ko_df is not None:
        plot_celltype_hub_knockout_heatmap(celltype_hub_ko_df, os.path.join(out_dir, "celltype_hub_knockout_heatmap.png"))

    artic_df = load_optional_csv(
        hub_removal_dir, "hub_articulation_overlap.csv",
        required_cols={"hub_definition", "frac_hub_nodes_that_are_articulation_points"},
        label="hub/articulation-point overlap",
    )
    if artic_df is not None:
        plot_hub_articulation_overlap(artic_df, os.path.join(out_dir, "hub_articulation_overlap.png"))

    participation_df = load_optional_csv(
        hub_removal_dir, "hub_layer_participation_summary.csv",
        required_cols={"hub_definition", "mean_frac_inter_layer", "module_wide_mean_frac_inter_layer"},
        label="layer participation",
    )
    if participation_df is not None:
        plot_layer_participation(participation_df, os.path.join(out_dir, "hub_layer_participation.png"))

    celltype_df = load_optional_csv(
        hub_removal_dir, "celltype_knockout_summary.csv",
        required_cols={"cell_type", "lcc_fraction_after", "frac_removed"},
        label="per-cell-type knockout",
    )
    if celltype_df is not None:
        plot_celltype_knockout(celltype_df, os.path.join(out_dir, "celltype_knockout.png"))

    seed_split_df = load_optional_csv(
        hub_removal_dir, "hub_knockout_by_seed_status.csv",
        required_cols={"hub_definition", "subset", "lcc_fraction_after"},
        label="seed-vs-bridging knockout split",
    )
    if seed_split_df is not None:
        plot_seed_vs_bridging(seed_split_df, os.path.join(out_dir, "hub_knockout_seed_vs_bridging.png"))

    ablation_df = load_optional_csv(
        hub_removal_dir, "ablation_f_c_f90.csv",
        required_cols={"mode", "cascading", "f_c", "f_90"},
        label="layer-ablation attack curves (--ablation-exclude-celltypes)",
    )
    if ablation_df is not None:
        observed_full_df = load_optional_csv(
            hub_removal_dir, "f_c_f90_observed.csv",
            required_cols={"mode", "cascading", "f_c", "f_90"},
            label="full-module f_c/f_90 (for ablation comparison)",
        )
        if observed_full_df is not None:
            plot_ablation_comparison(ablation_df, observed_full_df,
                                      os.path.join(out_dir, "layer_ablation_comparison.png"))

    node_roles_df = load_optional_csv(
        hub_removal_dir, "node_community_roles.csv",
        required_cols={"participation_coefficient", "within_module_z", "role"},
        label="community roles (Guimera-Amaral)",
    )
    if node_roles_df is not None:
        plot_community_roles_scatter(node_roles_df, os.path.join(out_dir, "community_roles_scatter.png"))

    hub_role_df = load_optional_csv(
        hub_removal_dir, "hub_sets_community_role_summary.csv",
        required_cols={"hub_definition"},
        label="hub-set community-role composition",
    )
    if hub_role_df is not None:
        plot_hub_community_role_composition(hub_role_df, os.path.join(out_dir, "hub_community_role_composition.png"))

    seed_comp_conn_df = load_optional_csv(
        hub_removal_dir, "seed_component_connectivity.csv",
        required_cols={"component_i", "component_j", "node_disjoint_paths"},
        label="seed-component connectivity",
    )
    if seed_comp_conn_df is not None:
        plot_seed_component_connectivity(seed_comp_conn_df, os.path.join(out_dir, "seed_component_connectivity.png"))

    # --- Layer-preserving null (step12b) 3-way comparison -------------------
    # step12b writes to <output-dir>/layer_preserving_null/, a sibling of
    # step12's <output-dir>/hub_removal/ -- guess that sibling path unless
    # the CSVs simply aren't there, in which case skip this whole section.
    print("\n--- Layer-preserving null comparison (step12b) ---")
    layer_null_dir = os.path.join(os.path.dirname(os.path.normpath(hub_removal_dir)), "layer_preserving_null")
    three_way_fc_df = load_optional_csv(
        layer_null_dir, "three_way_fc_f90_comparison.csv",
        required_cols={"mode", "cascading", "statistic", "observed", "layer_preserving_null_median"},
        label="3-way f_c/f_90 comparison (step12b_layer_preserving_null.py)",
    )
    if three_way_fc_df is not None:
        plot_three_way_fc(three_way_fc_df, os.path.join(out_dir, "three_way_fc_static.png"),
                           statistic="f_c", cascading=False)
        plot_three_way_fc(three_way_fc_df, os.path.join(out_dir, "three_way_fc_cascading.png"),
                           statistic="f_c", cascading=True)

    three_way_topo_df = load_optional_csv(
        layer_null_dir, "three_way_topology_comparison.csv",
        required_cols={"statistic", "observed", "layer_preserving_null_median"},
        label="3-way topology comparison (step12b_layer_preserving_null.py)",
    )
    if three_way_topo_df is not None:
        plot_three_way_topology(three_way_topo_df, os.path.join(out_dir, "three_way_topology_structural.png"),
                                 stat_subset=TOPOLOGY_STRUCTURAL_STATS)
        plot_three_way_topology(three_way_topo_df, os.path.join(out_dir, "three_way_topology_connectivity.png"),
                                 stat_subset=TOPOLOGY_CONNECTIVITY_STATS)

    print(f"\nAll available plots saved to: {out_dir}")