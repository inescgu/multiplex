#!/usr/bin/env python3
"""
Three separate step-wise visualizations of the full PPI network using
cartoGRAPHs-style UMAP layout, adapted for multiplex/cell-type-aware
disease modules.

Shows:
1. All nodes in grey (full reference network baseline)
2. Seed genes colored (showing starting point)
3. Full disease module colored by dominant cell type
"""

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize']      = 10000
matplotlib.rcParams['path.simplify']           = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import umap
from collections import defaultdict

# ============================================================================
# CONFIGURATION — edit these paths before running
# ============================================================================
REFERENCE_GRAPH_PKL = 'multixrank_input_consolidated/gene_reference_graph.pkl'   # or gene_reference_graph.pkl
MODULE_DIR          = Path('disease_module_output_v2')        # --output-dir from builder
PLOTS_DIR           = Path('plots_multiplex')

# ============================================================================
# CELL-TYPE COLOR PALETTE
# Add / modify entries to match your cell types.
# Keys should match the "dominant_cell_type" values in disease_module.csv.
# ============================================================================
CELL_TYPE_COLORS = {
    'thymic_merged':   '#E63946',
    'A-Propr.EPHA3':   '#F4A261',
    'C-NP.MRGPRX1_GFRA2':           '#2A9D8F',
    'C-NP.SST':     '#457B9D',
    'A-Propr.PCDH8':            '#8338EC',
    'A-PEP.KIT':          '#FB8500',
    'interstitial_cell_of_Cajal':                 '#06D6A0',
    'platelet':              '#aaaaaa',
}

# Fallback palette for cell types not listed above
_FALLBACK_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3',
    '#ff7f00','#a65628','#f781bf','#999999',
]

# Fixed colors for non-module nodes
OTHER_COLOR    = '#e8e8e8'
SEED_COLOR     = '#1B88A7'   # used in step 2 where CT not yet shown
EDGE_COLOR_HEX = '#cccccc'

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

# Reference graph (full network used for layout)
with open(REFERENCE_GRAPH_PKL, 'rb') as f:
    G_full = pickle.load(f)

# Normalise node names to uppercase strings (mirrors the builder)
mapping = {n: str(n).strip().upper() for n in G_full.nodes()}
G_full  = nx.relabel_nodes(G_full, mapping)

# Disease module outputs
module_csv  = MODULE_DIR / 'disease_module.csv'
module_df   = pd.read_csv(module_csv)
module_df['gene'] = module_df['gene'].astype(str).str.strip().str.upper()

# Module graph (subgraph of reference)
with open(MODULE_DIR / 'module_graph.pkl', 'rb') as f:
    module_graph = pickle.load(f)
mapping_m   = {n: str(n).strip().upper() for n in module_graph.nodes()}
module_graph = nx.relabel_nodes(module_graph, mapping_m)

# Gene sets
seed_genes_in_net = set(module_df.loc[module_df['is_seed'] == True, 'gene'])
module_genes      = set(module_df['gene'])
non_module_genes  = module_genes - seed_genes_in_net   # non-seed module members
other_genes       = set(G_full.nodes()) - module_genes

# Cell-type annotation per gene
if 'dominant_cell_type' in module_df.columns:
    gene_to_ct = dict(zip(module_df['gene'], module_df['dominant_cell_type']))
else:
    gene_to_ct = {g: 'unknown' for g in module_genes}

# Collect all cell types present and assign colors
all_ct = sorted(set(gene_to_ct.values()))
_extra = [ct for ct in all_ct if ct not in CELL_TYPE_COLORS]
for i, ct in enumerate(_extra):
    CELL_TYPE_COLORS[ct] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]

print(f"  Full network:       {G_full.number_of_nodes():,} nodes, "
      f"{G_full.number_of_edges():,} edges")
print(f"  Module genes:       {len(module_genes):,}")
print(f"  Seed genes:         {len(seed_genes_in_net):,}")
print(f"  Non-seed module:    {len(non_module_genes):,}")
print(f"  Other nodes:        {len(other_genes):,}")
print(f"  Cell types present: {all_ct}")

# ============================================================================
# LAYOUT — load from cache or compute
# ============================================================================
PLOTS_DIR.mkdir(exist_ok=True)
layout_cache = PLOTS_DIR / 'ppi_umap_layout.pkl'


def build_feature_matrix(G):
    """Compute cartoGRAPHs-style node feature matrix."""
    print("  Computing node features...")
    nodes = list(G.nodes())
    degree      = dict(G.degree())
    clustering  = nx.clustering(G)

    print("    betweenness (k=300)...")
    betweenness = nx.betweenness_centrality(
        G, k=min(300, len(nodes)), normalized=True, seed=42)

    print("    closeness...")
    closeness   = nx.closeness_centrality(G)

    print("    eigenvector...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in nodes}

    print("    triangles...")
    triangles   = nx.triangles(G)

    feat = np.array([
        [degree[n], clustering[n], betweenness[n],
         closeness[n], eigenvector[n], triangles[n]]
        for n in nodes
    ], dtype=float)

    for col in range(feat.shape[1]):
        lo, hi = feat[:, col].min(), feat[:, col].max()
        if hi > lo:
            feat[:, col] = (feat[:, col] - lo) / (hi - lo)

    return nodes, feat


def compute_umap_layout(G, n_neighbors=15, min_dist=0.1, random_state=42):
    print("\nComputing cartoGRAPHs-style UMAP layout...")
    nodes, feat = build_feature_matrix(G)

    print(f"  Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric='euclidean',
        init='spectral', random_state=random_state,
        low_memory=True,
    )
    emb = reducer.fit_transform(feat)

    for dim in range(2):
        lo, hi = emb[:, dim].min(), emb[:, dim].max()
        emb[:, dim] = (emb[:, dim] - lo) / (hi - lo) * 100

    pos = {node: emb[i] for i, node in enumerate(nodes)}
    print("  Layout complete.")
    return pos


if layout_cache.exists():
    print(f"Loading cached layout from {layout_cache}...")
    with open(layout_cache, 'rb') as f:
        pos = pickle.load(f)
else:
    pos = compute_umap_layout(G_full)
    with open(layout_cache, 'wb') as f:
        pickle.dump(pos, f)
    print(f"Layout cached → {layout_cache}")

# ============================================================================
# EDGE DRAWING HELPERS
# ============================================================================

def collect_edge_segments(G, pos, panel, seed_genes, module_genes):
    """
    Return edge segments grouped by (alpha, linewidth, zorder).

    Panels
    ------
    'all_grey'        : uniform light grey edges
    'seeds_only'      : seed–seed edges highlighted, rest faded
    'module_expanded' : module edges highlighted by membership
    """
    groups = defaultdict(lambda: {'xs': [], 'ys': []})

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        xu, yu = pos[u]
        xv, yv = pos[v]

        u_seed   = u in seed_genes
        v_seed   = v in seed_genes
        u_module = u in module_genes
        v_module = v in module_genes

        if panel == 'all_grey':
            key = (0.05, 0.20, 0)

        elif panel == 'seeds_only':
            if u_seed and v_seed:
                key = (0.75, 1.0, 2)
            elif u_seed or v_seed:
                key = (0.20, 0.5, 1)
            else:
                key = (0.04, 0.2, 0)

        else:  # module_expanded
            if u_seed and v_seed:
                key = (0.80, 1.2, 3)
            elif (u_seed and v_module) or (v_seed and u_module):
                key = (0.50, 0.7, 2)
            elif u_module and v_module:
                key = (0.30, 0.4, 1)
            else:
                key = (0.04, 0.2, 0)

        groups[key]['xs'] += [xu, xv, None]
        groups[key]['ys'] += [yu, yv, None]

    return groups


def draw_edge_group_chunked(ax, xs, ys, color, alpha, width, zorder,
                             chunk_size=5000):
    """Draw edges in chunks to avoid Agg cell-block overflow."""
    stride = chunk_size * 3
    for start in range(0, len(xs), stride):
        ax.plot(
            xs[start:start + stride],
            ys[start:start + stride],
            color=color, alpha=alpha, linewidth=width,
            solid_capstyle='round', zorder=zorder,
            rasterized=True,
        )

# ============================================================================
# SINGLE PANEL DRAWING
# ============================================================================

def scatter(ax, nodelist, pos, color, size, alpha, lw, ec, zorder):
    if not nodelist:
        return
    xs = [pos[n][0] for n in nodelist if n in pos]
    ys = [pos[n][1] for n in nodelist if n in pos]
    if not xs:
        return
    ax.scatter(xs, ys, c=color, s=size, alpha=alpha,
               linewidths=lw, edgecolors=ec,
               zorder=zorder, rasterized=True)


def scatter_ct(ax, nodelist, pos, gene_to_ct, ct_colors,
               size, alpha, lw, ec, zorder):
    """
    Scatter module genes coloured by dominant cell type.
    Groups genes by cell type for efficiency.
    """
    by_ct = defaultdict(list)
    for g in nodelist:
        ct = gene_to_ct.get(g, 'unknown')
        by_ct[ct].append(g)

    for ct, genes in by_ct.items():
        color = ct_colors.get(ct, '#aaaaaa')
        xs = [pos[g][0] for g in genes if g in pos]
        ys = [pos[g][1] for g in genes if g in pos]
        if xs:
            ax.scatter(xs, ys, c=color, s=size, alpha=alpha,
                       linewidths=lw, edgecolors=ec,
                       zorder=zorder, rasterized=True,
                       label=ct)   # label used for legend building


def draw_panel(ax, panel, title, subtitle, pos,
               G, seed_genes, module_genes, non_module_genes,
               other_genes, gene_to_ct):

    ax.set_facecolor('white')

    # ---- Edges ----
    print(f"    Collecting edges for panel '{panel}'...")
    edge_groups = collect_edge_segments(G, pos, panel, seed_genes, module_genes)
    print(f"    Drawing {len(edge_groups)} edge style groups (chunked)...")
    for (alpha, width, zorder), segs in sorted(
            edge_groups.items(), key=lambda x: x[0][2]):
        draw_edge_group_chunked(
            ax, segs['xs'], segs['ys'],
            color=EDGE_COLOR_HEX, alpha=alpha, width=width,
            zorder=zorder, chunk_size=5000,
        )

    # ---- Nodes (back → front) ----
    if panel == 'all_grey':
        scatter(ax, list(other_genes),       pos, OTHER_COLOR, 12, 0.70, 0,   'none',  4)
        scatter(ax, list(module_genes),      pos, OTHER_COLOR, 12, 0.70, 0,   'none',  4)

    elif panel == 'seeds_only':
        scatter(ax, list(other_genes),       pos, OTHER_COLOR, 12, 0.70, 0,   'none',  4)
        scatter(ax, list(non_module_genes),  pos, OTHER_COLOR, 12, 0.70, 0,   'none',  4)
        scatter(ax, list(seed_genes),        pos, SEED_COLOR,  55, 0.95, 0.4, 'white', 6)

    else:  # module_expanded — colour by cell type
        scatter(ax, list(other_genes), pos, OTHER_COLOR, 12, 0.70, 0, 'none', 3)

        # Non-seed module genes: cell-type colour, smaller
        scatter_ct(ax, list(non_module_genes), pos, gene_to_ct,
                   CELL_TYPE_COLORS, size=22, alpha=0.82,
                   lw=0.3, ec='white', zorder=5)

        # Seed genes on top: cell-type colour, larger + ring
        scatter_ct(ax, list(seed_genes), pos, gene_to_ct,
                   CELL_TYPE_COLORS, size=60, alpha=0.97,
                   lw=0.6, ec='white', zorder=6)

        # Extra white ring around seeds to make them pop
        scatter(ax, list(seed_genes), pos, 'none', 75, 1.0, 1.2, 'white', 5)

    # ---- Title / subtitle ----
    ax.set_title(title, fontsize=24, fontweight='bold', pad=16)
    ax.text(0.5, -0.02, subtitle,
            transform=ax.transAxes, fontsize=14,
            ha='center', va='top', color='#555555')
    ax.axis('off')

    # ---- Legend ----
    n_total  = G.number_of_nodes()
    n_seed   = len(seed_genes)
    n_module = len(module_genes)

    if panel == 'all_grey':
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=OTHER_COLOR, markersize=10,
                   label=f'All proteins  (n={n_total:,})'),
        ]

    elif panel == 'seeds_only':
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=SEED_COLOR, markersize=12,
                   label=f'Seed genes  (n={n_seed:,})'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=OTHER_COLOR, markersize=10,
                   label=f'Other proteins  (n={n_total - n_seed:,})'),
        ]

    else:
        # Cell-type handles for module genes
        ct_counts = defaultdict(int)
        for g in module_genes:
            ct_counts[gene_to_ct.get(g, 'unknown')] += 1

        handles = []
        for ct in sorted(ct_counts, key=lambda x: -ct_counts[x]):
            color = CELL_TYPE_COLORS.get(ct, '#aaaaaa')
            handles.append(
                Patch(facecolor=color, edgecolor='white', linewidth=0.5,
                      label=f'{ct}  (n={ct_counts[ct]:,})')
            )
        # Separator + background
        handles += [
            Line2D([0], [0], linestyle='none'),   # spacer
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=OTHER_COLOR, markersize=10,
                   label=f'Non-module  (n={n_total - n_module:,})'),
        ]
        # Seed marker style note
        handles.insert(0,
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='white', markeredgecolor='#555555',
                   markersize=13, markeredgewidth=1.2,
                   label=f'◎ = seed gene  (n={n_seed:,})'),
        )

    ax.legend(handles=handles, loc='lower left',
              frameon=True, framealpha=0.92,
              edgecolor='#dddddd', fontsize=11, borderpad=0.9)

# ============================================================================
# RENDER AND SAVE ALL THREE PANELS
# ============================================================================
panels = [
    (
        'all_grey',
        'Step 1: Full Reference Network',
        f'{G_full.number_of_nodes():,} proteins  ·  '
        f'{G_full.number_of_edges():,} interactions',
        'multiplex_step1_all_grey.png',
    ),
    (
        'seeds_only',
        'Step 2: Seed Gene Mapping',
        f'{len(seed_genes_in_net):,} disease seed genes highlighted',
        'multiplex_step2_seeds.png',
    ),
    (
        'module_expanded',
        'Step 3: Cell-Type-Aware Disease Module',
        f'{len(module_genes):,} module genes  ·  '
        f'{len(seed_genes_in_net):,} seeds  ·  '
        f'{len(all_ct)} cell types',
        'multiplex_step3_module.png',
    ),
]

for panel_id, title, subtitle, filename in panels:
    print(f"\n{'='*60}")
    print(f"Rendering: {title}")
    print(f"{'='*60}")

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fig.patch.set_facecolor('white')

    draw_panel(
        ax, panel_id, title, subtitle, pos,
        G_full,
        seed_genes_in_net,
        module_genes,
        non_module_genes,
        other_genes,
        gene_to_ct,
    )

    plt.tight_layout()
    save_path = PLOTS_DIR / filename
    print(f"  Saving → {save_path}...")
    fig.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")

print(f"\n{'='*60}")
print("All three multiplex plots saved successfully.")
print(f"{'='*60}")