#!/usr/bin/env python3
"""
Visualize Edge Overlaps in Integrated Multiplex
================================================
Generates UpSet plots, bar charts, and Venn diagrams showing:
- Edge type overlaps (PPI, pathway, ligand-receptor)
- Source pair overlaps
- Database overlaps (Reactome vs KEGG)
"""

import os
import pickle
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np

try:
    from upsetplot import UpSet, from_contents
except ImportError:
    print("Warning: upsetplot not installed. Install with: pip install upsetplot")
    UpSet = None

try:
    from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
except ImportError:
    print("Warning: matplotlib-venn not installed. Install with: pip install matplotlib-venn")
    venn2 = venn3 = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_multiplex(multiplex_path: str, registry_path: str) -> tuple:
    """Load the multiplex graph and node registry."""
    print(f"Loading multiplex from {multiplex_path}...")
    with open(multiplex_path, "rb") as f:
        G = pickle.load(f)
    
    registry_df = pd.read_csv(registry_path, index_col="node_id")
    print(f"  Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, registry_df


def extract_edge_metadata(G: nx.Graph) -> dict:
    """Extract comprehensive edge metadata."""
    metadata = {
        "edges_by_type": defaultdict(set),
        "edges_by_source_pair": defaultdict(set),
        "edges_by_db": defaultdict(set),
        "edges_by_pathway": defaultdict(set),
        "edge_type_per_edge": {},
        "source_pair_per_edge": {},
        "db_per_edge": {},
    }
    
    for u, v, d in G.edges(data=True):
        edge_key = (min(u, v), max(u, v))  # Normalize edge representation
        
        # Edge type
        etype = d.get("edge_type", "unknown")
        metadata["edges_by_type"][etype].add(edge_key)
        metadata["edge_type_per_edge"][edge_key] = etype
        
        # Source pair
        src_u = G.nodes[u].get("source", "unknown")
        src_v = G.nodes[v].get("source", "unknown")
        src_pair = tuple(sorted([src_u, src_v]))
        metadata["edges_by_source_pair"][src_pair].add(edge_key)
        metadata["source_pair_per_edge"][edge_key] = src_pair
        
        # Database (for pathway edges)
        if etype == "pathway":
            db = d.get("pathway_db", "unknown")
            metadata["edges_by_db"][db].add(edge_key)
            metadata["db_per_edge"][edge_key] = db
    
    return metadata


# ============================================================================
# UPSET PLOTS
# ============================================================================
def plot_upset_manual(sets: dict, title: str, filename: str, output_dir: str) -> None:
    """Manual UpSet-style intersection bar chart (replaces upsetplot)."""
    from itertools import combinations, chain

    set_names = sorted(sets.keys())
    # Build all non-empty intersections
    rows = []
    for r in range(1, len(set_names) + 1):
        for combo in combinations(set_names, r):
            intersection = set.intersection(*(sets[k] for k in combo))
            # Exclude elements that appear in sets outside this combo
            exclusive = intersection - set.union(*(sets[k] for k in set_names if k not in combo), set())
            if len(intersection) > 0:
                rows.append({"combo": combo, "count": len(intersection)})

    rows.sort(key=lambda x: -x["count"])
    labels = [" ∩ ".join(r["combo"]) for r in rows]
    counts = [r["count"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.4 + 2)))
    colors = sns.color_palette("husl", len(rows))
    bars = ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of Edges", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


def plot_upset_edge_types(metadata: dict, output_dir: str) -> None:
    print("Generating intersection chart: Edge type overlaps...")
    plot_upset_manual(metadata["edges_by_type"], "Edge Type Overlaps", "upset_edge_types.png", output_dir)

def plot_upset_source_pairs(metadata: dict, output_dir: str) -> None:
    print("Generating intersection chart: Source pair overlaps...")
    sets = {str(pair): edges for pair, edges in metadata["edges_by_source_pair"].items()}
    plot_upset_manual(sets, "Source Pair Overlaps", "upset_source_pairs.png", output_dir)

def plot_upset_db_overlaps(metadata: dict, output_dir: str) -> None:
    print("Generating intersection chart: Database overlaps...")
    if len(metadata["edges_by_db"]) < 2:
        print("  Skipped: Only one database found")
        return
    plot_upset_manual(metadata["edges_by_db"], "Database Overlaps", "upset_database_overlaps.png", output_dir)


# ============================================================================
# VENN DIAGRAMS
# ============================================================================

def plot_venn_edge_types(metadata: dict, output_dir: str) -> None:
    """Venn diagram: Edge type overlaps (2-way and 3-way)."""
    if venn2 is None or venn3 is None:
        print("Skipping Venn diagrams (matplotlib-venn not installed)")
        return
    
    print("Generating Venn diagrams: Edge type overlaps...")
    
    edge_types = metadata["edges_by_type"]
    etypes = sorted(edge_types.keys())
    
    if len(etypes) == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        venn2([edge_types[etypes[0]], edge_types[etypes[1]]], 
              set_labels=etypes, ax=ax)
        venn2_circles([edge_types[etypes[0]], edge_types[etypes[1]]], ax=ax)
        plt.title("Edge Type Overlaps (2-way Venn)", fontsize=14, fontweight="bold")
        
        output_path = Path(output_dir) / "venn_edge_types_2way.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")
        plt.close()
    
    elif len(etypes) >= 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        venn3([edge_types[etypes[0]], edge_types[etypes[1]], edge_types[etypes[2]]], 
              set_labels=etypes[:3], ax=ax)
        venn3_circles([edge_types[etypes[0]], edge_types[etypes[1]], edge_types[etypes[2]]], ax=ax)
        plt.title("Edge Type Overlaps (3-way Venn)", fontsize=14, fontweight="bold")
        
        output_path = Path(output_dir) / "venn_edge_types_3way.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")
        plt.close()


def plot_venn_source_pairs(metadata: dict, output_dir: str) -> None:
    """Venn diagram: Top source pair overlaps."""
    if venn2 is None or venn3 is None:
        print("Skipping Venn diagrams (matplotlib-venn not installed)")
        return
    
    print("Generating Venn diagrams: Source pair overlaps...")
    
    source_pairs = metadata["edges_by_source_pair"]
    # Get top 3 source pairs by edge count
    top_pairs = sorted(source_pairs.items(), key=lambda x: -len(x[1]))[:3]
    
    if len(top_pairs) == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = [f"{pair[0]} ↔ {pair[1]}" for pair, _ in top_pairs]
        venn2([edges for _, edges in top_pairs], set_labels=labels, ax=ax)
        venn2_circles([edges for _, edges in top_pairs], ax=ax)
        plt.title("Top Source Pair Overlaps (2-way Venn)", fontsize=14, fontweight="bold")
        
        output_path = Path(output_dir) / "venn_source_pairs_2way.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")
        plt.close()
    
    elif len(top_pairs) >= 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = [f"{pair[0]} ↔ {pair[1]}" for pair, _ in top_pairs[:3]]
        venn3([edges for _, edges in top_pairs[:3]], set_labels=labels, ax=ax)
        venn3_circles([edges for _, edges in top_pairs[:3]], ax=ax)
        plt.title("Top Source Pair Overlaps (3-way Venn)", fontsize=14, fontweight="bold")
        
        output_path = Path(output_dir) / "venn_source_pairs_3way.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")
        plt.close()


def plot_venn_db_overlaps(metadata: dict, output_dir: str) -> None:
    """Venn diagram: Database overlaps."""
    if venn2 is None:
        print("Skipping Venn diagrams (matplotlib-venn not installed)")
        return
    
    print("Generating Venn diagrams: Database overlaps...")
    
    db_edges = metadata["edges_by_db"]
    if len(db_edges) < 2:
        print("  Skipped: Only one database found")
        return
    
    dbs = sorted(db_edges.keys())
    
    if len(dbs) == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        venn2([db_edges[dbs[0]], db_edges[dbs[1]]], set_labels=dbs, ax=ax)
        venn2_circles([db_edges[dbs[0]], db_edges[dbs[1]]], ax=ax)
        plt.title("Database Overlaps (Venn Diagram)", fontsize=14, fontweight="bold")
        
        output_path = Path(output_dir) / "venn_database_overlaps.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {output_path}")
        plt.close()


# ============================================================================
# BAR CHARTS
# ============================================================================

def plot_edge_type_distribution(metadata: dict, output_dir: str) -> None:
    """Bar chart: Edge type distribution."""
    print("Generating bar chart: Edge type distribution...")
    
    edge_types = metadata["edges_by_type"]
    data = {etype: len(edges) for etype, edges in edge_types.items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(data))
    bars = ax.bar(data.keys(), data.values(), color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Number of Edges", fontsize=12, fontweight="bold")
    ax.set_xlabel("Edge Type", fontsize=12, fontweight="bold")
    ax.set_title("Edge Type Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    output_path = Path(output_dir) / "bar_edge_type_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


def plot_source_pair_distribution(metadata: dict, output_dir: str) -> None:
    """Bar chart: Source pair distribution."""
    print("Generating bar chart: Source pair distribution...")
    
    source_pairs = metadata["edges_by_source_pair"]
    data = {f"{pair[0]} ↔ {pair[1]}": len(edges) 
            for pair, edges in sorted(source_pairs.items(), key=lambda x: -len(x[1]))}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(data))
    bars = ax.bar(range(len(data)), list(data.values()), color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    ax.set_ylabel("Number of Edges", fontsize=12, fontweight="bold")
    ax.set_xlabel("Source Pair", fontsize=12, fontweight="bold")
    ax.set_title("Edge Distribution by Source Pair", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.keys(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    output_path = Path(output_dir) / "bar_source_pair_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


def plot_edge_type_by_source_pair(metadata: dict, output_dir: str) -> None:
    """Stacked bar chart: Edge types by source pair."""
    print("Generating stacked bar chart: Edge types by source pair...")
    
    # Build matrix: source_pair x edge_type
    source_pairs = sorted(metadata["edges_by_source_pair"].keys())
    edge_types = sorted(metadata["edges_by_type"].keys())
    
    matrix = defaultdict(lambda: defaultdict(int))
    for edge_key, etype in metadata["edge_type_per_edge"].items():
        src_pair = metadata["source_pair_per_edge"][edge_key]
        matrix[src_pair][etype] += 1
    
    # Prepare data for stacked bar
    data_dict = {etype: [] for etype in edge_types}
    for src_pair in source_pairs:
        for etype in edge_types:
            data_dict[etype].append(matrix[src_pair][etype])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(source_pairs))
    width = 0.6
    colors = sns.color_palette("husl", len(edge_types))
    
    bottom = np.zeros(len(source_pairs))
    for i, etype in enumerate(edge_types):
        ax.bar(x, data_dict[etype], width, label=etype, bottom=bottom, 
               color=colors[i], edgecolor="black", linewidth=0.5)
        bottom += np.array(data_dict[etype])
    
    ax.set_ylabel("Number of Edges", fontsize=12, fontweight="bold")
    ax.set_xlabel("Source Pair", fontsize=12, fontweight="bold")
    ax.set_title("Edge Type Distribution by Source Pair", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pair[0]} ↔ {pair[1]}" for pair in source_pairs], 
                       rotation=45, ha="right")
    ax.legend(title="Edge Type", loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    output_path = Path(output_dir) / "bar_edge_type_by_source_pair.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


def plot_database_distribution(metadata: dict, output_dir: str) -> None:
    """Bar chart: Database distribution (for pathway edges)."""
    print("Generating bar chart: Database distribution...")
    
    db_edges = metadata["edges_by_db"]
    if len(db_edges) < 1:
        print("  Skipped: No database information found")
        return
    
    data = {db: len(edges) for db, edges in db_edges.items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(data))
    bars = ax.bar(data.keys(), data.values(), color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Number of Pathway Edges", fontsize=12, fontweight="bold")
    ax.set_xlabel("Database", fontsize=12, fontweight="bold")
    ax.set_title("Pathway Edge Distribution by Database", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    output_path = Path(output_dir) / "bar_database_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


# ============================================================================
# HEATMAPS
# ============================================================================

def plot_edge_type_source_pair_heatmap(metadata: dict, output_dir: str) -> None:
    """Heatmap: Edge type vs source pair."""
    print("Generating heatmap: Edge type vs source pair...")
    
    source_pairs = sorted(metadata["edges_by_source_pair"].keys())
    edge_types = sorted(metadata["edges_by_type"].keys())
    
    matrix = defaultdict(lambda: defaultdict(int))
    for edge_key, etype in metadata["edge_type_per_edge"].items():
        src_pair = metadata["source_pair_per_edge"][edge_key]
        matrix[src_pair][etype] += 1
    
    # Convert to DataFrame
    data = []
    for src_pair in source_pairs:
        row = {etype: matrix[src_pair][etype] for etype in edge_types}
        row["source_pair"] = f"{src_pair[0]} ↔ {src_pair[1]}"
        data.append(row)
    
    df = pd.DataFrame(data).set_index("source_pair")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={"label": "Edge Count"},
                linewidths=0.5, ax=ax)
    ax.set_title("Edge Type vs Source Pair Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Edge Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Source Pair", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    output_path = Path(output_dir) / "heatmap_edge_type_source_pair.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.close()


# ============================================================================
# OVERLAP STATISTICS TABLE
# ============================================================================

def generate_overlap_statistics(metadata: dict, output_dir: str) -> None:
    """Generate detailed overlap statistics table."""
    print("Generating overlap statistics table...")
    
    edge_types = metadata["edges_by_type"]
    etypes = sorted(edge_types.keys())
    
    # Pairwise overlaps
    overlap_data = []
    for et1, et2 in combinations(etypes, 2):
        overlap = len(edge_types[et1] & edge_types[et2])
        total_et1 = len(edge_types[et1])
        total_et2 = len(edge_types[et2])
        union = len(edge_types[et1] | edge_types[et2])
        jaccard = overlap / union if union > 0 else 0
        
        overlap_data.append({
            "edge_type_1": et1,
            "edge_type_2": et2,
            "overlap_count": overlap,
            "total_type_1": total_et1,
            "total_type_2": total_et2,
            "union_count": union,
            "jaccard_similarity": f"{jaccard:.4f}",
            "overlap_pct_type1": f"{100*overlap/total_et1:.2f}%",
            "overlap_pct_type2": f"{100*overlap/total_et2:.2f}%",
        })
    
    overlap_df = pd.DataFrame(overlap_data)
    overlap_path = Path(output_dir) / "overlap_statistics_edge_types.csv"
    overlap_df.to_csv(overlap_path, index=False)
    print(f"  Saved → {overlap_path}")
    
    # Source pair overlaps
    source_pairs = metadata["edges_by_source_pair"]
    spairs = sorted(source_pairs.keys())
    
    spair_overlap_data = []
    for sp1, sp2 in combinations(spairs, 2):
        overlap = len(source_pairs[sp1] & source_pairs[sp2])
        total_sp1 = len(source_pairs[sp1])
        total_sp2 = len(source_pairs[sp2])
        union = len(source_pairs[sp1] | source_pairs[sp2])
        jaccard = overlap / union if union > 0 else 0
        
        spair_overlap_data.append({
            "source_pair_1": f"{sp1[0]} ↔ {sp1[1]}",
            "source_pair_2": f"{sp2[0]} ↔ {sp2[1]}",
            "overlap_count": overlap,
            "total_pair_1": total_sp1,
            "total_pair_2": total_sp2,
            "union_count": union,
            "jaccard_similarity": f"{jaccard:.4f}",
        })
    
    spair_overlap_df = pd.DataFrame(spair_overlap_data)
    spair_overlap_path = Path(output_dir) / "overlap_statistics_source_pairs.csv"
    spair_overlap_df.to_csv(spair_overlap_path, index=False)
    print(f"  Saved → {spair_overlap_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize edge overlaps in integrated multiplex"
    )
    parser.add_argument("--multiplex-pkl", required=True,
                        help="Path to integrated_multiplex.pkl")
    parser.add_argument("--registry-csv", required=True,
                        help="Path to node_registry.csv")
    parser.add_argument("--output-dir", default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EDGE OVERLAP VISUALIZATION")
    print("=" * 70 + "\n")
    
    # Load data
    G, registry_df = load_multiplex(args.multiplex_pkl, args.registry_csv)
    
    # Extract metadata
    print("Extracting edge metadata...")
    metadata = extract_edge_metadata(G)
    print(f"  Edge types: {list(metadata['edges_by_type'].keys())}")
    print(f"  Source pairs: {list(metadata['edges_by_source_pair'].keys())}")
    print(f"  Databases: {list(metadata['edges_by_db'].keys())}\n")
    
    # Generate visualizations
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    # UpSet plots
    plot_upset_edge_types(metadata, output_dir)
    plot_upset_source_pairs(metadata, output_dir)
    plot_upset_db_overlaps(metadata, output_dir)
    
    # Venn diagrams
    plot_venn_edge_types(metadata, output_dir)
    plot_venn_source_pairs(metadata, output_dir)
    plot_venn_db_overlaps(metadata, output_dir)
    
    # Bar charts
    plot_edge_type_distribution(metadata, output_dir)
    plot_source_pair_distribution(metadata, output_dir)
    plot_edge_type_by_source_pair(metadata, output_dir)
    plot_database_distribution(metadata, output_dir)
    
    # Heatmaps
    plot_edge_type_source_pair_heatmap(metadata, output_dir)
    
    # Statistics
    generate_overlap_statistics(metadata, output_dir)
    
    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()