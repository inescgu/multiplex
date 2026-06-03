#!/usr/bin/env python3
"""
filter_lr_pairs.py
==================
Filter an lr_pairs file to only keep pairs where both the ligand
and receptor genes appear in the consolidated multiplex layers.

This is necessary after consolidate_layers.py removes cell types,
because inter-layer edges in the multiplex are only meaningful if
both endpoints exist in the surviving layers.

Usage
-----
python filter_lr_pairs.py \
    --lr-pairs      lr_pairs.tsv \
    --wdir          path/to/consolidated_wdir \
    --output        lr_pairs_filtered.tsv \
    [--ligand-col   human_ligand] \
    [--receptor-col human_receptor] \
    [--sep          "\\t"]

What it does
------------
1. Reads consolidated config.yml  → gets list of surviving layer TSVs
2. Builds gene_universe           → all genes present in those layers
3. Filters lr_pairs               → keeps rows where BOTH ligand AND
                                    receptor are in gene_universe
4. Reports what was dropped and why
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Build gene universe from consolidated layers
# ---------------------------------------------------------------------------

def build_gene_universe(wdir: Path) -> set[str]:
    """
    Read config.yml from the consolidated wdir and collect every gene
    symbol that appears in any surviving layer TSV.
    """
    config_path = wdir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yml found in {wdir}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    layer_paths: list[str] = (
        cfg["multiplex"]["gene_multiplex"]["layers"]
    )

    gene_universe: set[str] = set()
    missing_files: list[str] = []

    print(f"Reading {len(layer_paths)} layer TSV(s) from config …")
    for rel_path in sorted(layer_paths):
        abs_path = wdir / rel_path
        if not abs_path.exists():
            missing_files.append(rel_path)
            continue

        df = pd.read_csv(
            abs_path,
            sep="\t",
            header=None,
            names=["g1", "g2"],
            dtype=str,
        )
        genes = (
            set(df["g1"].str.strip().str.upper()) |
            set(df["g2"].str.strip().str.upper())
        )
        gene_universe.update(genes)
        print(f"  {Path(rel_path).stem:<50} {len(genes):>8,} genes")

    if missing_files:
        print(f"\n⚠  {len(missing_files)} layer file(s) listed in config "
              f"but not found on disk:")
        for p in missing_files:
            print(f"     {p}")

    print(f"\nGene universe: {len(gene_universe):,} unique genes\n")
    return gene_universe


# ---------------------------------------------------------------------------
# Optional: cell-type-aware filtering
# ---------------------------------------------------------------------------

def build_celltype_gene_map(wdir: Path) -> dict[str, set[str]]:
    """
    Build a map of  cell_type_stem → set of genes  from layer_info.json
    and the actual TSV files.  Only intra-type layers are included.

    This is optional — used only when you want to report WHICH cell types
    lost coverage, not just which genes disappeared.
    """
    layer_info_path = wdir / "layer_info.json"
    if not layer_info_path.exists():
        return {}

    with open(layer_info_path) as f:
        layer_info: dict = json.load(f)

    config_path = wdir / "config.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    layer_paths: list[str] = cfg["multiplex"]["gene_multiplex"]["layers"]
    stem_to_path: dict[str, Path] = {
        Path(p).stem: wdir / p for p in layer_paths
    }

    ct_gene_map: dict[str, set[str]] = {}

    for stem, info in layer_info.items():
        if info.get("type") != "intra":
            continue
        path = stem_to_path.get(stem)
        if not path or not path.exists():
            continue
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["g1", "g2"], dtype=str,
        )
        genes = (
            set(df["g1"].str.strip().str.upper()) |
            set(df["g2"].str.strip().str.upper())
        )
        ct_label = info["cell_types"][0] if info["cell_types"] else stem
        ct_gene_map[ct_label] = genes

    return ct_gene_map


# ---------------------------------------------------------------------------
# Filter lr_pairs
# ---------------------------------------------------------------------------

def filter_lr_pairs(
    lr_path: str,
    gene_universe: set[str],
    ligand_col: str,
    receptor_col: str,
    sep: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Keep only rows where both ligand AND receptor are in gene_universe.

    Reports:
      - total pairs in input
      - pairs kept
      - pairs dropped because ligand missing
      - pairs dropped because receptor missing
      - pairs dropped because both missing
    """
    df = pd.read_csv(lr_path, sep=sep)

    # Validate columns
    for col in [ligand_col, receptor_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {lr_path}.\n"
                f"Available columns: {list(df.columns)}"
            )

    # Normalise to upper-case
    df[ligand_col]   = df[ligand_col].astype(str).str.strip().str.upper()
    df[receptor_col] = df[receptor_col].astype(str).str.strip().str.upper()

    total = len(df)

    lig_in  = df[ligand_col].isin(gene_universe)
    rec_in  = df[receptor_col].isin(gene_universe)

    kept_mask       = lig_in  &  rec_in
    lig_only_mask   = lig_in  & ~rec_in
    rec_only_mask   = ~lig_in &  rec_in
    both_miss_mask  = ~lig_in & ~rec_in

    df_kept = df[kept_mask].copy()

    # ---- reporting -------------------------------------------------------
    print("=" * 60)
    print("LR-pair filtering report")
    print("=" * 60)
    print(f"Input pairs              : {total:>8,}")
    print(f"Kept (both in universe)  : {kept_mask.sum():>8,}")
    print(f"Dropped — ligand missing : {(~lig_in & rec_in).sum():>8,}")   # noqa
    print(f"Dropped — receptor miss  : {(lig_in & ~rec_in).sum():>8,}")   # noqa
    print(f"Dropped — both missing   : {both_miss_mask.sum():>8,}")
    print(f"Total dropped            : {total - kept_mask.sum():>8,}")
    print()

    # Genes that disappeared
    missing_ligands = (
        set(df.loc[~lig_in,  ligand_col].unique())
    )
    missing_receptors = (
        set(df.loc[~rec_in, receptor_col].unique())
    )
    if missing_ligands:
        print(f"Ligand genes NOT in universe "
              f"({len(missing_ligands):,} unique):")
        shown = sorted(missing_ligands)[:20]
        print(f"  {shown}"
              + (" …" if len(missing_ligands) > 20 else ""))
        print()
    if missing_receptors:
        print(f"Receptor genes NOT in universe "
              f"({len(missing_receptors):,} unique):")
        shown = sorted(missing_receptors)[:20]
        print(f"  {shown}"
              + (" …" if len(missing_receptors) > 20 else ""))
        print()

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_csv(out, sep=sep, index=False)
    print(f"Filtered LR pairs saved → {out}")
    print(f"  {len(df_kept):,} pairs retained out of {total:,}\n")

    return df_kept


# ---------------------------------------------------------------------------
# Validation: check seeds still have LR coverage
# ---------------------------------------------------------------------------

def check_seed_lr_coverage(
    df_filtered: pd.DataFrame,
    seeds_path: Path,
    ligand_col: str,
    receptor_col: str,
) -> None:
    """
    Warn if any seed gene has lost ALL its LR connections after filtering.
    """
    if not seeds_path.exists():
        return

    seeds = {
        s.strip().upper()
        for s in seeds_path.read_text().splitlines()
        if s.strip()
    }
    if not seeds:
        return

    lr_genes = (
        set(df_filtered[ligand_col].str.upper()) |
        set(df_filtered[receptor_col].str.upper())
    )

    seeds_with_lr    = seeds & lr_genes
    seeds_without_lr = seeds - lr_genes

    print("Seed LR-coverage check:")
    print(f"  Seeds total         : {len(seeds):,}")
    print(f"  Seeds with LR edge  : {len(seeds_with_lr):,}")
    print(f"  Seeds with NO LR    : {len(seeds_without_lr):,}")

    if seeds_without_lr:
        print(f"  ⚠  Seeds with no LR connection after filter:")
        for s in sorted(seeds_without_lr):
            print(f"       {s}")
    else:
        print("  ✓  All seeds retain at least one LR connection")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter lr_pairs to genes present in consolidated "
            "MultiXrank layers"
        )
    )
    parser.add_argument(
        "--lr-pairs", required=True,
        help="Path to original lr_pairs file (TSV or CSV)",
    )
    parser.add_argument(
        "--wdir", required=True,
        help="Consolidated MultiXrank working directory "
             "(must contain config.yml)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for filtered lr_pairs file",
    )
    parser.add_argument(
        "--ligand-col", default="human_ligand",
        help="Column name for ligand gene symbols (default: human_ligand)",
    )
    parser.add_argument(
        "--receptor-col", default="human_receptor",
        help="Column name for receptor gene symbols "
             "(default: human_receptor)",
    )
    parser.add_argument(
        "--sep", default="\t",
        help=r"Field separator for lr_pairs file (default: \t). "
             r"Use ',' for CSV",
    )
    args = parser.parse_args()

    wdir = Path(args.wdir)

    # 1. Build gene universe from surviving layers
    gene_universe = build_gene_universe(wdir)

    # 2. Filter
    df_filtered = filter_lr_pairs(
        lr_path      = args.lr_pairs,
        gene_universe= gene_universe,
        ligand_col   = args.ligand_col,
        receptor_col = args.receptor_col,
        sep          = args.sep,
        output_path  = args.output,
    )

    # 3. Seed coverage check (optional but useful)
    seeds_path = wdir / "seeds" / "seeds.txt"
    check_seed_lr_coverage(
        df_filtered,
        seeds_path,
        ligand_col   = args.ligand_col,
        receptor_col = args.receptor_col,
    )

    # 4. Summary of what downstream scripts need to pick up
    print("=" * 60)
    print("Next steps")
    print("=" * 60)
    print(f"  Filtered lr_pairs  → {args.output}")
    print(f"  Pass this file to whatever rebuilds the multiplex pkl")
    print(f"  before re-running dm_step1.py")
    print()
    print("  dm_step1.py and dm_step2.py will auto-regenerate:")
    print("    multiplex/*.tsv")
    print("    layer_info.json")
    print("    cell_type_metadata.json")
    print("    config.yml")
    print("    gene_reference_graph.pkl")
    print("    seeds/seeds.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()