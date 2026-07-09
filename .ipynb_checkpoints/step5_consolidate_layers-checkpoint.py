#!/usr/bin/env python3
"""
Build a MultiXrank working directory from a curated cell-type list.
Cell types are read from layers_categories.csv (same directory as script).
Layers are kept individually (one per cell type) unless --merge-by-category
is specified, in which case layers within the same category are merged.

Exclusions:
  --exclude-broad "Immune Tissue"   drop all rows where Broad Category matches
  --exclude-ai    "immune tissue"   drop all rows where ai category matches
  (both flags accept multiple values)
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent


def load_categories(
    csv_path: Path,
    exclude_broad: list[str] | None = None,
    exclude_ai: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load layers_categories.csv and apply optional exclusions.

    Expected columns (case-insensitive strip):
        Cell Type | Broad Category | ai category
    """
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Accept a leading index column produced by e.g. df.to_csv()
    if df.columns[0].strip() in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]

    required = {"Cell Type", "Broad Category - ai", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"layers_categories.csv is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Strip whitespace in string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    before = len(df)

    if exclude_broad:
        mask = df["Broad Category - ai"].str.lower().isin(
            [e.lower() for e in exclude_broad]
        )
        df = df[~mask]
        print(
            f"Excluded {before - len(df)} rows by Broad Category "
            f"({exclude_broad})"
        )
        before = len(df)

    if exclude_ai:
        mask = df["category"].str.lower().isin(
            [e.lower() for e in exclude_ai]
        )
        df = df[~mask]
        print(
            f"Excluded {before - len(df)} rows by ai category "
            f"({exclude_ai})"
        )

    print(f"Retained {len(df)} cell types from CSV")
    return df.reset_index(drop=True)


def stem_from_cell_type(cell_type: str) -> str:
    """
    Convert a Cell Type name to the TSV stem used in the multiplex dir.
    e.g.  'A-LTMR.TAC3'  →  'A-LTMR.TAC3_ppi_layer'
    (matches what script 4 produces)
    """
    safe_name = cell_type.replace(" ", "_").replace("/", "-").replace(",", "")
    return safe_name + "_ppi_layer"

def build_consolidation_map(df: pd.DataFrame, group_col: str) -> dict[str, list[str]]:
    """
    Build a consolidation map grouped by *group_col*.
    merged name = <group>_merged  (spaces → underscores)
    """
    consolidation: dict[str, list[str]] = {}
    for group, sub in df.groupby(group_col):
        merged_name = group.replace(" ", "_") + "_merged"
        stems = [stem_from_cell_type(ct) for ct in sub["Cell Type"]]
        consolidation[merged_name] = stems
    return consolidation

def safe_copy(src: Path, dst: Path) -> None:
    """Copy src to dst, skipping if they resolve to the same file."""
    try:
        if src.resolve() == dst.resolve():
            return
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def check_seed_coverage(output_wdir: Path, new_layer_paths: list[str]) -> None:
    seeds_path = output_wdir / "seeds" / "seeds.txt"
    if not seeds_path.exists():
        return

    seeds = {
        s.strip().upper()
        for s in seeds_path.read_text().splitlines()
        if s.strip()
    }

    covered: set[str] = set()
    for rel_path in new_layer_paths:
        abs_path = output_wdir / rel_path
        if not abs_path.exists():
            continue
        df = pd.read_csv(
            abs_path, sep="\t", header=None, names=["g1", "g2"], dtype=str
        )
        genes = set(df["g1"].str.strip().str.upper()) | set(
            df["g2"].str.strip().str.upper()
        )
        covered.update(seeds & genes)

    missing = seeds - covered
    if missing:
        print(f"\n⚠  WARNING: Seeds NOT covered by any layer: {missing}")
        print("   → They will cause MultiXrank errors.")
    else:
        print(f"\n✓  All {len(seeds)} seeds covered by consolidated layers")


def rebuild_layer_info(new_layer_paths: list[str], output_wdir: Path) -> None:
    layer_info: dict = {}

    for rel_path in new_layer_paths:
        stem = Path(rel_path).stem

        if stem in ("ligand_receptor_layer", "pathway_layer", "ppi_layer"):
            layer_info[stem] = {
                "type": "special",
                "cell_types": [],
                "edge_type": stem.replace("_layer", ""),
            }
        elif stem.endswith("_merged"):
            layer_info[stem] = {
                "type": "intra",
                "cell_types": [stem],
                "edge_type": "ppi",
            }
        else:
            cell_type = stem.replace("_layer", "")
            layer_info[stem] = {
                "type": "intra",
                "cell_types": [cell_type],
                "edge_type": "ppi",
            }

    out_path = output_wdir / "layer_info.json"
    with open(out_path, "w") as f:
        json.dump(layer_info, f, indent=2)

    n_intra = sum(1 for v in layer_info.values() if v["type"] == "intra")
    n_special = sum(1 for v in layer_info.values() if v["type"] == "special")
    print(f"Rebuilt layer_info.json → {out_path}")
    print(f"  {n_intra} intra layers, {n_special} special layers")


def merge_layers(
    wdir: Path,
    output_wdir: Path,
    allowed_stems: set[str],
    consolidation_map: dict | None,
    extra_keep: list[str],
    remove_full_network: bool = True,
) -> None:
    multiplex_dir = wdir / "multiplex" / "gene_multiplex"
    out_multiplex_dir = output_wdir / "multiplex" / "gene_multiplex"
    out_multiplex_dir.mkdir(parents=True, exist_ok=True)
    
    # ✓ CREATE SEEDS DIRECTORY FIRST
    seeds_out_dir = output_wdir / "seeds"
    seeds_out_dir.mkdir(parents=True, exist_ok=True)
    
    # ✓ THEN copy seeds
    seeds_src = wdir / "seeds" / "seeds.txt"
    if seeds_src.exists():
        safe_copy(seeds_src, seeds_out_dir / "seeds.txt")
        print(f"✓ Copied seeds.txt → {seeds_out_dir / 'seeds.txt'}")
    else:
        print(f"⚠  WARNING: seeds.txt not found at {seeds_src}")
        print("   → MultiXrank will fail! Check step 4 output.")
        
    for fname in [
        "layer_info.json",
        "cell_type_metadata.json",
        "gene_reference_graph.pkl",
        "gene_ppi_reference_graph.pkl",
    ]:
        if (wdir / fname).exists():
            safe_copy(wdir / fname, output_wdir / fname)  

    # ---- read original config -----------------------------------------
    with open(wdir / "config.yml") as f:
        cfg = yaml.safe_load(f)

    existing_layers: list[str] = cfg["multiplex"]["gene_multiplex"]["layers"]

    # stem → absolute path
    layer_file_map: dict[str, Path] = {
        Path(p).stem: wdir / p for p in existing_layers
    }

    print(f"\nInput layers : {len(existing_layers)}")
    print(f"CSV-allowed  : {len(allowed_stems)} cell-type stems")

    # ---- which stems are in scope? ------------------------------------
    # scope = CSV-allowed stems  UNION  extra_keep stems
    extra_keep_set = set(extra_keep)
    in_scope_stems = (allowed_stems & set(layer_file_map)) | (
        extra_keep_set & set(layer_file_map)
    )

    not_found = allowed_stems - set(layer_file_map)
    if not_found:
        print(
            f"\n⚠  {len(not_found)} CSV stems not found in input multiplex "
            f"(they will be silently skipped):"
        )
        for s in sorted(not_found):
            print(f"     {s}")

    new_layer_paths: list[str] = []

    # ---- Pass A: merged layers (optional) -----------------------------
    handled_stems: set[str] = set()

    if consolidation_map:
        print("\nMerging layers by category…")
        for merged_name, source_stems in consolidation_map.items():
            all_edges: set[tuple[str, str]] = set()
            found_sources: list[str] = []

            for stem in source_stems:
                if stem not in in_scope_stems:
                    continue          # excluded by CSV filter
                path = layer_file_map.get(stem)
                if path and path.exists():
                    df = pd.read_csv(
                        path, sep="\t", header=None,
                        names=["g1", "g2"], dtype=str,
                    )
                    for _, row in df.iterrows():
                        a, b = sorted(
                            [row["g1"].strip(), row["g2"].strip()]
                        )
                        all_edges.add((a, b))
                    found_sources.append(stem)
                    handled_stems.add(stem)

            if not all_edges:
                print(f"  ⚠  {merged_name}: no edges found, skipping")
                continue

            rel_path = (
                Path("multiplex") / "gene_multiplex" / f"{merged_name}.tsv"
            )
            abs_path = output_wdir / rel_path
            pd.DataFrame(
                sorted(all_edges), columns=["g1", "g2"]
            ).to_csv(abs_path, sep="\t", index=False, header=False)
            new_layer_paths.append(str(rel_path))
            print(
                f"  {merged_name}: {len(found_sources)} sources "
                f"→ {len(all_edges):,} unique edges"
            )

    # ---- Pass B: individual layers ------------------------------------
    print("\nCopying individual layers…")
    for stem, path in layer_file_map.items():
        if stem in handled_stems:
            continue
        if remove_full_network and stem == "full_network_layer":
            print(f"  Removing : {stem} (full_network)")
            continue

        # Keep if in CSV allowed stems
        if stem in allowed_stems:
            rel_path = (
                Path("multiplex") / "gene_multiplex" / f"{stem}.tsv"
            )
            abs_path = output_wdir / rel_path
            safe_copy(path, abs_path)
            new_layer_paths.append(str(rel_path))
            print(f"  Kept : {stem}")

    # ---- Pass C: extra_keep layers (ALWAYS add, regardless of CSV) ----
    print("\nCopying extra_keep layers…")
    for stem in extra_keep_set:
        if stem in handled_stems:
            continue
        path = layer_file_map.get(stem)
        if path is None:
            print(f"  Missing: {stem} (not found in multiplex)")
            continue

        rel_path = (
            Path("multiplex") / "gene_multiplex" / f"{stem}.tsv"
        )
        abs_path = output_wdir / rel_path
        safe_copy(path, abs_path)
        new_layer_paths.append(str(rel_path))
        print(f"  Kept : {stem} ← extra_keep")

    # ---- Pass C: write config -----------------------------------------
    n_new = len(new_layer_paths)
    if n_new == 0:
        print("\n⚠  ERROR: No layers to output. Check your filters and CSV.")
        return

    tau_value = round(1.0 / n_new, 6)

    new_cfg = {
        "seed": cfg["seed"],
        "self_loops": cfg.get("self_loops", 0),
        "r": cfg.get("r", 0.7),
        "multiplex": {
            "gene_multiplex": {
                "layers": sorted(new_layer_paths),
                "graph_type": ["00"] * n_new,
                "delta": cfg["multiplex"]["gene_multiplex"].get("delta", 0.5),
                "tau": [tau_value] * n_new,
            }
        },
    }

    with open(output_wdir / "config.yml", "w") as f:
        yaml.safe_dump(new_cfg, f, default_flow_style=False, sort_keys=False)

    rebuild_layer_info(new_layer_paths, output_wdir)
    check_seed_coverage(output_wdir, new_layer_paths)

    print(f"\n{'='*60}")
    print(f"Consolidation complete")
    print(f"  Input layers  : {len(existing_layers)}")
    print(f"  Output layers : {n_new}")
    print(f"  Reduction     : {len(existing_layers) - n_new} layers removed")
    print(f"  New config    → {output_wdir / 'config.yml'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Always present in the output regardless of what the CSV says
DEFAULT_EXTRA_KEEP = [
    "ppi_layer",
    "ligand_receptor_layer",
    "pathway_layer",
    "reactome_layer",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a MultiXrank multiplex from layers_categories.csv"
    )
    parser.add_argument(
        "--wdir", required=True,
        help="Original MultiXrank working directory",
    )
    parser.add_argument(
        "--output-wdir", required=True,
        help="New working directory with selected/consolidated layers",
    )
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR / "layers_categories.csv"),
        help="Path to layers_categories.csv  "
             "(default: same directory as this script)",
    )
    parser.add_argument(
        "--exclude-broad",
        nargs="+",
        metavar="CATEGORY",
        default=[],
        help='Exclude rows whose "Broad Category" matches any of these '
             '(case-insensitive). Example: --exclude-broad "Immune Tissue"',
    )
    parser.add_argument(
        "--exclude-ai",
        nargs="+",
        metavar="CATEGORY",
        default=[],
        help='Exclude rows whose "ai category" matches any of these '
             '(case-insensitive). Example: --exclude-ai "immune tissue"',
    )
    parser.add_argument(
        "--merge-by",
        choices=["broad", "ai", "none"],
        default="none",
        help=(
            "Merge layers within the same category into one layer. "
            '"broad" → merge by Broad Category, '
            '"ai"    → merge by ai category, '
            '"none"  → keep every cell type as its own layer (default).'
        ),
    )
    parser.add_argument(
        "--keep-full-network",
        action="store_true",
        help="Keep the full_network layer (not recommended)",
    )
    parser.add_argument(
        "--extra-keep",
        nargs="+",
        metavar="STEM",
        default=DEFAULT_EXTRA_KEEP,
        help=(
            "Additional layer stems to always include regardless of CSV "
            f"(default: {DEFAULT_EXTRA_KEEP})"
        ),
    )

    args = parser.parse_args()

    # 1. Load & filter CSV
    df = load_categories(
        Path(args.csv),
        exclude_broad=args.exclude_broad or None,
        exclude_ai=args.exclude_ai or None,
    )

    # 2. Convert Cell Type names → layer stems
    allowed_stems: set[str] = {stem_from_cell_type(ct) for ct in df["Cell Type"]}

    # 3. Optional consolidation map
    consolidation_map: dict | None = None
    if args.merge_by == "broad":
        consolidation_map = build_consolidation_map(df, "Broad Category")
        print(f"\nMerge-by=broad → {len(consolidation_map)} merged groups")
    elif args.merge_by == "ai":
        consolidation_map = build_consolidation_map(df, "ai category")
        print(f"\nMerge-by=ai → {len(consolidation_map)} merged groups")
    else:
        print("\nMerge-by=none → keeping every cell type as individual layer")

    # 4. Run
    merge_layers(
        wdir=Path(args.wdir),
        output_wdir=Path(args.output_wdir),
        allowed_stems=allowed_stems,
        consolidation_map=consolidation_map,
        extra_keep=args.extra_keep,
        remove_full_network=not args.keep_full_network,
    )


if __name__ == "__main__":
    main()