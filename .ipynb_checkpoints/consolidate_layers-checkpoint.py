#!/usr/bin/env python3
"""
Consolidate redundant cell-type layers for MultiXrank.
Merges biologically related subtypes into representative layers.
Reduces layer count from 177 → ~40-50 without losing gene coverage.
"""
import pandas as pd
from pathlib import Path
import shutil
import yaml
import json

# Define consolidation groups
# Keys = new merged layer name
# Values = list of existing layer prefixes to merge
CONSOLIDATION_MAP = {
    "T_cell_merged": [
        "T_cell_layer",
        "CD4-positive,_alpha-beta_T_cell_layer",
        "CD8-positive,_alpha-beta_T_cell_layer",
        "CD4-positive,_alpha-beta_thymocyte_layer",
        "CD8-positive,_alpha-beta_thymocyte_layer",
        "activated_CD4-positive,_alpha-beta_T_cell_layer",
        "activated_CD8-positive,_alpha-beta_T_cell_layer",
        "naive_thymus-derived_CD4-positive,_alpha-beta_T_cell_layer",
        "naive_thymus-derived_CD8-positive,_alpha-beta_T_cell_layer",
        "regulatory_T_cell_layer",
        "gamma-delta_T_cell_layer",
        "mature_NK_T_cell_layer",
        "thymocyte_layer",
    ],
    "NK_cell_merged": [
        "natural_killer_cell_layer",
        "innate_lymphoid_cell_layer",
    ],
    "endothelial_merged": [
        "endothelial_cell_layer",
        "capillary_endothelial_cell_layer",
     #   "cardiac_endothelial_cell_layer",
        "colon_endothelial_cell_layer",
        "endothelial_cell_of_arteriole_layer",
        "endothelial_cell_of_artery_layer",
        "endothelial_cell_of_lymphatic_vessel_layer",
        "endothelial_cell_of_venule_layer",
        "endothelial_cell_of_vascular_tree_layer",
        "retinal_blood_vessel_endothelial_cell_layer",
        "vein_endothelial_cell_layer",
    ],
    "fibroblast_stromal_merged": [
        "fibroblast_layer",
        "fibroblast_of_breast_layer",
        "fibroblast_of_cardiac_tissue_layer",
        "stromal_cell_layer",
        "stromal_cell_of_ovary_layer",
        "hepatic_stellate_cell_layer",
        "adventitial_cell_layer",
        "alveolar_adventitial_fibroblast_layer",
        "connective_tissue_cell_layer",
        "mesenchymal_stem_cell_layer",
        "mesenchymal_stem_cell_of_adipose_tissue_layer",
        "mural_cell_layer",
        "pericyte_layer",
        "myofibroblast_cell_layer",
        "tendon_cell_layer",
    ],
    "enterocyte_merged": [
        "enterocyte_of_epithelium_of_large_intestine_layer",
        "enterocyte_of_epithelium_proper_of_duodenum_layer",
        "enterocyte_of_epithelium_proper_of_ileum_layer",
        "enterocyte_of_epithelium_proper_of_jejunum_layer",
        "enterocyte_of_epithelium_proper_of_small_intestine_layer",
        "BEST4+_enterocyte_layer",
    ],
    "macrophage_monocyte_merged": [
        "macrophage_layer",
        "classical_monocyte_layer",
        "non-classical_monocyte_layer",
        "monocyte_layer",
        "colon_macrophage_layer",
        "tissue-resident_macrophage_layer",
        "mononuclear_phagocyte_layer",
        "myeloid_leukocyte_layer",
        "intermediate_monocyte_layer",
    ],
    "dendritic_cell_merged": [
        "myeloid_dendritic_cell_layer",
        "plasmacytoid_dendritic_cell_layer",
        "Langerhans_cell_layer",
    ],
    "smooth_muscle_merged": [
        "smooth_muscle_cell_layer",
        "blood_vessel_smooth_muscle_cell_layer",
        "bronchial_smooth_muscle_cell_layer",
        "vascular_associated_smooth_muscle_cell_layer",
        "myometrial_cell_layer",
    ],
    "epithelial_merged": [
        "epithelial_cell_layer",
        "basal_cell_layer",
        "basal_cell_of_prostate_epithelium_layer",
        "bladder_urothelial_cell_layer",
        "ciliated_epithelial_cell_layer",
       # "club_cell_layer",
        "conjunctival_epithelial_cell_layer",
        "corneal_epithelial_cell_layer",
        "epithelial_cell_of_uterus_layer",
        "glandular_secretory_epithelial_cell_layer",
        "stratified_squamous_epithelial_cell_layer",
        "ovarian_surface_epithelial_cell_layer",
    ],
    "goblet_secretory_merged": [
        "large_intestine_goblet_cell_layer",
        "small_intestine_goblet_cell_layer",
        "respiratory_tract_goblet_cell_layer",
        "tracheal_goblet_cell_layer",
        "mucus_secreting_cell_layer",
        "club_cell_layer",
        "serous_cell_of_epithelium_of_trachea_layer",
    ],
    "hematopoietic_progenitor_merged": [
        "hematopoietic_cell_layer",
        "hematopoietic_precursor_cell_layer",
        "hematopoietic_stem_cell_layer",
        "common_myeloid_progenitor_layer",
        "erythroid_progenitor_cell_layer",
    ],
    "myeloid_merged": [
        "myeloid_cell_layer",
        "granulocyte_layer",
        "neutrophil_layer",
        "basophil_layer",
        "mast_cell_layer",
        "leukocyte_layer",
    ],
    "cardiac_merged": [
        "regular_atrial_cardiac_myocyte_layer",
        "ventricular_cardiac_muscle_cell_layer",
        "cardiac_endothelial_cell_layer",
    ],
    "skeletal_muscle_merged": [
        "fast_muscle_cell_layer",
        "slow_muscle_cell_layer",
        "muscle_cell_layer",
        "skeletal_muscle_satellite_stem_cell_layer",
        "tongue_muscle_cell_layer",
    ],
    "acinar_merged": [
        "acinar_cell_layer",
        "acinar_cell_of_salivary_gland_layer",
        "pancreatic_acinar_cell_layer",
        "salivary_gland_cell_layer",
    ],
    "paneth_intestinal_crypt_merged": [
        "paneth_cell_of_colon_layer",
        "paneth_cell_of_epithelium_of_small_intestine_layer",
        "intestinal_crypt_stem_cell_of_colon_layer",
        "intestinal_crypt_stem_cell_of_small_intestine_layer",
        "transit_amplifying_cell_of_colon_layer",
        "enteroendocrine_cell_of_small_intestine_layer",
        "intestinal_tuft_cell_layer",
        "tuft_cell_of_colon_layer",
        "enteroglial_cell_layer",
    ],
    "lung_epithelial_merged": [
        "lung_multiciliated_epithelial_cell_layer",
        "multiciliated_columnar_cell_of_tracheobronchial_tree_layer",
        "pulmonary_alveolar_type_1_cell_layer",
        "pulmonary_alveolar_type_2_cell_layer",
    ],
    "retinal_merged": [
        "eye_photoreceptor_cell_layer",
        "retinal_bipolar_neuron_layer",
        "retinal_pigment_epithelial_cell_layer",
        "Mueller_cell_layer",
    ],
    "sperm_merged": [
        "male_germ_cell_layer",
        "spermatid_layer",
        "spermatocyte_layer",
        "spermatogonium_layer",
    ],
    "thymic_merged": [
        "myo-medullary_thymic_epithelial_cell_layer",
        "neuro-medullary_thymic_epithelial_cell_layer",
        "thymic_fibroblast_type_1_layer",
        "thymic_fibroblast_type_2_layer",
    ],
}

# These stay as individual layers (biologically distinct / disease-relevant)
KEEP_INDIVIDUAL = [
    "ppi_layer",               # contains SSTR4, TLR9 (global PPI backbone)
    "ligand_receptor_layer",    # ADD THIS - inter-layer LR edges
    "pathway_layer",            # ADD THIS - inter-layer pathway edges
    "duct_epithelial_cell_layer",  # ADD THIS - currently orphaned
    "B_cell_layer",
    "hepatocyte_layer",
    "neuron_layer",
    "microglial_cell_layer",
    "Schwann_cell_layer",
    "melanocyte_layer",
    "plasma_cell_layer",
    "erythrocyte_layer",
    "platelet_layer",
    "pancreatic_ductal_cell_layer",
    "type_B_pancreatic_cell_layer",
    "pancreatic_stellate_cell_layer",
    "kidney_epithelial_cell_layer",
    "ionocyte_layer",
    "keratocyte_layer",
    "sebocyte_layer",
    "myoepithelial_cell_layer",
    "mesothelial_cell_layer",
    "intrahepatic_cholangiocyte_layer",
    "interstitial_cell_of_Cajal_layer",
    "peritubular_myoid_cell_layer",
    "theca_cell_layer",
    "supporting_cell_of_vestibular_epithelium_layer",
    "luminal_cell_of_prostate_epithelium_layer",
    "luminal_epithelial_cell_of_mammary_gland_layer",
    "progenitor_cell_of_mammary_luminal_epithelium_layer",
    "ATF3_layer",
    # Sensory neuron subtypes - keep individual (likely disease-relevant)
    "A-LTMR.TAC3_layer",
    "A-PEP.CHRNA7_SLC18A3_layer",
    "A-PEP.KIT_layer",
    "A-PEP.NTRK3_S100A16_layer",
    "A-PEP.SCGN_ADRA2C_layer",
    "A-Propr.EPHA3_layer",
    "A-Propr.HAPLN4_layer",
    "A-Propr.PCDH8_layer",
    "Ab-LTMR.CCKAR_layer",
    "Ab-LTMR.ETV1_layer",
    "Ab-LTMR.LGI2_layer",
    "Ab-LTMR.NSG2_layer",
    "C-LTMR.CDH9_layer",
    "C-NP.MRGPRX1_GFRA2_layer",
    "C-NP.MRGPRX1_MRGPRX4_layer",
    "C-NP.SST_layer",
    "C-PEP.ADORA2B_layer",
    "C-PEP.TAC1_CACNG5_layer",
    "C-PEP.TAC1_CHRNA3_layer",
    "C-Thermo.RXFP1_layer",
    "C-Thermo.TRPM8_layer",
]

def check_seed_coverage(
    output_wdir: Path,
    new_layer_paths: list,
) -> None:
    """Verify all seeds appear in at least one consolidated layer."""
    seeds_path = output_wdir / "seeds" / "seeds.txt"
    if not seeds_path.exists():
        return
    
    seeds = {s.strip().upper() 
             for s in seeds_path.read_text().splitlines() 
             if s.strip()}
    
    covered = set()
    for rel_path in new_layer_paths:
        abs_path = output_wdir / rel_path
        if not abs_path.exists():
            continue
        df = pd.read_csv(abs_path, sep="\t", header=None,
                        names=["g1", "g2"], dtype=str)
        genes = (set(df["g1"].str.strip().str.upper()) | 
                set(df["g2"].str.strip().str.upper()))
        covered.update(seeds & genes)
    
    missing = seeds - covered
    if missing:
        print(f"\n⚠ WARNING: These seeds are NOT covered by any layer: {missing}")
        print("  → They will cause MultiXrank errors.")
        print("  → Fix: ensure the original layers containing these genes")
        print("    are included in CONSOLIDATION_MAP or KEEP_INDIVIDUAL")
    else:
        print(f"\n✓ All {len(seeds)} seeds covered by consolidated layers")

def rebuild_layer_info(new_layer_paths: list, output_wdir: Path) -> None:
    """Rebuild layer_info.json to match consolidated layer names."""
    layer_info = {}
    
    for rel_path in new_layer_paths:
        stem = Path(rel_path).stem  # e.g. "T_cell_merged" or "ligand_receptor_layer"
        
        # Classify by name pattern
        if stem in ("ligand_receptor_layer", "pathway_layer", "ppi_layer"):
            # These are global/special layers - treat as intra for scoring
            # or explicitly mark as special
            layer_info[stem] = {
                "type": "special",
                "cell_types": [],
                "edge_type": stem.replace("_layer", "")
            }
        elif stem.endswith("_merged"):
            # Consolidated cell type group
            layer_info[stem] = {
                "type": "intra",
                "cell_types": [stem],   # use merged name as cell type label
                "edge_type": "ppi"
            }
        else:
            # Individual kept layer - strip "_layer" suffix for cell type name
            cell_type = stem.replace("_layer", "")
            layer_info[stem] = {
                "type": "intra",
                "cell_types": [cell_type],
                "edge_type": "ppi"
            }
    
    out_path = output_wdir / "layer_info.json"
    with open(out_path, "w") as f:
        json.dump(layer_info, f, indent=2)
    print(f"Rebuilt layer_info.json → {out_path}")
    print(f"  {sum(1 for v in layer_info.values() if v['type']=='intra')} intra layers")
    print(f"  {sum(1 for v in layer_info.values() if v['type']=='special')} special layers")
    
def merge_layers(
    wdir: Path,
    output_wdir: Path,
    consolidation_map: dict,
    keep_individual: list,
    remove_full_network: bool = True,
) -> None:
    """
    Read existing layer TSVs, merge according to consolidation_map,
    write new TSVs, and update config.yml.
    
    Merging strategy: UNION of all edges across merged layers.
    This preserves ALL edges - no information is lost, only redundant
    layer structure is removed.
    """
    multiplex_dir = wdir / "multiplex" / "gene_multiplex"
    out_multiplex_dir = output_wdir / "multiplex" / "gene_multiplex"
    out_multiplex_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy seeds and other files
    (output_wdir / "seeds").mkdir(exist_ok=True)
    shutil.copy(wdir / "seeds" / "seeds.txt", 
                output_wdir / "seeds" / "seeds.txt")
    for f in ["layer_info.json", "cell_type_metadata.json",
              "gene_reference_graph.pkl", "gene_ppi_reference_graph.pkl"]:
        if (wdir / f).exists():
            shutil.copy(wdir / f, output_wdir / f)
    
    # Load existing config
    with open(wdir / "config.yml") as f:
        cfg = yaml.safe_load(f)
    
    existing_layers = cfg["multiplex"]["gene_multiplex"]["layers"]
    
    # Build filename -> full path map
    layer_file_map = {}
    for layer_path in existing_layers:
        fname = Path(layer_path).name
        stem = fname.replace(".tsv", "")
        layer_file_map[stem] = wdir / layer_path
    
    print(f"Input layers: {len(existing_layers)}")
    
        # After building layer_file_map, add:
    print(f"\nAvailable layer stems (first 10):")
    for stem in list(layer_file_map.keys())[:10]:
        print(f"  {repr(stem)}")
    
    # Check specifically for seeds' home layers:
    target_genes = {"SSTR4", "TLR9"}
    print(f"\nSearching for layers containing {target_genes}...")
    for stem, path in layer_file_map.items():
        if path.exists():
            df = pd.read_csv(path, sep="\t", header=None,
                           names=["g1","g2"], dtype=str)
            genes = set(df["g1"].str.upper()) | set(df["g2"].str.upper())
            if target_genes & genes:
                found = target_genes & genes
                print(f"  {stem}: contains {found}")
                
    # ---------------------------------------------------------------
    # Pass 1: determine which input layers are handled
    # ---------------------------------------------------------------
    handled_stems = set()
    for stems in consolidation_map.values():
        handled_stems.update(stems)
    
    new_layer_paths = []
    
    # ---------------------------------------------------------------
    # Pass 2: write merged layers
    # ---------------------------------------------------------------
    print("\nMerging layers...")
    for merged_name, source_stems in consolidation_map.items():
        all_edges: set[tuple[str, str]] = set()
        found_sources = []
        
        for stem in source_stems:
            path = layer_file_map.get(stem)
            if path and path.exists():
                df = pd.read_csv(path, sep="\t", header=None,
                                names=["g1", "g2"], dtype=str)
                for _, row in df.iterrows():
                    a, b = sorted([row["g1"].strip(), row["g2"].strip()])
                    all_edges.add((a, b))
                found_sources.append(stem)
        
        if not all_edges:
            print(f"  ⚠ {merged_name}: no edges found, skipping")
            continue
        
        rel_path = Path("multiplex") / "gene_multiplex" / f"{merged_name}.tsv"
        abs_path = output_wdir / rel_path
        
        df_out = pd.DataFrame(sorted(all_edges), columns=["g1", "g2"])
        df_out.to_csv(abs_path, sep="\t", index=False, header=False)
        new_layer_paths.append(str(rel_path))
        
        print(f"  {merged_name}: {len(found_sources)} sources "
              f"→ {len(all_edges):,} unique edges")
    
    # ---------------------------------------------------------------
    # Pass 3 (FIXED): copy layers correctly
    # ---------------------------------------------------------------

    # Build full set of stems that consolidation handled
    handled_stems = set()
    for stems in consolidation_map.values():
        handled_stems.update(stems)

    print("\nCopying individual layers...")

    for stem, path in layer_file_map.items():
        # Skip layers absorbed into merged groups
        if stem in handled_stems:
            continue

        # Skip full_network unless explicitly kept
        if remove_full_network and stem == "full_network_layer":
            print(f"  Removing: {stem} (full_network)")
            continue

        # Copy everything else (includes ppi_layer, ligand_receptor_layer,
        # pathway_layer, and all KEEP_INDIVIDUAL entries)
        rel_path = Path("multiplex") / "gene_multiplex" / f"{stem}.tsv"
        abs_path = output_wdir / rel_path
        shutil.copy(path, abs_path)
        new_layer_paths.append(str(rel_path))

        # Flag if this was explicitly listed as keep_individual
        tag = " ← KEEP_INDIVIDUAL" if stem in keep_individual else ""
        print(f"  Kept: {stem}{tag}")

    # Warn about KEEP_INDIVIDUAL entries that weren't found at all
    keep_stems = set(keep_individual)
    found_stems = set(layer_file_map.keys())
    missing_keeps = keep_stems - found_stems - handled_stems
    if missing_keeps:
        print(f"\n  WARNING: These KEEP_INDIVIDUAL stems not found in input:")
        for s in sorted(missing_keeps):
            print(f"    {s}")
    
    # CRITICAL: warn about anything that was silently dropped
    # (your original code had no fallback for unmatched layers)
    
    # ---------------------------------------------------------------
    # Pass 4: write new config
    # ---------------------------------------------------------------
    n_new = len(new_layer_paths)
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
    
    # After Pass 4 (writing config):
    rebuild_layer_info(new_layer_paths, output_wdir)
    check_seed_coverage(output_wdir, new_layer_paths)
    
    print(f"\n{'='*60}")
    print(f"Consolidation complete")
    print(f"  Input layers  : {len(existing_layers)}")
    print(f"  Output layers : {n_new}")
    print(f"  Reduction     : {len(existing_layers) - n_new} layers removed")
    print(f"  New config    → {output_wdir / 'config.yml'}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Consolidate redundant MultiXrank layers"
    )
    parser.add_argument("--wdir", required=True,
                        help="Original MultiXrank working directory")
    parser.add_argument("--output-wdir", required=True,
                        help="New working directory with consolidated layers")
    parser.add_argument("--keep-full-network", action="store_true",
                        help="Keep the full_network layer (not recommended)")
    args = parser.parse_args()
    
    merge_layers(
        wdir=Path(args.wdir),
        output_wdir=Path(args.output_wdir),
        consolidation_map=CONSOLIDATION_MAP,
        keep_individual=KEEP_INDIVIDUAL,
        remove_full_network=not args.keep_full_network,
    )


if __name__ == "__main__":
    main()