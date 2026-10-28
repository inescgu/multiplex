# multiplex

A heterogeneous multiplex network for **human dorsal root ganglion (DRG) neuron types**, used to build and characterise a **pain disease module**.

Each of the 22 neuron types from the 2025 human DRG atlas (HDRG, pseudobulk expression) becomes its own PPI layer. Every node is namespaced as `<cell_type>::GENE`, so `OPRM1` in `C-PEP.TAC1_CHRNA3` and in `A-PEP.KIT` are different nodes. The layers are joined by bipartite edges:

- ligand–receptor pairs (flat list, Reactome, and LIANA / NeuronChat scored interactions);
- identity edges to a `global_ppi` backbone layer.

The pipeline then:

1. runs a random walk with restart (**MultiXrank**) from 408 seed genes;
2. grows a connected **disease module** from the top-ranked genes;
3. analyses the module from several angles: layer topology, validation against nulls, two independent clustering methods (**HiDeF** and **spectral**), random-matrix bridge detection (**Marchenko–Pastur**), and **hub-removal** robustness;
4. compares the result with the earlier single-layer analysis in `../og_gene_set`.

The canonical version is the **pseudobulk** build: every path ends in `*_pseudobulk`. Superseded builds and code live in `archive/`, which is not tracked by git; tag `pre-audit-2026-10-21` has the last tracked copy.

---

## Quick start

All launchers are SLURM batch scripts that `cd` into this directory. Since 2026-10-21 they `set -eo pipefail`, so a Python crash fails the job and `afterok` dependents don't start.

```bash
cd /lisc/home/user/gerardursin/multiplex

# Part A: build the multiplex and the disease module (steps 1-7)
sub() { local dep=$1; shift; sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@"; }
J=$(sub ""  step2_buildcoex.sh)      # per-cell-type layers (runs BEFORE step1, see note below)
J=$(sub $J  step1.sh)
J=$(sub $J  step2.sh)
J=$(sub $J  step3.sh)
J=$(sub $J  step3b.sh)
J=$(sub $J  step5.sh)
J=$(sub $J  step4.sh)
J6=$(sub $J step6.sh)
for s in step6b step6c step6d step6e step7_top step7b step7c step7d step7dn; do sub $J6 $s.sh; done

# Part B: everything downstream of the module (steps 8-12). Checks Part A first.
PART_A="<first job id> <last job id>" bash submit_partB.sh
```

Run `submit_partB.sh` with **`bash`, not `sbatch`**: it runs on the login node and calls `sbatch` itself. If a Part A job that Part B doesn't read is still running (usually step 7b, ~17 h) or needs a separate rerun, leave it out of the check with `SKIP_IDS="<job id> ..."`.

After Part B, run these by hand:

1. `bash submit_step11i.sh pilot`, then `bash submit_step11i.sh full` (HiDeF cluster-level null).
2. The `phen/` enrichment (`sbatch phen/run.sh`), on the new HiDeF clusters.
3. `sbatch compare_ml_sl.sh`, then `sbatch plot_sl.sh`, then `sbatch step6f.sh` (multiplex vs single-layer).
4. `bash deck_figure_scripts/painfigs.sh` (figures for the slides).

> **Step numbers are not the run order at the start.** `step2_buildcoex.sh` (running `step2_build_25hdrg_multiplex.py`) builds the per-cell-type layers that `step1.sh` reads. The two "step 2" scripts are different steps.

> **For a clean rebuild, start step 1 from an empty `04_multixrank_prep/multixrank_hetero_input_pseudobulk/`.** Move the old folder aside first. `step3b` now refreshes stale `.tsv.orig` backups itself, but an empty folder removes any doubt.

---

## Pipeline

| Step | Launcher | Script | What it does | Main output |
|---|---|---|---|---|
| 0a | `step0a.sh` | `step0a_run_liana.R` | LIANA (natmi, connectome, logfc, sca, cellphonedb + consensus) on the HDRG Seurat object | `pathway_prep/lr_sources/liana_long.csv` |
| 0a′ | — | `filterliana.ipynb` | Keeps LIANA consensus interactions with score ≥ 0.995 | `pathway_prep/lr_sources/liana_consensus_top.csv` |
| 0b | `step0b.sh` | `step0b_run_neuronchat.R` | NeuronChat on the same object | `pathway_prep/lr_sources/neuronchat_long.csv` |
| 0c | `step2b_pathways.sh` | `pathway_prep/build_pathway_data.py` | Reactome / LR reference tables | `pathway_prep/*.csv` |
| — | `step0_compare.sh` | `step0_compare_multiplex_layers.py` | Compares the older filtered marker layers with the pseudobulk layers (one-off check) | `02_hdrg_multiplex/layer_comparison_pseudobulk/` |
| 2-layers | `step2_buildcoex.sh` | `step2_build_25hdrg_multiplex.py` | One PPI layer per cell type from expressed genes | `02_hdrg_multiplex/multiplex_2025hdrg_pseudobulk/layers/` |
| 1 | `step1.sh` | `step1_build_celltype_ppi.py` | Writes MultiXrank multiplex folders for the 22 neuron types (`neurons_only.csv`) + `global_ppi` backbone | `04_…/multixrank_hetero_input_pseudobulk/multiplex/` |
| 2 | `step2.sh` | `step2_build_bipartite_edges.py` | Bipartite edges between cell types: LR, Reactome, scored LIANA/NeuronChat, `global_ppi` identity edges | `…/bipartite/`, `bipartite_manifest.json`, `scored_lr_edge_provenance.csv` |
| 3 | `step3.sh` | `step3_build_config_seeds.py` | MultiXrank `config.yml` (r = 0.7), filters seeds, reference graphs | `config.yml`, `seeds/seeds.txt`, `gene_reference_graph.pkl` |
| 3b | `step3b.sh` | `step3b_namespace_nodes.py` | Renames nodes to `<cell_type>::GENE`, expands seeds per cell type (not into `global_ppi`) | namespaced `.tsv` + `.tsv.orig` backups |
| 5 | `step5.sh` | `step5_preflight_seed_reachability.py` | Checks that seeds are reachable before the walk | log only |
| 4 | `step4.sh` | `step4_run_multixrank.py` | MultiXrank; aggregates scores across cell types with `max` | `05_multixrank_results/multixrank_pseudo_results/` |
| 6 | `step6.sh` | `step6_build_disease_module.py` | Top-200 genes + greedy bridging into a connected module; elbow diagnostics | `06_disease_module/disease_module_output_pseudobulk/` (`disease_module.csv`, `module_graph.pkl`) |
| 6b | `step6b.sh` | `step6b_module_diagnostics.py` | Composition / LR / marker diagnostics | `module_exploratory_*` in the module folder |
| 6c–e | `step6c/6d/6e.sh` | `step6c/6d/6e_export_*.py` | Exports for DataDiVR (module, 1-/2-hop context) and wwiznet | `06_disease_module/datadivr/`, `06_disease_module/wwiznet_export/` |
| 7 | `step7_top.sh` | `step7_layer_topology_stats.py` | Per-layer topology metrics | `07_analysis/topology_output/` |
| 7b | `step7b.sh` | `step7b_layer_lcc_analysis.py` | Seed LCC significance per cell type | `07b_layer_lcc_analysis/` |
| 7c | `step7c.sh` | `step7c_interlayer_connectivity_analysis.py` | Bipartite connectivity by edge class | `07_analysis/interlayer_connectivity/` |
| 7d | `step7d.sh`, `step7dn.sh` | `step7d_seed_layer_profile.py` | Seed (7d) / module-neighbour (7dn) presence, degree, hubness, markers | `07d_seed_layer_profile/`, `07d_module_neighbors/` |
| 8 | `step8.sh`, `step8ctd.sh`, `step8f.sh` | `step8*_*.py` | Randomised spectral clustering of the module, plus diagnostics (8b–d), forced k (8e), nulls (8f), seed core/periphery (8g), figures (8h–i) | `08_spectral_analysis/` |
| 9 | `step9.sh`, `step9b.sh`, `step9v.sh` | `step9_multiplex_marchenko_pastur.py`, `step9b_…`, `step9viz_…` | Marchenko–Pastur bridge genes over an ω sweep, permutation test, slide visuals | `09_marchenko_pastur/` |
| 10 | `step10.sh` (`NULLS=seeds\|rewired\|neighbors`) | `step10_validate_module.py` (+ `size_adjust_nulls.py`) | Is the module better than chance? Nulls are rebuilt through the same MultiXrank + step 6 pipeline | `10_validation/` |
| 11a–b | `step11a.sh`, `step11preb.sh`, `step11b.sh` | `step11a_supragraph.py`, … | Supra-graph; resolution profile; Leiden CPM resolution sweep | `11_hidef_module/supra_graph.pkl`, `ensemble.pkl` |
| 11c | `step11c1.sh`, `step11c2.sh` | `step11c_phase1_consensus.py`, `step11c_phase2_weave.py` (+ `weaver_fast_patch.py`) | HiDeF windowed consensus + weave into a hierarchy | `$RUN_DIR/heterogeneous_hidef_faithful_cpm.{nodes,edges}` |
| 11d–k | `step11d…k.sh` | `step11d…k_*.py` | Infomap cross-check (d), characterise (e), visualise (f), presentation plots (g), persistence tiers (h), confounds (j), DAG nesting (k) | `$RUN_DIR/…` |
| 11i | `submit_step11i.sh` | `step11i_cluster_level_null_model.py`, `step11i_aggregate.py` | Cluster-level null (re-run MultiXrank + clustering on null seeds) | `results/11i_null/…` |
| 12a | `step12a.sh` | `step12a_main.py` (generated, see below) | Hub-removal attack curves (f_c, f_90), layer ablation | `12_hub_removal/hub_removal/` |
| 12b–c | `step12b_null.sh` (array), `step12c_aggregate.sh` | `step12b_null_replicate.py`, `step12c_aggregate_null.py` | Degree-preserving null replicates and their aggregation | `…/hub_removal/null/` |
| 12d | `step12d_tail.sh` | `step12d_tail.py` | Hub-bias permutation test; Enrichr on removed genes (needs internet) | `…/enrichment/` |
| 12e–f | `submit_step12e.sh` | `step12e_matched_replicate.py`, `step12f_aggregate_matched.py` | Observed vs naive / layer / layer+seed nulls under identical attack settings | `…/hub_removal/matched/` |
| 12g | `step12g.sh` | `step12g_removed_genes.py` | Which genes the attack removes up to f_c, and what they share | `…/hub_removal/removed_genes/` |
| 12-plots | `step12c.sh` | `step12c_plothubremoval.py` | Robustness plots | `12_hub_removal/plots/` |
| SL | `compare_ml_sl.sh`, `plot_sl.sh`, `step6f.sh` | `compare_multiplex_vs_singlelayer.py`, `plot_sl_module_decomposition.py`, `step6f_export_sl_decomposition_wwiznet.py` | Matches multiplex HiDeF clusters to single-layer HiDeF modules; decomposition figures / animation | `$RUN_DIR/compare_singlelayer/` |

`RUN_DIR` is set in `run_env.sh` (currently `11_hidef_module/results_cpm`).

### Runtime and memory

**Part A** was measured on the 2026-10-21 rebuild (jobs 7521343–7521359), checked against the Oct 19 run. Each job uses 1 CPU unless stated.

| Step | Wall time | Peak memory | Requested (time / mem) | Note |
|---|---|---|---|---|
| 2-layers `step2_buildcoex` | < 1 min | 0.9 GB | 12 h / 10 GB | |
| 1 | < 1 min | 1.4 GB | 2 h / 20 GB | |
| 2 bipartite | < 1 min | 0.4–2.0 GB | 2 h / 25 GB | |
| 3 | < 1 min | 0.7–2.9 GB | 2 h / 10 GB | |
| 3b | 1.5 min | 3.7 GB | 12 h / 25 GB | |
| 5 | 3 min | 3.9 GB | 12 h / 10 GB | |
| 4 MultiXrank | 6 min | 2.7 GB | 8 h / 10 GB | |
| 6 module | **1.1–1.5 h** | 4.0 GB | 10 h / **5 GB** | memory request is tight |
| 6b | < 1 min | 0.5 GB | 3 h / 35 GB | |
| 6c | 4 min | 0.2 GB | 4 h / 10 GB | |
| 6d | **3.6–3.7 h** | 4.9 GB | **4 h** / 10 GB | time limit is tight |
| 6e | < 1 min | 0.3 GB | 4 h / 10 GB | |
| 7 | 2 min | 1.1 GB | 3 h / 15 GB | |
| 7b | **~17.4 h** (4 CPUs) | 4.3 GB | 24 h / 20 GB | timed out once at a 13 h limit |
| 7c | < 1 min | 0.4 GB | 3 h / 30 GB | |
| 7d, 7dn | < 1 min | 0.5 GB | 3 h / 10 GB | |

Part A takes about **5.5 h** start to finish (step 6, then 6d), or about **18 h** if you wait for 7b.

Not measured here:
- **0a LIANA:** requested 18 h / 20 GB.
- **0b NeuronChat:** requested 4 h / 40 GB.
- **External `pseudo.sh`:** 1 min to 2.7 h depending on the script, up to 12.7 GB; it ran out of memory once at 10 GB.

**Part B has no clean measurements yet.** Logs from the earlier runs weren't kept, and SLURM accounting only goes back to mid-October. Until the 2026-10-22 run finishes, the launcher requests below are the only estimate. A few jobs from the 2026-10-21 run on the previous module did finish and give a rough idea:
- `step9_mp`: 3 min, 0.4 GB.
- Each `step12e` replicate: ~15 s, 0.2–0.3 GB.

| Step | Requested (time / mem / CPUs) |
|---|---|
| 8 spectral (`step8.sh`) | 24 h / 10 GB / 10 |
| 8 diagnostics (`step8ctd.sh`, `step8f.sh`) | 5 h / 10 GB / 1; 4 h / 10 GB / 2 |
| 9, 9b, 9v | 16 h / 20, 30, 10 GB / 1 |
| 10 validation, per null | 48 h / 64 GB / 24 (`seeds`, `rewired` submitted with 25 CPUs; `neighbors` with 6 CPUs and 2 h) |
| 11a / 11preb / 11b | 4 h / 32 GB; 2 h / 5 GB; 4 h / 10 GB |
| 11c1 consensus / 11c2 weave | 4 h / 55 GB; 6 h / 80 GB / 3 |
| 11d–k | 4–8 h / 8–10 GB / 1 |
| 11i (per array task) | 6 h / 32 GB |
| 12a main | 24 h / 10 GB / 4 |
| 12b null replicates (50, 10 at a time) | 4 h / 6 GB each |
| 12e matched replicates (4 variants × 10–50) | 4 h / 6 GB each |
| 12g removed genes | 24 h / 6 GB |
| 12c, 12f aggregate; 12c plots | ≤ 30 min / 4 GB |
| comparison `compare_ml_sl` / `plot_sl` / `step6f` | 2 h / 1 h / 1 h, 16 GB |
| `phen/run.sh` | 4 h / 8 GB |

*TODO: after the 2026-10-22 run, fill in measured values with* `sacct -j <ids> -o JobName,Elapsed,MaxRSS,AllocCPUS`.

### Why HiDeF and spectral clustering both stay

They answer different questions. HiDeF gives a persistence-ranked hierarchy of communities. Spectral clustering gives a flat partition with its own null models. In the single-layer project the two complemented each other, so both are kept here for a like-for-like comparison.

### Step 12: the generated `step12a_main.py`

`step12_hub_removal_analysis.py` is the full analysis. Steps 12b, 12c, 12d and 12g import helpers from it, so **don't move or rename it**. `step12a_make_main.py` patches it into `step12a_main.py`, adding `--skip-null`, `--skip-ablation` and `--skip-permutation`, so the expensive stages can run as separate array jobs. After editing `step12_hub_removal_analysis.py`, regenerate:

```bash
python step12a_make_main.py      # must report "7 patch groups applied cleanly"
```

`step12g` must use the same `--cascading-step` / `--betweenness-k` as `step12a`, otherwise it reconstructs a different removal set.

---

## Layout

```
00_inputs/                 seed genes, PPI, HDRG pseudobulk expression & markers, GO files
pathway_prep/              LR pairs, Reactome tables, LIANA/NeuronChat outputs (lr_sources/)
02_hdrg_multiplex/         per-cell-type layers from step2_build_25hdrg_multiplex
04_multixrank_prep/        MultiXrank working dir (not tracked; regenerate with steps 1-3b)
05_multixrank_results/     random-walk scores
06_disease_module/         module + exports
07_analysis/ 07b_… 07d_…   layer analyses
08_spectral_analysis/      spectral clustering
09_marchenko_pastur/       MP bridges
10_validation/             module nulls
11_hidef_module/           HiDeF supra-graph, ensemble, $RUN_DIR results
12_hub_removal/            hub-removal robustness
results/                   11i nulls, 11c sweeps, RESULTS_INDEX.csv
phen/                      enrichment of HiDeF clusters (proteomics, phenotypes, neuropathic pain, spatial, OMIM)
deck_figure_scripts/       slide figures from the outputs above (painfigs.sh)
archive/                   superseded code and pre-pseudobulk results (not tracked)
slurm_logs/                job logs (not tracked)
```

---

## Inputs

| File | Contents |
|---|---|
| `00_inputs/seed_genes.csv` | 408 seed genes (`source`: `zhao` 312, `hpg_zhao` 96) |
| `00_inputs/ppi_network.pkl` | Human PPI used for every layer and the `global_ppi` backbone |
| `00_inputs/hdrg_expressed_pseudobulk.csv` | Per-gene, per-cell-type pseudobulk expression (`frac_blocks_expressed`, `mean_cpm`, markers) |
| `00_inputs/hdrg_pseudobulk_markers_all.csv` | Marker statistics used by steps 6b and 7d |
| `neurons_only.csv` / `layers_categories.csv` | Which cell types become layers / category labels |
| `pathway_prep/lr_pairs_with_pathways.csv` | Flat ligand–receptor list with pathways |
| `pathway_prep/reactome_interactions_with_pathways_v2.csv` | Reactome interactions |

**Dependencies outside this repo:**
- `../R/convert_hdrg_human_2025/seurat_hdrg/seurat_object_preprint_release.Rds`: HDRG Seurat object (steps 0a, 0b).
- `../og_gene_set/`: single-layer project. It supplies the background network for steps 12/12d (`temp/combined26/full_network.pkl`) and the single-layer HiDeF communities, module graph and enrichment for the comparison scripts.
- The pseudobulk inputs come from `../R/convert_hdrg_human_2025/` (run with `pseudo.sh`). The copies in `00_inputs/` are byte-identical to these files:
  - `pseudobulk_expressed_layers.R` → `cell_type_expressed_genes.csv` = `00_inputs/hdrg_expressed_pseudobulk.csv`
  - `pseudobulk_markers.R` → `pseudobulk_markers_all.csv` = `00_inputs/hdrg_pseudobulk_markers_all.csv`
  - `pseudobulk_markers.R` → `cell_type_markers_pseudobulk.csv` = `00_inputs/hdrg_markers_pseudobulk.csv`

---

## Environments

Launchers load a cluster Python module and a venv under `/lisc/data/scratch/menche/ines/`:

| venv | Python module | Used by |
|---|---|---|
| `multixrank_env` | `Python/3.12.3-GCCcore-13.3.0` | steps 1–7 (except 7b), 8, 9, 10 |
| `hidef_env` | `Python/3.10.4-GCCcore-11.3.0` | steps 6c–f, 11*, `phen/`, single-layer comparison |
| `general_env` | `Python/3.12.3-GCCcore-13.3.0` (the venv's own base; launchers used to load `-weekly` builds, which no longer exist) | steps 7b, 12* |
| `multi_env` | `Python/3.13.5-GCCcore-14.3.0` | `step2_buildcoex.sh` |
| R module | — | steps 0a, 0b (`install_packages.R`, `installR.sh`) |

Don't use the cluster's `-weekly` Python modules in launchers: they get removed, and the job then dies at `module load`. Load the Python the venv was built on (`cat <venv>/pyvenv.cfg`). `phen/phen.sh`, `np.sh`, `prot.sh` and `st.sh` (single-layer jobs) still load a `-weekly` build. *TODO: `pip freeze > envs/<name>.txt` for each venv.*

---

## Conventions

- **Node names.** `<cell_type>::GENE` everywhere after step 3b (separator `SEP = "::"`). Step 4's aggregated `gene_rwr_scores.csv` is the exception: it uses bare gene symbols (max across cell types).
- **`.tsv.orig` files** in `04_…/multiplex/*/` and `04_…/bipartite/` are step 3b's pre-namespacing backups. Steps 6b and 7d read them. Don't delete them.
- **Provenance.** HiDeF-era scripts call `provenance.write_manifest()`, which writes `manifest_<script>_<timestamp>.json` into their output folder: arguments, git commit, host, SLURM job. These manifests are tracked despite the `*.json` ignore rule. Record a result you decide to keep with:
  ```bash
  python keep.py <step> <output_dir> "<key params>" "<decision>"   # -> results/RESULTS_INDEX.csv
  ```
- **`run_env.sh`** (sourced by the step-11 launchers and the comparison launchers):
  - `RUN_DIR` and the HiDeF parameters (`NEIGHBORS=2 CHI=5 CONSENSUS_P=75 JACCARD=0.75 MINSIZE=4`);
  - `snapshot <dir>` copies the submitted launcher into the output folder;
  - `guard <path>` refuses to overwrite an existing result.
- **Not tracked by git:** `*.pkl`, `*.csv`, `*.json`, `*.npy`, `04_multixrank_prep/`, `archive/`, `slurm_logs/`. Most outputs exist only on disk; regenerate them with the pipeline rather than relying on git.

---

## Known issues / TODO

- `deck_figure_scripts/painfigs.sh` calls `fig_step12g_removed.py`, but the file is `fig_step12_removed.py`.
- Environments aren't pinned (see above).
- `step6.sh` requests 5 GB but uses 4.0 GB, and `step6d.sh` requests 4 h but takes 3.7 h. Raise both before a larger module pushes them over.
- The Oct 19 pseudobulk module (archived at `archive/results_pre_pseudobulk/04/…_oct19`) was built from the unfiltered `liana_long.csv`, because of a stale-backup bug in step 3b (fixed 2026-10-21). Results from the 2026-10-21 rerun onwards use `liana_consensus_top.csv`.
