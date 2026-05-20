# Multiplex Stochastic Block Model (SBM) for Community Detection

A scalable, production-ready implementation of multiplex SBM with variational inference for community detection on integrated multiplex interactome networks.

## Overview

This framework fits a **multiplex stochastic block model** where:
- All edge types (`ppi`, `pathway`, `ligand_receptor`) are treated equally as separate layers
- Each layer has its own block probability matrix **B^l**, but all layers share a single community assignment vector **z**
- Variational inference (mean-field EM) provides scalable approximate posterior inference
- Optional node metadata (e.g. cell type) can be incorporated as a prior on community assignments

## File Structure

| File | Description |
|------|-------------|
| `sbm_core.py` | Core SBM: data loading, variational EM, E-step/M-step, ELBO computation |
| `model_selection.py` | Automatic K selection with BIC/AIC/ICL; parallel sweep; saving utilities |
| `sbm_visualization.py` | All plots: heatmaps, block matrices, sizes, model comparison, network layout |
| `run_multiplex_sbm.py` | CLI driver script integrating all components |
| `test_sbm.py` | End-to-end tests with synthetic planted-partition networks |

## Quick Start

### Model selection (recommended)
```bash
python run_multiplex_sbm.py \
    --graph /path/to/integrated_multiplex.pkl \
    --registry /path/to/node_registry.csv \
    --output-dir sbm_results \
    --k-min 2 --k-max 20 \
    --n-cores 4
```

### Fixed K with metadata
```bash
python run_multiplex_sbm.py \
    --graph integrated_multiplex.pkl \
    --registry node_registry.csv \
    --output-dir sbm_results \
    --fixed-k 10 \
    --use-metadata --metadata-weight 1.0
```

### Full sweep, 10 cores, all features
```bash
python run_multiplex_sbm.py \
    --graph integrated_multiplex.pkl \
    --registry node_registry.csv \
    --output-dir sbm_results \
    --k-min 2 --k-max 50 \
    --n-cores 10 \
    --use-metadata \
    --max-iter 300 --tol 1e-7 \
    --n-restarts 10
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--graph` | required | Path to `integrated_multiplex.pkl` |
| `--registry` | required | Path to `node_registry.csv` |
| `--output-dir` | `sbm_results` | Output directory |
| `--fixed-k` | None | Fixed K (skip model selection) |
| `--k-min` / `--k-max` | 2 / 20 | K range for model selection |
| `--k-step` | 1 | Step size for K sweep |
| `--criterion` | `ICL` | Selection criterion: BIC, AIC, or ICL |
| `--use-metadata` | False | Incorporate cell_type as prior |
| `--metadata-col` | `cell_type` | Metadata column name |
| `--metadata-weight` | 1.0 | Weight (η) for metadata prior |
| `--max-iter` | 200 | Max EM iterations per fit |
| `--tol` | 1e-6 | Convergence threshold |
| `--n-restarts` | 5 | Random restarts per K |
| `--chunk-size` | 5000 | E-step chunk size (memory control) |
| `--n-cores` | 1 | CPU cores for parallel K sweep (max 10) |
| `--no-viz` | False | Skip visualisation generation |
| `--edge-types` | all three | Edge types to include as layers |

## Outputs

| File | Description |
|------|-------------|
| `community_assignments.csv` | Node-level: node_id, gene, source, cell_type, community, posterior probs |
| `model_parameters.json` | π, B matrices, convergence info, ELBO history |
| `model_comparison.csv` | BIC/AIC/ICL for each K tested |
| `tau_posteriors.npz` | Full posterior matrix (compressed) |
| `run_summary.json` | Summary of the run |
| `run_config.json` | Exact arguments used |
| `sbm_run.log` | Full log |
| `*.png` | Visualisations (heatmap, block matrices, sizes, network layout, etc.) |

## Algorithm Details

**Variational EM** with mean-field approximation:
1. **E-step**: Update q(z_i) for each node given current B^l and π
2. **M-step**: Update π and B^l given current q(z)
3. **Convergence**: Monitor ELBO; stop when relative change < tolerance
4. **Restarts**: Multiple random initialisations; keep best ELBO

**Model selection** compares K values using:
- **BIC** = -2·LL + p·log(n) — penalises model complexity
- **AIC** = -2·LL + 2·p — less conservative than BIC
- **ICL** = BIC + entropy(τ) — additionally penalises fuzzy assignments (recommended)

**Memory efficiency**: Sparse adjacency matrices (scipy.sparse CSR), chunked E-step updates.

## Dependencies

```
numpy scipy networkx pandas matplotlib seaborn scikit-learn tqdm
```

## References

1. Multiplex SBM with hierarchical structure — [Oxford Academic](https://academic.oup.com/comnet/article/14/1/cnaf049/8441056)
2. SBM foundations — Holland et al. (1983)
3. SBM with unknown K + metadata — [ICML 2012](https://icml.cc/2012/papers/771.pdf)
