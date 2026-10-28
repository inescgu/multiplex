#!/usr/bin/env python3
"""
step0c: Joint UMAP comparing cell-type clusters between
  (A) the HDRG human DRG Seurat atlas, and
  (B) Tabula Sapiens (CELLxGENE Census, same version as step1).

Lives next to step1_ts_api.py. Example:

    python step0c_umap_compare.py \
        --seurat-rds ../R/convert_hdrg_human_2025/seurat_hdrg/seurat_object_preprint_release.Rds \
        --census-version "2025-11-08" \
        --output-dir 01_ts_multiplex/step0c_umap

How it works
  1. Calls Rscript to export a per-cell-type subsample of the Seurat RNA *counts*
     (Matrix Market + gene list + metadata). Python can't read .Rds natively.
  2. Pulls a per-cell-type subsample of Tabula Sapiens raw counts from Census.
  3. Intersects genes, normalises both the same way, picks HVGs, runs PCA,
     corrects the source effect with Harmony, then UMAP.
  4. Saves UMAPs (by source, each atlas highlighted, labelled) plus a
     cell-type-to-cell-type similarity heatmap and best-match table.

Dependencies (in addition to step1's):
    pip install scanpy harmonypy anndata matplotlib
    R with Seurat + Matrix available as `Rscript` (e.g. `module load R`)
"""

import os
import sys
import argparse
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellxgene_census

warnings.filterwarnings("ignore")

TABULA_SAPIENS_COLLECTION = "Tabula Sapiens"

R_EXPORT = r"""
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; label_col <- args[2]; max_per <- as.integer(args[3]); out <- args[4]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

obj <- readRDS(rds)
DefaultAssay(obj) <- "RNA"
try(obj <- JoinLayers(obj), silent = TRUE)   # Seurat v5 split layers; no-op otherwise
counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
md <- obj@meta.data
stopifnot(label_col %in% colnames(md))

set.seed(0)
idx <- unlist(lapply(split(seq_len(nrow(md)), md[[label_col]]), function(i) {
  if (length(i) > max_per) sample(i, max_per) else i
}))
counts <- counts[, idx]

writeMM(counts, file.path(out, "counts.mtx"))
writeLines(rownames(counts), file.path(out, "genes.txt"))
write.csv(data.frame(cell = colnames(counts), label = as.character(md[[label_col]][idx])),
          file.path(out, "meta.csv"), row.names = FALSE)
cat("Exported", ncol(counts), "cells x", nrow(counts), "genes\n")
"""


# ----------------------------------------------------------------------------
# Load HDRG (Seurat) via R export
# ----------------------------------------------------------------------------
def load_hdrg(rds_path, label_col, max_per_type, workdir):
    if not os.path.exists(rds_path):
        raise FileNotFoundError(f"Seurat RDS not found: {rds_path}")

    export_dir = os.path.join(workdir, "hdrg_export")
    r_script = os.path.join(workdir, "export_seurat.R")
    with open(r_script, "w") as f:
        f.write(R_EXPORT)

    print(f"\nExporting Seurat object via Rscript (max {max_per_type} cells/type) ...")
    subprocess.run(
        ["Rscript", r_script, rds_path, label_col, str(max_per_type), export_dir],
        check=True,
    )

    X = scipy.io.mmread(os.path.join(export_dir, "counts.mtx")).tocsr().T.tocsr()  # cells x genes
    genes = [g.strip().upper() for g in open(os.path.join(export_dir, "genes.txt")).read().splitlines()]
    meta = pd.read_csv(os.path.join(export_dir, "meta.csv"))

    adata = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=meta["cell"].astype(str)))
    adata.var_names = genes
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    adata.obs["celltype"] = meta["label"].astype(str).values
    adata.obs["source"] = "HDRG"
    print(f"  HDRG: {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
          f"{adata.obs['celltype'].nunique()} cell types")
    return adata


# ----------------------------------------------------------------------------
# Load Tabula Sapiens from Census (same filters as step1)
# ----------------------------------------------------------------------------
def load_ts(census_version, max_per_type, min_cells, tissue=None):
    print(f"\nOpening CELLxGENE Census (version='{census_version}') ...")
    with cellxgene_census.open_soma(census_version=census_version) as census:
        datasets = (
            census["census_info"]["datasets"]
            .read(column_names=["dataset_id", "collection_name"])
            .concat().to_pandas()
        )
        ts_ids = datasets[
            datasets["collection_name"].str.contains(TABULA_SAPIENS_COLLECTION, case=False, na=False)
        ]["dataset_id"].tolist()
        if not ts_ids:
            raise RuntimeError("No Tabula Sapiens datasets found in this Census version.")

        obs_filter = f"dataset_id in {ts_ids} and is_primary_data == True and disease == 'normal'"
        if tissue:
            obs_filter += f" and tissue_general == '{tissue}'"

        obs = cellxgene_census.get_obs(
            census, "homo_sapiens", value_filter=obs_filter,
            column_names=["soma_joinid", "cell_type"],
        )
        counts = obs["cell_type"].value_counts()
        keep = counts[counts >= min_cells].index
        obs = obs[obs["cell_type"].isin(keep)]

        # Subsample per cell type
        obs = (
            obs.groupby("cell_type", group_keys=False)
            .apply(lambda d: d.sample(min(len(d), max_per_type), random_state=0))
        )
        joinids = np.sort(obs["soma_joinid"].values)
        print(f"  Fetching {len(joinids):,} cells across {obs['cell_type'].nunique()} cell types ...")

        adata = cellxgene_census.get_anndata(
            census, organism="Homo sapiens",
            obs_coords=joinids.tolist(),
            obs_column_names=["cell_type"],
            X_name="raw",
        )

    adata.var_names = adata.var["feature_name"].astype(str).str.upper().values
    adata = adata[:, ~adata.var_names.duplicated()].copy()
    adata.obs_names = [f"TS_{i}" for i in adata.obs_names]
    adata.obs["celltype"] = adata.obs["cell_type"].astype(str).values
    adata.obs["source"] = "TabulaSapiens"
    adata.obs = adata.obs[["celltype", "source"]]
    print(f"  TS: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


# ----------------------------------------------------------------------------
# Integrate + UMAP
# ----------------------------------------------------------------------------
def integrate(a_hdrg, a_ts, n_hvg, n_pcs):
    shared = a_hdrg.var_names.intersection(a_ts.var_names)
    print(f"\nShared genes: {len(shared):,}")
    adata = ad.concat([a_hdrg[:, shared], a_ts[:, shared]], join="inner", index_unique=None)
    adata.obs["source"] = adata.obs["source"].astype("category")
    adata.obs["source_celltype"] = adata.obs["source"].astype(str) + ": " + adata.obs["celltype"]

    adata.X = sp.csr_matrix(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key="source", flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)

    rep = "X_pca"
    try:
        sc.external.pp.harmony_integrate(adata, key="source")
        rep = "X_pca_harmony"
        print("Harmony correction applied.")
    except Exception as e:
        print(f"WARNING: Harmony unavailable ({e}); using uncorrected PCA. "
              "Run `pip install harmonypy` for batch correction.")

    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=30)
    sc.tl.umap(adata)
    return adata, rep


# ----------------------------------------------------------------------------
# Plots + cell type correspondence
# ----------------------------------------------------------------------------
def make_plots(adata, rep, outdir):
    os.makedirs(outdir, exist_ok=True)
    um = adata.obsm["X_umap"]
    src = adata.obs["source"].values
    ct = adata.obs["celltype"].values
    colors = {"HDRG": "#d95f02", "TabulaSapiens": "#1b9e77"}

    # 1. by source
    fig, ax = plt.subplots(figsize=(8, 7))
    order = np.random.RandomState(0).permutation(adata.n_obs)
    ax.scatter(um[order, 0], um[order, 1], s=3, linewidths=0,
               c=[colors[s] for s in src[order]], rasterized=True)
    for k, c in colors.items():
        ax.scatter([], [], c=c, label=k, s=30)
    ax.legend(frameon=False); ax.set_title("Joint UMAP by source")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    fig.savefig(os.path.join(outdir, "umap_by_source.pdf"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 2. each atlas coloured by its own cell types, other in grey (labelled at centroids)
    for name in ["HDRG", "TabulaSapiens"]:
        fig, ax = plt.subplots(figsize=(10, 8))
        m = src == name
        ax.scatter(um[~m, 0], um[~m, 1], s=2, c="#dddddd", linewidths=0, rasterized=True)
        cats = sorted(set(ct[m]))
        cmap = plt.get_cmap("tab20" if len(cats) <= 20 else "nipy_spectral")
        for i, c in enumerate(cats):
            sel = m & (ct == c)
            col = cmap(i / max(len(cats) - 1, 1)) if len(cats) > 20 else cmap(i % 20)
            ax.scatter(um[sel, 0], um[sel, 1], s=4, color=col, linewidths=0, rasterized=True)
            ax.text(*np.median(um[sel], axis=0), c, fontsize=6, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))
        ax.set_title(f"{name} cell types (other atlas in grey)")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        fig.savefig(os.path.join(outdir, f"umap_{name}_celltypes.pdf"), bbox_inches="tight", dpi=200)
        plt.close(fig)

    # 3. all clusters together, source encoded by colour family, labelled at centroids
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(um[:, 0], um[:, 1], s=2, linewidths=0, rasterized=True,
               c=[colors[s] for s in src], alpha=0.5)
    for lab in sorted(set(adata.obs["source_celltype"])):
        sel = (adata.obs["source_celltype"] == lab).values
        s_name = lab.split(":")[0]
        ax.text(*np.median(um[sel], axis=0), lab.split(": ", 1)[1], fontsize=5,
                color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc=colors[s_name], alpha=0.35, lw=0))
    for k, c in colors.items():
        ax.scatter([], [], c=c, label=k, s=30)
    ax.legend(frameon=False); ax.set_title("Joint UMAP: all cell-type clusters")
    fig.savefig(os.path.join(outdir, "umap_joint_labelled.pdf"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 4. cell-type centroid correlation in the integrated embedding
    emb = adata.obsm[rep]
    cent = {}
    for lab in sorted(set(adata.obs["source_celltype"])):
        cent[lab] = emb[(adata.obs["source_celltype"] == lab).values].mean(axis=0)
    cent = pd.DataFrame(cent).T
    rows = [i for i in cent.index if i.startswith("HDRG: ")]
    cols = [i for i in cent.index if i.startswith("TabulaSapiens: ")]
    corr = np.corrcoef(cent.values)
    corr = pd.DataFrame(corr, index=cent.index, columns=cent.index).loc[rows, cols]
    corr.index = [r.split(": ", 1)[1] for r in rows]
    corr.columns = [c.split(": ", 1)[1] for c in cols]
    corr.to_csv(os.path.join(outdir, "celltype_similarity_matrix.csv"))

    best = pd.DataFrame({
        "hdrg_cell_type": corr.index,
        "best_ts_match": corr.idxmax(axis=1).values,
        "correlation": corr.max(axis=1).values,
    }).sort_values("correlation", ascending=False)
    best.to_csv(os.path.join(outdir, "best_celltype_matches.csv"), index=False)

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * corr.shape[1]), max(6, 0.25 * corr.shape[0])))
    im = ax.imshow(corr.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_xlabel("Tabula Sapiens"); ax.set_ylabel("HDRG")
    fig.colorbar(im, ax=ax, label="centroid correlation")
    fig.savefig(os.path.join(outdir, "celltype_similarity_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Joint UMAP: HDRG Seurat atlas vs Tabula Sapiens (Census)")
    p.add_argument("--seurat-rds", required=True, help="Path to HDRG Seurat .Rds")
    p.add_argument("--seurat-label-col", default="Atlas_annotation")
    p.add_argument("--census-version", default="stable")
    p.add_argument("--output-dir", default="01_ts_multiplex/step0c_umap")
    p.add_argument("--max-cells-per-type", type=int, default=300,
                   help="Subsample cap per cell type, per atlas (default 300)")
    p.add_argument("--min-cells", type=int, default=50, help="Min TS cells per cell type")
    p.add_argument("--tissue", default=None,
                   help="Optional Census tissue_general filter for TS, e.g. 'nervous system'")
    p.add_argument("--n-hvg", type=int, default=3000)
    p.add_argument("--n-pcs", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp:
        a_hdrg = load_hdrg(args.seurat_rds, args.seurat_label_col, args.max_cells_per_type, tmp)
    a_ts = load_ts(args.census_version, args.max_cells_per_type, args.min_cells, args.tissue)

    adata, rep = integrate(a_hdrg, a_ts, args.n_hvg, args.n_pcs)
    make_plots(adata, rep, args.output_dir)

    adata.write_h5ad(os.path.join(args.output_dir, "joint_umap.h5ad"))
    pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"]) \
        .join(adata.obs).to_csv(os.path.join(args.output_dir, "joint_umap_coordinates.csv"))
    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())