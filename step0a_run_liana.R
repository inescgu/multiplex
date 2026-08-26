#!/usr/bin/env Rscript
# Step 0a: Run LIANA (the R package, NOT liana-py — no AnnData conversion
# needed) directly on the same Seurat object used for NeuronChat in
# step0b_run_neuronchat.R. LIANA's R implementation wraps CellPhoneDB,
# NATMI, Connectome, SingleCellSignalR, and CellChat's LR-inference logic,
# plus its own robust-rank-aggregate consensus across all of them.
#
# Output schema matches step0b exactly, so step0c can merge both directly:
#   ligand, receptor, source, score, cell_type_source, cell_type_target
#
# ASSUMPTIONS (matches seurat_object_preprint_release.Rds structure):
#   - Cell type identity lives in meta.data$Atlas_annotation (not a generic
#     "cell_type" column) — default --celltype-col below reflects this.
#   - Object has two assays ("RNA" and "integrated"); we explicitly drop
#     "integrated" and run on RNA (log-normalized data) only, since
#     "integrated" is batch-corrected for clustering, not appropriate for
#     expression-magnitude-based LR scoring — same choice your own
#     extract_expression_matrix.R script made.
#   - resource = "Consensus" (LIANA's curated human LR resource). For mouse,
#     switch to resource = "MouseConsensus".
#   - Some rownames in this atlas are placeholders like
#     "DEPRECATED-ENSG00000179826" (visible in your
#     extract_expression_matrix.R output) rather than real gene symbols —
#     dropped before running LIANA so they don't pollute the LR gene universe.
#
# CAUTION: liana_aggregate()'s exact output column names (source, target,
# ligand.complex, receptor.complex, aggregate_rank, and the per-method score
# columns below) have shifted across liana versions before. The first time
# you run this, do `colnames(liana_res)` / `colnames(liana_aggr)`
# interactively and check them against what's assumed in
# extract_method_scores() below before trusting the output.
#
# KNOWN LIANA BUG (worked around below): liana's Seurat-input code path
# (liana_prep.Seurat) still calls GetAssayData(..., slot = "counts"), which
# SeuratObject 5.0.0+ made fully defunct (not just deprecated) — see
# https://github.com/saezlab/liana/issues/194, still open, unfixed as of
# this writing. Passing a Seurat object directly to liana_wrap() errors.
# Workaround: convert to a SingleCellExperiment via Seurat's own (working)
# as.SingleCellExperiment() first, and hand THAT to liana_wrap() — liana
# accepts SCE input natively and this path never calls the broken function.

suppressMessages({
  library(optparse)
  library(Seurat)
  library(liana)
  library(dplyr)
  library(tidyr)
})
# NOTE: deliberately NOT library(SingleCellExperiment) here — attaching it
# pulls in BiocGenerics, which masks setdiff()/intersect()/union() with S4
# generics for the rest of the script, including inside Seurat's own
# subset()/Assays() internals (which don't base::-qualify their calls) —
# causing "no method for coercing this S4 class to a vector" errors at
# whatever unrelated call happens to use one of those functions next.
# SingleCellExperiment::as.SingleCellExperiment() below is called with an
# explicit :: instead, so the package is used without ever being attached.

option_list <- list(
  make_option("--seurat-rds", type = "character"),
  make_option("--celltype-col", type = "character", default = "Atlas_annotation"),
  make_option("--assay", type = "character", default = "RNA"),
  make_option("--resource", type = "character", default = "Consensus"),
  make_option("--methods", type = "character",
              default = "natmi,connectome,logfc,sca,cellphonedb",
              help = "comma-separated liana method names (default: natmi,connectome,logfc,sca,cellphonedb; 'cellchat' can be added if your liana version supports it)"),
  make_option("--min-cells", type = "integer", default = 5),
  make_option("--min-detect-frac", type = "double", default = 0,
              help = "drop genes detected in fewer than this fraction of ALL cells before running LIANA (0 = disabled, keep every gene)"),
  make_option("--max-cells-per-type", type = "integer", default = 0,
              help = "downsample each cell type to at most this many cells before running LIANA (0 = disabled, keep every cell)"),
  make_option("--seed", type = "integer", default = 1, help = "random seed, only used if --max-cells-per-type > 0"),
  make_option("--drop-counts", action = "store_true", default = TRUE,
              help = "alias the SCE's 'counts' assay to 'logcounts' after conversion (default TRUE) — liana requires both assays present but never reads counts' values for natmi/connectome/logfc/sca/cellphonedb, and R's copy-on-write means aliasing costs ~no extra memory vs the real duplicate raw-counts matrix Seurat's conversion produces. Pass --no-drop-counts to keep the real counts matrix."),
  make_option("--no-drop-counts", action = "store_false", dest = "drop-counts"),
  make_option("--batch-by-pair", action = "store_true", default = FALSE,
              help = "process one cell-type pair at a time instead of the whole atlas at once (default FALSE). Peak memory becomes bounded by your largest single pair's cell count rather than the whole atlas, at the cost of ~(n_celltypes choose 2) separate liana_wrap() calls and longer runtime. Uses every gene and every cell, just not all simultaneously. NOTE: this changes cellphonedb's permutation null slightly — labels are shuffled only within each pair's cells rather than across the full 22-type pool — statistically reasonable but not bit-identical to a whole-atlas run."),
  make_option("--out", type = "character")
)
opt <- parse_args(OptionParser(option_list = option_list))
methods <- strsplit(opt$methods, ",")[[1]]
set.seed(opt$seed)

cat(sprintf("Loading Seurat object from %s ...\n", opt$`seurat-rds`))
seu <- readRDS(opt$`seurat-rds`)

# Keep only the requested assay (default RNA) — drop "integrated" so LIANA
# can't accidentally pick it up as DefaultAssay, and drop DEPRECATED-ENSG*
# placeholder rownames from the gene universe.
DefaultAssay(seu) <- opt$assay
other_assays <- base::setdiff(Assays(seu), opt$assay)  # base:: kept defensively — a masked setdiff() from any Bioconductor package caused this exact error before
for (a in other_assays) seu[[a]] <- NULL
keep_genes <- rownames(seu)[!grepl("^DEPRECATED-", rownames(seu))]
seu <- subset(seu, features = keep_genes)

# --- Optional gene-detection filter (off by default, --min-detect-frac 0) --
if (opt$`min-detect-frac` > 0) {
  counts_mat <- GetAssayData(seu, assay = opt$assay, layer = "counts")
  detect_frac <- Matrix::rowSums(counts_mat > 0) / ncol(counts_mat)
  keep_detected <- rownames(seu)[detect_frac >= opt$`min-detect-frac`]
  cat(sprintf("Gene filter: keeping %d / %d genes detected in >= %.1f%% of cells\n",
              length(keep_detected), nrow(seu), 100 * opt$`min-detect-frac`))
  seu <- subset(seu, features = keep_detected)
  rm(counts_mat, detect_frac); gc(verbose = FALSE)
}

# --- Optional cell downsampling (off by default, --max-cells-per-type 0) ---
if (opt$`max-cells-per-type` > 0) {
  Idents(seu) <- seu@meta.data[[opt$`celltype-col`]]
  keep_cells <- unlist(lapply(levels(Idents(seu)), function(ct) {
    cells_ct <- WhichCells(seu, idents = ct)
    if (length(cells_ct) > opt$`max-cells-per-type`) {
      sample(cells_ct, opt$`max-cells-per-type`)
    } else {
      cells_ct
    }
  }))
  cat(sprintf("Cell downsampling: keeping %d / %d cells (cap %d/cell type)\n",
              length(keep_cells), ncol(seu), opt$`max-cells-per-type`))
  seu <- subset(seu, cells = keep_cells)
}

Idents(seu) <- seu@meta.data[[opt$`celltype-col`]]
cat(sprintf("  %d cells x %d genes; %d cell types (idents_col='%s')\n\n",
            ncol(seu), nrow(seu), length(unique(Idents(seu))), opt$`celltype-col`))

# Convert to SCE instead of passing the Seurat object directly to liana_wrap
# — see the KNOWN LIANA BUG note above. Seurat::as.SingleCellExperiment()
# (defined by Seurat itself, not the SingleCellExperiment package — Bioc's
# package has no knowledge of Seurat objects) maps the RNA assay's
# "counts"/"data" layers to SCE's "counts"/"logcounts" assays, and carries
# meta.data (including Atlas_annotation) into colData, so idents_col below
# still resolves the same way it would for a Seurat object.
cat("Converting Seurat object to SingleCellExperiment (works around saezlab/liana#194) ...\n")
sce <- Seurat::as.SingleCellExperiment(seu, assay = opt$assay)
rm(seu); gc(verbose = FALSE)  # the Seurat object is fully superseded by sce now — free it

cat(sprintf("  logcounts assay class: %s (sparse if this says 'dgCMatrix', not 'matrix')\n",
            paste(class(SummarizedExperiment::assay(sce, "logcounts")), collapse = "/")))

if (isTRUE(opt$`drop-counts`) && "counts" %in% SummarizedExperiment::assayNames(sce)) {
  # liana validates that BOTH `counts` and `logcounts` are present in the SCE
  # (even though none of natmi/connectome/logfc/sca/cellphonedb read counts'
  # actual values) — so we can't delete the assay outright, that fails the
  # check before any scoring runs. Instead, alias `counts` to the SAME
  # object as `logcounts`. R's copy-on-write semantics mean this doesn't
  # duplicate the underlying matrix in memory (a real copy only happens on
  # modification, and liana never modifies `counts` here) — so we keep
  # essentially all the memory savings while satisfying the presence check.
  cat("Aliasing 'counts' assay to 'logcounts' (liana requires both present, but never reads counts' values for these methods; copy-on-write means this costs ~no extra memory) ...\n")
  SummarizedExperiment::assay(sce, "counts") <- SummarizedExperiment::assay(sce, "logcounts")
  gc(verbose = FALSE)
}

cat(sprintf("Running LIANA methods: %s (resource='%s') ...\n",
            paste(methods, collapse = ", "), opt$resource))

# --- Per-method + consensus long-format extraction, reused per batch -------
# liana_res is a named list, one data.frame per method in `methods`, each
# with (at minimum) source, target, ligand.complex, receptor.complex, plus
# that method's own score column. Column names below are liana's current
# convention (v0.1.x) — verify with colnames(liana_res[[<method>]]) if this
# errors or looks wrong for your version.
method_score_cols <- list(
  cellphonedb = "lr.mean",
  natmi = "edge_specificity",
  connectome = "weight_sc",
  sca = "LRscore",
  logfc = "logfc_comb"
)

extract_method_scores <- function(liana_res) {
  rows <- list()
  for (m in names(liana_res)) {
    df <- liana_res[[m]]
    score_col <- method_score_cols[[m]]
    if (is.null(score_col) || !(score_col %in% colnames(df))) {
      cat(sprintf("  ! skipping '%s': expected score column not found (have: %s)\n",
                   m, paste(colnames(df), collapse = ", ")))
      next
    }
    sub <- df %>%
      transmute(
        ligand = toupper(ligand.complex),
        receptor = toupper(receptor.complex),
        cell_type_source = source,   # must come BEFORE `source` is overwritten below
        cell_type_target = target,
        score = .data[[score_col]],
        source = m
      )
    rows[[m]] <- sub
  }
  bind_rows(rows)
}

run_liana_on <- function(sce_subset) {
  liana_res <- liana_wrap(
    sce_subset,
    method = methods,
    resource = opt$resource,
    idents_col = opt$`celltype-col`,
    min_cells = opt$`min-cells`,
    verbose = FALSE
  )
  liana_aggr <- liana_aggregate(liana_res)
  long_methods <- extract_method_scores(liana_res)
  consensus_long <- liana_aggr %>%
    transmute(
      ligand = toupper(ligand.complex),
      receptor = toupper(receptor.complex),
      cell_type_source = source,   # must come BEFORE `source` is overwritten below
      cell_type_target = target,
      score = 1 - aggregate_rank,  # invert so higher = stronger, matches other sources
      source = "liana_consensus"
    )
  bind_rows(long_methods, consensus_long) %>%
    filter(!is.na(ligand), !is.na(receptor), cell_type_source != cell_type_target)
}

empty_long_template <- data.frame(
  ligand = character(), receptor = character(),
  cell_type_source = character(), cell_type_target = character(),
  score = numeric(), source = character()
)

if (opt$`batch-by-pair`) {
  # One cell-type pair at a time — peak memory bounded by the largest single
  # pair's cells, not the whole atlas. All genes/cells still used, just not
  # loaded into liana_wrap() simultaneously. See --batch-by-pair help text
  # above for the cellphonedb-permutation-null caveat.
  cts <- levels(factor(SummarizedExperiment::colData(sce)[[opt$`celltype-col`]]))
  pairs <- combn(cts, 2, simplify = FALSE)
  cat(sprintf("Batching by pair: %d cell types -> %d unordered pairs\n", length(cts), length(pairs)))

  # Checkpoint each pair's result to its own file the moment it's computed,
  # and skip pairs whose checkpoint already exists on disk. This means a
  # SLURM timeout mid-run costs NOTHING except needing to resubmit the exact
  # same command — already-finished pairs are skipped instantly and the run
  # picks up where it left off, instead of losing everything computed so far
  # (which is what the old single-write-at-the-end version did).
  ckpt_dir <- paste0(tools::file_path_sans_ext(opt$out), "_pair_checkpoints")
  dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)
  sanitize <- function(x) gsub("[^A-Za-z0-9_.-]", "-", x)

  for (i in seq_along(pairs)) {
    ct_a <- pairs[[i]][1]; ct_b <- pairs[[i]][2]
    ckpt_file <- file.path(ckpt_dir, sprintf("%s__%s.csv", sanitize(ct_a), sanitize(ct_b)))
    if (file.exists(ckpt_file)) {
      cat(sprintf("  [%d/%d] %s <-> %s ... already done, skipping\n", i, length(pairs), ct_a, ct_b))
      next
    }
    cat(sprintf("  [%d/%d] %s <-> %s ...\n", i, length(pairs), ct_a, ct_b))
    keep <- SummarizedExperiment::colData(sce)[[opt$`celltype-col`]] %in% c(ct_a, ct_b)
    sce_pair <- sce[, keep]
    pair_result <- tryCatch(run_liana_on(sce_pair),
                             error = function(e) {
                               cat(sprintf("    ! failed on %s <-> %s: %s\n", ct_a, ct_b, conditionMessage(e)))
                               NULL
                             })
    rm(sce_pair); gc(verbose = FALSE)
    if (is.null(pair_result)) pair_result <- empty_long_template
    write.csv(pair_result, ckpt_file, row.names = FALSE)  # written immediately — this pair survives a timeout from here on
  }

  cat("All pairs processed (or already checkpointed) — concatenating ...\n")
  ckpt_files <- list.files(ckpt_dir, pattern = "\\.csv$", full.names = TRUE)
  long_df <- bind_rows(lapply(ckpt_files, function(f) {
    tryCatch(read.csv(f, stringsAsFactors = FALSE), error = function(e) NULL)
  }))
} else {
  long_df <- run_liana_on(sce)
}

dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
write.csv(long_df, opt$out, row.names = FALSE)
cat(sprintf("Wrote %d long-format LR rows -> %s\n", nrow(long_df), opt$out))
print(table(long_df$source))