#!/usr/bin/env Rscript
# Step 0b: Run NeuronChat (R-only, no Python port) and export a long-format
# LR edge table with the SAME core schema as step0a_run_liana.py's output:
#   ligand, receptor, source, score, cell_type_source, cell_type_target
# so step0c can merge both directly. `source` is always the literal string
# "neuronchat" (not e.g. "neuronchat_neurotransmitter") — step2's
# add_scored_lr_edges()/main() splits scored-LR rows into liana vs
# neuronchat via an EXACT match on source == "neuronchat", so a category-
# specific source string would silently get bucketed into "liana" instead.
# See PLUS ONE EXTRA COLUMN below for where the category actually goes.
#
# PLUS ONE EXTRA COLUMN vs step0a: `ligand_type`, NeuronChat's own broad
# ligand category (e.g. "Neurotransmitter", "Neuropeptide", "Gas", "Gap
# junction protein" — pulled from x@DB[[lr_name]]$ligand_type, see
# ASSUMPTIONS below). step2 doesn't read this column today and will ignore
# it harmlessly (pandas keeps extra columns, downstream code only accesses
# the ones it names); it's here for optional future filtering/analysis
# without touching source's role as the method identifier.
#
# ASSUMPTIONS (matches seurat_object_preprint_release.Rds structure):
#   - Cell type identity lives in meta.data$Atlas_annotation (not a generic
#     "cell_type" column) — default --celltype-col below reflects this.
#   - Object has two assays ("RNA" and "integrated"); we explicitly use RNA
#     (log-normalized data), same choice your own extract_expression_matrix.R
#     script made — "integrated" is batch-corrected for clustering, not
#     appropriate for expression-magnitude-based LR scoring.
#   - Seurat v5 object -> GetAssayData(..., layer = "data"), not the old
#     slot = "data" argument.
#   - species = "human"; NeuronChat also ships a mouse database
#     (interactionDB_mouse) — swap createNeuronChat(..., DB = 'mouse') and
#     the relevant NeuronChat DB call below if needed.
#   - Some Atlas_annotation values contain "/" (e.g. "C-NP.MRGPRX1/GFRA2")
#     — fine as CSV cell values here; step1/step2 already sanitize "/" out
#     of filenames downstream, no change needed in this script for that.
#   - x@DB is a NAMED LIST OF LISTS (not a data.frame): each top-level
#     element is one LR pair (named identically to names(x@net), e.g.
#     "5HT_HTR1A"), and each element has fields interaction_name,
#     interaction_type, ligand_type, etc. Confirmed via str(x@DB) on the
#     installed version. ligand_type gives the broad category (e.g.
#     "Neurotransmitter", "Neuropeptide", "Gas", "Gap junction protein").
#
# NeuronChat's own README/vignette is the source of truth for exact function
# names/args in your installed version — this mirrors the standard
# createNeuronChat() -> run_NeuronChat() -> net_aggregated_x workflow, but
# verify against `?NeuronChat::run_NeuronChat` before trusting output.

suppressMessages({
  library(optparse)
  library(Seurat)
  library(NeuronChat)
  library(dplyr)
})

option_list <- list(
  make_option("--seurat-rds", type = "character"),
  make_option("--celltype-col", type = "character", default = "Atlas_annotation"),
  make_option("--assay", type = "character", default = "RNA"),
  make_option("--species", type = "character", default = "human"),
  make_option("--out", type = "character")
)
opt <- parse_args(OptionParser(option_list = option_list))

cat(sprintf("Loading Seurat object from %s ...\n", opt$`seurat-rds`))
seu <- readRDS(opt$`seurat-rds`)

expr_mat <- GetAssayData(seu, assay = opt$assay, layer = "data")  # normalized, Seurat v5 syntax
rownames(expr_mat) <- toupper(rownames(expr_mat))
# This atlas has some placeholder rownames like "DEPRECATED-ENSG00000179826"
# (visible in your extract_expression_matrix.R output) — not real gene
# symbols, drop them so they don't pollute the LR gene universe.
expr_mat <- expr_mat[!grepl("^DEPRECATED-", rownames(expr_mat)), ]
cell_types <- as.character(seu@meta.data[[opt$`celltype-col`]])
cat(sprintf("  %d cells x %d genes (after dropping DEPRECATED- placeholders); %d cell types\n\n",
            ncol(expr_mat), nrow(expr_mat), length(unique(cell_types))))

cat("Creating NeuronChat object and running pairwise communication ...\n")
db_name <- if (opt$species == "mouse") "mouse" else "human"
x <- createNeuronChat(as.matrix(expr_mat), DB = db_name, group.by = cell_types)
x <- run_NeuronChat(x, M = 100)  # M = permutations; see NeuronChat docs

# Save the object so downstream debugging/iteration doesn't require
# re-running the ~9 min computation above.
saveRDS(x, file.path(dirname(opt$out), "neuronchat_object_debug.rds"))

# Build lr_name -> ligand-category lookup. x@DB is a named list of lists
# (see ASSUMPTIONS above); sapply() pulls the ligand_type field out of every
# element, giving a named character vector whose names match names(x@net).
lr_category <- sapply(x@DB, function(pair) pair$ligand_type)

cat("Flattening to long format ...\n")
rows <- list()
i <- 1
for (lr_name in names(x@net)) {
  mat <- x@net[[lr_name]]  # source (rows) x target (cols) strength matrix
  parts <- strsplit(lr_name, "_")[[1]]
  ligand <- toupper(parts[1])
  receptor <- toupper(paste(parts[-1], collapse = "_"))

  # Single-bracket indexing: returns NA (not an error) if lr_name is
  # somehow missing from lr_category, unlike [[ ]] which would halt the loop.
  category <- lr_category[lr_name]
  if (is.na(category)) category <- "unknown"

  for (src in rownames(mat)) {
    for (tgt in colnames(mat)) {
      val <- mat[src, tgt]
      if (!is.na(val) && val > 0 && src != tgt) {
        rows[[i]] <- data.frame(
          ligand = ligand, receptor = receptor, source = "neuronchat",
          score = val, cell_type_source = src, cell_type_target = tgt,
          ligand_type = category
        )
        i <- i + 1
      }
    }
  }
}
long_df <- bind_rows(rows)

dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
write.csv(long_df, opt$out, row.names = FALSE)
cat(sprintf("Wrote %d long-format LR rows -> %s\n", nrow(long_df), opt$out))