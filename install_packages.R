# install_packages.R
# Run via install_r_pkgs.sh, NOT bare `Rscript install_packages.R` — that
# script sets LD_LIBRARY_PATH first, which every C++ package here needs
# both to compile AND to load successfully afterward.

repos <- "https://cloud.r-project.org"

# --- 1. Plain CRAN package# install_packages.R
# Run via install_r_pkgs.sh, NOT bare `Rscript install_packages.R` — that
# script sets LD_LIBRARY_PATH first, which every C++ package here needs
# both to compile AND to load successfully afterward.

repos <- "https://cloud.r-project.org"

# --- 1. Plain CRAN packages your scripts actually need --------------------
# optparse/dplyr/tidyr were missing entirely last run (that's the final
# "there is no package called 'optparse'" error) — added here.
cran_pkgs <- c("optparse", "dplyr", "tidyr", "Seurat", "BiocManager", "remotes", "devtools")
to_install <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, repos = repos)

# --- 2. Bioconductor dependencies CellChat needs ---------------------------
# Installing these explicitly via BiocManager rather than hoping
# remotes::install_github resolves them automatically.
bioc_pkgs <- c("Biobase", "BiocGenerics", "BiocNeighbors", "ComplexHeatmap")
missing_bioc <- setdiff(bioc_pkgs, rownames(installed.packages()))
if (length(missing_bioc)) BiocManager::install(missing_bioc, update = FALSE, ask = FALSE)

# --- 3. GitHub packages -----------------------------------------------------
# force = TRUE on the ones that got a broken .so last time (they show as
# "installed" but fail to load) — otherwise remotes sees the same commit SHA
# and skips reinstalling, leaving the broken binary in place.
devtools::install_github("sqjin/CellChat", force = TRUE)
devtools::install_github("Wei-BioMath/NeuronChat", force = TRUE)   # depends on CellChat above

# basilisk + basilisk.utils MUST be a matched pair (liana's basilisk-based
# setup calls internal basilisk.utils functions like `useSystemDir`, which
# only exist in versions basilisk expects). Pulling them independently from
# GitHub HEAD — as remotes does automatically when resolving liana's deps —
# can land on two commits that don't agree with each other. Installing both
# from the same Bioconductor release instead guarantees they match.
BiocManager::install(c("basilisk.utils", "basilisk"), update = TRUE, ask = FALSE, force = TRUE)

devtools::install_github("saezlab/liana", force = TRUE)             # pulls in metapod/scran
devtools::install_github("immunogenomics/presto", force = TRUE)

cat("\n=== Verifying ===\n")
for (p in c("Seurat", "CellChat", "NeuronChat", "liana", "presto",
            "optparse", "dplyr", "tidyr")) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("  %-12s %s\n", p, if (ok) "OK" else "MISSING"))
}