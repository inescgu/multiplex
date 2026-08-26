#!/bin/bash
#SBATCH --job-name=install_r_pkgs
#SBATCH --output=slurm_logs/install_r_pkgs_%j.out
#SBATCH --error=slurm_logs/install_r_pkgs_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=8G

cd /lisc/home/user/gerardursin/multiplex/

module load R

# --- THE FIX -----------------------------------------------------------
# gcc/g++ used to compile packages here is conda-forge GCC 14.3.0, which
# ships its own newer libstdc++.so.6 (has GLIBCXX_3.4.32). By default the
# dynamic loader finds the OLDER system one in /lib64 instead, so any
# compiled C++ package (CellChat, metapod, presto, ...) builds fine but
# then fails with "GLIBCXX_3.4.32 not found" the moment R tries to load
# it. Point LD_LIBRARY_PATH at the same compiler's lib dir so R finds the
# matching libstdc++ both now (while installing) and later (whenever you
# `library(CellChat)` etc. in any future script/job).
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
echo "Using libstdc++ from: $GCC_LIBDIR"
export LD_LIBRARY_PATH="$GCC_LIBDIR:$LD_LIBRARY_PATH"
# -------------------------------------------------------------------------

Rscript install_packages.R