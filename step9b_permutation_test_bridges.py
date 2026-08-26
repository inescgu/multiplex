#!/usr/bin/env python3
"""
permutation_test_bridges.py
------------------------------------------------------------------------------
Empirical significance test for the multiplex-bridge genes flagged by
step9_multiplex_marchenko_pastur_v3.py.

WHY THIS EXISTS
----------------
Neither the "absolute" (>=2 minority-sign layers) nor the "fractional"
(>=X% of present layers on the minority side) bridge-flagging rule has a
real p-value behind it -- they're both just thresholds you pick. This
script replaces "is X% (or count N) enough?" with an actual empirical
null: if a gene's minority/majority Fiedler sign in each cell type were
random noise with the SAME per-cell-type base rate as your real data, how
often would a gene with that many cell types present reach its observed
fraction of minority-sign layers just by chance?

NOTE ON SIGN: a Fiedler eigenvector's sign is mathematically arbitrary
(eigsh could equally validly return it negated), so this script tests
MINORITY-sign consistency, not literal negativity -- it auto-detects
which sign (+/-) is less common in the data every time it's run (see
determine_minority_sign()). A real run showed a 94.5%/5.5% negative/
positive split; hardcoding "negative" there would have tested the
uninformative majority group entirely by accident.

METHOD
-------
For each cell-type column (i.e. each layer's Fiedler-sign column across
genes present in that layer), we repeatedly shuffle the negative/positive
labels among the genes present in that layer. This is a standard
label-permutation / column-permutation test:
  - it preserves each gene's number of cell types present (n_celltypes_
    present) exactly, since we only reshuffle WITHIN a column, never
    across genes' presence patterns;
  - it preserves each layer's marginal negative-vs-positive rate exactly;
  - it destroys any real cross-layer correlation in a gene's sign pattern
    -- which is exactly the thing "being a genuine multiplex bridge" is
    supposed to reflect. If a gene's real fraction-negative is no more
    extreme than this shuffled null, its "bridge" status is not
    distinguishable from chance co-occurrence given the marginal rates.

For each of --n-perm permutations we recompute, for every gene (== every
row, tracked by row/gene identity throughout), a synthetic frac_neg value.
Comparing the REAL frac_neg for each gene against its own row's
distribution of synthetic frac_neg values across all permutations gives a
one-sided empirical p-value:

    p_gene = (1 + #{permutations where synthetic frac_neg >= real frac_neg})
             / (1 + n_perm)

(+1 in numerator/denominator is the standard small-sample correction so
p is never exactly 0.) Genes with too few cell types present
(insufficient_data, same filter as v3's --min-celltypes-present) are
skipped -- there's no meaningful null to test them against.

Multiple-testing correction (Benjamini-Hochberg FDR) is applied across all
tested genes, since you're asking this question for hundreds of genes at
once.

OUTPUT
-------
  - <out>/bridge_permutation_results.csv : every tested gene with its real
    frac_neg, empirical p-value, BH-adjusted q-value, and a
    significant_at_fdr05 boolean.
  - <out>/permutation_null_diagram.png : histogram of the null distribution
    of "total genes flagged as bridges per permutation" (using the SAME
    frac_neg_threshold as your real run) vs. your real flagged count, for
    a global sanity check.
  - Printed summary: how many genes are significant at FDR 0.05 / 0.01,
    and how that compares to the raw flagged count from v3.

USAGE
------
    python permutation_test_bridges.py \\
        --bridges-csv 09_marchenko_pastur/multiplex_bridges.csv \\
        --out-dir 09_marchenko_pastur/permutation_test \\
        --frac-minority-threshold 0.6 \\
        --min-celltypes-present 3 \\
        --n-perm 2000
(match --frac-minority-threshold / --min-celltypes-present to whatever you used
in the v3 run you're testing.)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def determine_minority_sign(values):
    """
    Same fix as v5 of the main script, applied here independently since
    this script re-derives everything from raw fiedler_<celltype> values
    rather than trusting any precomputed column: a Fiedler eigenvector's
    sign is arbitrary, so "negative" is not inherently meaningful --
    what matters is which sign is the MINORITY. A real run showed a
    94.5%/5.5% split; hardcoding "negative" would have tested the wrong
    (majority, uninformative) group entirely. Detected fresh from the data
    every time, not assumed.
    """
    flat = values[~np.isnan(values)]
    n_neg = int(np.sum(flat < 0))
    n_pos = int(np.sum(flat > 0))
    total = flat.size
    print(f"Global Fiedler sign split (from raw values in this CSV): "
          f"{n_neg:,} negative ({100*n_neg/total:.1f}%), "
          f"{n_pos:,} positive ({100*n_pos/total:.1f}%)")
    if n_neg == 0 or n_pos == 0:
        print("  WARNING: one sign is entirely empty -- defaulting to negative.")
        return -1
    minority_sign = -1 if n_neg < n_pos else 1
    print(f"  Minority sign: {'negative' if minority_sign == -1 else 'positive'} "
          f"-- testing MINORITY-side consistency, not literal negativity.")
    return minority_sign


def load_fiedler_matrix(bridges_csv):
    """Rebuild the (genes x celltypes) boolean-minority-sign matrix (with
    NaN for absent) directly from multiplex_bridges.csv's fiedler_<celltype>
    columns -- re-derived here so the permutation test doesn't depend on
    the main script's internals, just its raw per-layer Fiedler values."""
    df = pd.read_csv(bridges_csv)
    fiedler_cols = [c for c in df.columns if c.startswith("fiedler_") and c not in
                    ("fiedler_mean", "fiedler_std")]
    celltypes = [c[len("fiedler_"):] for c in fiedler_cols]
    values = df[fiedler_cols].to_numpy(dtype=float)   # NaN where absent
    minority_sign = determine_minority_sign(values)
    minority = np.where(np.isnan(values), np.nan,
                         ((values * minority_sign) > 0).astype(float))
    return df, minority, celltypes, minority_sign


def permute_columns(neg, rng):
    """One permutation: independently shuffle the non-NaN entries within
    each column (cell type). Preserves each gene's NaN pattern (hence its
    n_celltypes_present) and each column's marginal negative rate exactly;
    destroys cross-column (cross-cell-type) correlation for each gene."""
    out = neg.copy()
    for j in range(neg.shape[1]):
        col = out[:, j]
        present_idx = np.where(~np.isnan(col))[0]
        if present_idx.size < 2:
            continue
        vals = col[present_idx].copy()
        rng.shuffle(vals)
        out[present_idx, j] = vals
    return out


def frac_minority_per_row(minority):
    with np.errstate(invalid="ignore"):
        n_present = np.sum(~np.isnan(minority), axis=1)
        n_minority = np.nansum(minority, axis=1)
    frac = np.divide(n_minority, n_present, out=np.full(n_present.shape, np.nan), where=n_present > 0)
    return frac, n_present


def benjamini_hochberg(pvals):
    """Standard BH step-up FDR correction. Returns q-values in the same
    order as the input p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridges-csv", required=True,
                     help="multiplex_bridges.csv from step9_multiplex_marchenko_pastur_v3.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-celltypes-present", type=int, default=3)
    ap.add_argument("--frac-minority-threshold", type=float, default=0.6,
                     help="Must match the threshold used to flag bridges in your "
                          "real v3 run, so the null total-flagged-count comparison "
                          "is apples-to-apples.")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fdr", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df, minority, celltypes, minority_sign = load_fiedler_matrix(args.bridges_csv)
    n_genes = minority.shape[0]
    print(f"Loaded {n_genes:,} genes x {len(celltypes)} cell types from {args.bridges_csv}")

    real_frac, n_present = frac_minority_per_row(minority)
    testable = n_present >= args.min_celltypes_present
    n_testable = int(testable.sum())
    print(f"{n_testable:,} / {n_genes:,} genes present in >= {args.min_celltypes_present} "
          f"cell types (testable); {n_genes - n_testable:,} excluded as insufficient_data")

    # ---- per-gene permutation p-values ------------------------------------
    ge_count = np.zeros(n_genes, dtype=np.int64)   # count of perms with synthetic >= real
    null_flagged_totals = np.zeros(args.n_perm, dtype=np.int64)

    print(f"Running {args.n_perm:,} permutations ...")
    for p in range(args.n_perm):
        minority_perm = permute_columns(minority, rng)
        frac_perm, n_present_perm = frac_minority_per_row(minority_perm)
        # sanity: n_present_perm should equal n_present exactly (NaN pattern untouched)
        with np.errstate(invalid="ignore"):
            ge = (frac_perm >= real_frac)
        ge_count += np.where(testable & ~np.isnan(frac_perm), ge, False).astype(np.int64)

        null_flagged = testable & (frac_perm >= args.frac_minority_threshold)
        null_flagged_totals[p] = int(null_flagged.sum())

        if (p + 1) % max(1, args.n_perm // 10) == 0:
            print(f"  {p+1:,}/{args.n_perm:,} permutations done")

    pvals = np.full(n_genes, np.nan)
    pvals[testable] = (1 + ge_count[testable]) / (1 + args.n_perm)

    qvals = np.full(n_genes, np.nan)
    qvals[testable] = benjamini_hochberg(pvals[testable])

    df_out = df.copy()
    df_out["n_celltypes_present_check"] = n_present
    df_out["real_frac_minority"] = real_frac
    df_out["testable"] = testable
    df_out["perm_pvalue"] = pvals
    df_out["perm_qvalue_bh"] = qvals
    df_out[f"significant_at_fdr{args.fdr:g}"] = testable & (qvals <= args.fdr)

    df_out = df_out.sort_values("perm_pvalue", na_position="last")
    results_csv = out / "bridge_permutation_results.csv"
    df_out.to_csv(results_csv, index=False)

    n_sig = int(df_out[f"significant_at_fdr{args.fdr:g}"].sum())
    n_raw_flagged = int(df.get("is_multiplex_bridge", pd.Series(dtype=bool)).sum()) \
        if "is_multiplex_bridge" in df.columns else None

    print(f"\nSaved per-gene results -> {results_csv}")
    print(f"\n=== SUMMARY ===")
    print(f"Testable genes (>= {args.min_celltypes_present} cell types present): {n_testable:,}")
    if n_raw_flagged is not None:
        print(f"Raw flagged bridges from input CSV (frac >= {args.frac_minority_threshold}, "
              f"no significance test): {n_raw_flagged:,}")
    print(f"Genes significant at FDR <= {args.fdr}: {n_sig:,}")
    for alt_fdr in (0.01, 0.10):
        n_alt = int((testable & (qvals <= alt_fdr)).sum())
        print(f"  (for reference, at FDR <= {alt_fdr}: {n_alt:,})")

    # ---- global null-vs-observed plot --------------------------------------
    real_flagged_count = int((testable & (real_frac >= args.frac_minority_threshold)).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_flagged_totals, bins=40, alpha=0.7, color="steelblue",
            label="null: # genes flagged per permutation")
    ax.axvline(real_flagged_count, color="crimson", lw=2,
                label=f"observed: {real_flagged_count} genes")
    ax.set_xlabel("# genes with frac_minority >= threshold")
    ax.set_ylabel("# permutations")
    ax.set_title(f"Null distribution of flagged-bridge count "
                 f"(threshold={args.frac_minority_threshold}, {args.n_perm} permutations)")
    ax.legend()
    fig.tight_layout()
    plot_path = out / "permutation_null_diagram.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved null-distribution plot -> {plot_path}")

    empirical_p_global = (1 + int(np.sum(null_flagged_totals >= real_flagged_count))) / (1 + args.n_perm)
    print(f"Global empirical p-value (observed flagged-count vs. null): {empirical_p_global:.4f}")

    summary = dict(
        bridges_csv=str(args.bridges_csv),
        n_genes=n_genes, n_testable=n_testable,
        min_celltypes_present=args.min_celltypes_present,
        frac_neg_threshold=args.frac_minority_threshold,
        n_perm=args.n_perm,
        n_significant_fdr=n_sig, fdr_level=args.fdr,
        real_flagged_count=real_flagged_count,
        global_empirical_pvalue=empirical_p_global,
    )
    with open(out / "permutation_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved summary -> {out / 'permutation_test_summary.json'}")


if __name__ == "__main__":
    main()