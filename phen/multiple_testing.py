"""
multiple_testing.py

ONE implementation of multiple-testing correction for every enrichment
analyzer (proteomics, phenotypes, neuropathic pain, spatial, OMIM PS,
Orphanet).  The method and alpha live in pipeline_config.py
(MULTIPLE_TESTING_METHOD, ALPHA), so changing them there changes every
analysis consistently.

RULES THE ANALYZERS FOLLOW
--------------------------
1. The family of tests is EVERY test actually performed in one analysis
   (all cluster x term pairs), not just the one "best" term per cluster.
   Picking the best term first and correcting only across clusters ignores
   the selection step and is anti-conservative.
2. Tests that were skippable-by-construction (e.g. zero overlap, so p == 1
   by definition) are counted via `n_untested` so the family size is
   (clusters x terms), not "whatever happened to overlap".
3. Selecting a "best" term per cluster (by fold enrichment, combined score,
   odds ratio, ...) happens AFTER correction; the reported adjusted p-value
   of that row is the adjusted p-value of that specific test.
4. Every summary CSV has `p_value` (raw) and `adjusted_p_value`
   (corrected with the configured method); every gene-level CSV has
   `cluster_p_value`, `cluster_adjusted_p_value`, `is_cluster_significant`.
"""
import numpy as np
from statsmodels.stats.multitest import multipletests

_METHOD_LABELS = {
    "fdr_bh": "Benjamini-Hochberg FDR",
    "fdr_by": "Benjamini-Yekutieli FDR",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
}


def get_method():
    """Configured correction method (pipeline_config.MULTIPLE_TESTING_METHOD)."""
    import pipeline_config as cfg
    method = getattr(cfg, "MULTIPLE_TESTING_METHOD", "fdr_bh")
    if method not in _METHOD_LABELS:
        raise ValueError(
            f"Unsupported MULTIPLE_TESTING_METHOD '{method}'. "
            f"Choose one of {sorted(_METHOD_LABELS)}")
    return method


def get_alpha():
    """Configured significance level (pipeline_config.ALPHA)."""
    import pipeline_config as cfg
    return float(getattr(cfg, "ALPHA", 0.05))


def method_label(method=None):
    """Human-readable name, e.g. for print statements and plot titles."""
    return _METHOD_LABELS[method or get_method()]


def adjust_pvalues(pvals, method=None, n_untested=0):
    """Correct a family of p-values.

    Parameters
    ----------
    pvals : array-like of raw p-values (one per test performed). NaNs are
        excluded from the family and returned as NaN.
    method : statsmodels method name; defaults to the configured one.
    n_untested : number of additional tests that belong to the family but
        were not evaluated because their p-value is 1 by construction
        (e.g. zero overlap).  They are added to the family size and then
        dropped from the output.

    Returns
    -------
    np.ndarray of adjusted p-values, same length/order as `pvals`.
    """
    method = method or get_method()
    if method not in _METHOD_LABELS:
        raise ValueError(f"Unsupported method '{method}'")

    p = np.asarray(pvals, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return out

    family = p[valid]
    if n_untested and n_untested > 0:
        family = np.concatenate([family, np.ones(int(n_untested))])

    adjusted = multipletests(family, method=method)[1][:n_valid]
    out[valid] = adjusted
    return out