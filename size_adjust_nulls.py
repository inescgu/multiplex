#!/usr/bin/env python3
"""
size_adjust_nulls.py -- compare the true module with the null modules AFTER accounting for module size.

Why: the 'seeds' and 'rewired' nulls build modules whose size varies (random seeds need more or fewer
bridging genes to connect), and most metrics depend on size: drug-target fractions dilute as a module
grows, edge density falls, conductance shifts. A raw z-score against a null of a different size
mixes the effect you care about with a size effect.

Reads (from --validation-dir, written by step10_validate_module.py):
    null_<tag>_partial*.csv          one row per finished permutation (needs diag_size)
    true_module_metrics.csv          the true module's metrics

For every metric it reports, with z oriented so that POSITIVE = module better than the null
(conductance is lower-is-better, so it is sign-flipped):
    z_raw            the usual z against all nulls
    size_r2          share of the null variation in the metric explained by log(module size)
    z_size_adj       true value vs the fit (metric ~ log size) evaluated at the true size, in units of
                     the fit's residual SD
    p_size_adj       empirical: fraction of null residuals >= the true module's residual
    n_window, z_window, p_window    same z/p using only nulls within +-window of the true size
Writes null_<tag>_size_adjusted.csv.

Caveats: linear-in-log-size is an assumption; the fit is only trustworthy if the true size lies
inside the range of null sizes (a warning is printed if not); with few permutations the fit is noisy.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"conductance"}


def load_partial(d, tag):
    fs = sorted(Path(d).glob(f"null_{tag}_partial*.csv"))
    if not fs:
        raise SystemExit(f"ERROR: no null_{tag}_partial*.csv in {d}")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df.drop_duplicates("perm_i").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--null", required=True, choices=["seeds", "rewired"])
    ap.add_argument("--true-size", type=int, default=9940)
    ap.add_argument("--window", type=float, default=0.25, help="+- fraction of true size for the window comparison")
    ap.add_argument("--min-n", type=int, default=8, help="min permutations needed for the fit")
    args = ap.parse_args()

    d = Path(args.validation_dir)
    df = load_partial(d, args.null)
    true = pd.read_csv(d / "true_module_metrics.csv", index_col=0)["value"]
    size = df["diag_size"].to_numpy(float)
    x, xt = np.log(size), np.log(args.true_size)
    print(f"{args.null}: {len(df)} permutations | null size min {size.min():.0f}, median {np.median(size):.0f}, "
          f"max {size.max():.0f} | true {args.true_size}")
    if not (x.min() <= xt <= x.max()):
        print("WARNING: true size is outside the range of null sizes; the size fit is an extrapolation.")
    win = np.abs(size - args.true_size) <= args.window * args.true_size
    print(f"nulls within +-{args.window:.0%} of the true size: {int(win.sum())}\n")

    rows = []
    for m in df.columns:
        if m.startswith("diag_") or m == "perm_i" or m not in true.index or pd.isna(true[m]):
            continue
        y = df[m].to_numpy(float)
        ok = ~np.isnan(y)
        if ok.sum() < args.min_n:
            continue
        sign = -1.0 if m.split("__")[0] in LOWER_IS_BETTER else 1.0
        yo, xx, tv = sign * y[ok], x[ok], sign * float(true[m])
        n = len(yo)
        sd = yo.std(ddof=1)
        z_raw = (tv - yo.mean()) / sd if sd > 0 else np.nan
        p_raw = (1 + (yo >= tv).sum()) / (1 + n)
        b, a = np.polyfit(xx, yo, 1)
        res = yo - (a + b * xx)
        rsd = res.std(ddof=2)
        pred = a + b * xt
        z_adj = (tv - pred) / rsd if rsd > 0 else np.nan
        p_adj = (1 + (res >= tv - pred).sum()) / (1 + n)
        r2 = 1 - res.var() / yo.var() if yo.var() > 0 else np.nan
        w = win[ok]
        if w.sum() >= 5:
            yw = yo[w]
            z_w = (tv - yw.mean()) / yw.std(ddof=1) if yw.std(ddof=1) > 0 else np.nan
            p_w = (1 + (yw >= tv).sum()) / (1 + len(yw))
        else:
            z_w = p_w = np.nan
        rows.append(dict(metric=m, true_value=float(true[m]), null_mean=sign * yo.mean(), n_null=n,
                         z_raw=z_raw, p_raw=p_raw, size_r2=r2, z_size_adj=z_adj, p_size_adj=p_adj,
                         n_window=int(w.sum()), z_window=z_w, p_window=p_w))
    out = pd.DataFrame(rows)
    out.to_csv(d / f"null_{args.null}_size_adjusted.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nz > 0 = module better than null. If z_raw is large but z_size_adj ~ 0 (and size_r2 is high), "
          "the raw difference was a size effect.")


if __name__ == "__main__":
    main()