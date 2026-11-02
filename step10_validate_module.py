#!/usr/bin/env python
"""
step10_validate_module.py  --  is the disease module better than chance?

Replaces step10_permutation_test_multiplex.py, step10b_functional_permutation_
test_multiplex.py and step10c_functional_permutation_multixrank_null.py.
Needs step6_build_disease_module.py in the same directory.

Every null module is built the SAME way as the real one: MultiXrank on the
heterogeneous network (your config.yml) -> ranked pool -> step6's bridging +
LCC fallback. The nulls differ only in what gets randomised:

  seeds      Replace every seed with a random node from the same cell-type
             layer and similar within-layer PPI degree.
             Asks: are MY seeds special?
  rewired    Keep your seeds. Shuffle the PPI edges inside each cell-type
             layer (every node keeps its degree; cell-type-to-cell-type links
             are untouched), then run MultiXrank on the shuffled network.
             Asks: does the real wiring matter?
  neighbors  Keep your seeds. Swap each neighbor for a random node from the
             same layer with similar PPI degree. No walk at all.
             Asks: do walk-selected neighbors beat random genes of the same kind?

Metrics: functional (drug-target fractions, pain phenotypes, optional GO) are
always scored at GENE level (a gene in three cell types counts once).
--topology adds edge density / mean internal degree / conductance for the
'seeds' and 'rewired' nulls (meaningless for random neighbors).

Seed-gene scores are reported only for the 'seeds' null. In the other two the
seeds are fixed, so their seed score never changes and says nothing.

p-value = (1 + #null as extreme or more) / (1 + #valid null permutations).

CHANGES IN THIS VERSION (fixes a silent parameter mismatch)
-----------------------------------------------------------
The 'seeds' and 'rewired' nulls rebuild a module with step6's own build_module.
If --k / --max-bridge / --exclude-seeds differ from the step6 run that made the
true module, the null modules are a different kind of object (an earlier run used
k=300, max-bridge=2000 and no excluded seeds against a true module built with
k=200, max-bridge=30000 and 11 excluded seeds: the nulls never connected, were
trimmed to ~3.1k nodes, and were compared with a 9,940-node module).
  1. --k / --max-bridge defaults now match the step6 run (200 / 30000).
  2. Excluded seeds are read from the true module's is_excluded_seed column when
     --exclude-seeds is not given, and a disagreement is flagged.
  3. IDENTITY CHECK (on by default): before any null runs, the TRUE seeds are run
     through the same MultiXrank + build_module path on the real network. It must
     reproduce the true module (size and Jaccard). If it does not, the script stops
     before spending hours on nulls. --skip-identity-check turns it off.
  4. diag_mean_ldeg: mean within-layer degree of the non-anchor genes, reported
     for every null and for the true module, so the degree matching of the
     'neighbors' null can be checked rather than assumed.

Assumptions to verify: config.yml has a top-level `seed:` path (relative to
wdir); layer files are unweighted whitespace-separated edge lists with no
header; random_walk_rank() returns a DataFrame with multiplex/node/score.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys, argparse, copy, io, re, shutil, time, contextlib, warnings, multiprocessing
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import multixrank
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6_build_disease_module as s6

warnings.filterwarnings("ignore")

# Pain EFO list (extracted May 13, 2026 by searching 'pain' at https://www.ebi.ac.uk/ols4/ontologies/efo)
DEFAULT_PAIN_EFOS = [
    'EFO_0010639','EFO_0801084','EFO_0005762','EFO_0009625','EFO_0007889',
    'EFO_0010099','EFO_0010100','EFO_0022195','EFO_0022861','EFO_0022862',
    'EFO_0022863','EFO_0022864','EFO_0022865','EFO_0022837','EFO_0008330',
    'EFO_0008316','EFO_0022960','EFO_0008338','EFO_0008340','EFO_0008339',
    'EFO_0010640','EFO_0007809','EFO_0010072','EFO_1001412','EFO_0011014',
    'EFO_0005404','EFO_0022850','EFO_0010263','EFO_1001833','EFO_0004149',
    'EFO_0920015','EFO_0009515','EFO_1001370','EFO_0009615','EFO_1001289',
    'EFO_1001776','EFO_1001258','EFO_1001785','EFO_1001404','EFO_1001366',
    'EFO_1002027','EFO_1001845','EFO_1001756','EFO_1001909','EFO_1000869',
    'EFO_1000254','EFO_1000542','EFO_0007299','EFO_1000446','EFO_0008544',
    'EFO_0020919','EFO_1001367','EFO_1001453','EFO_0008998','EFO_0003901',
    'EFO_0009748','EFO_0022849','EFO_1000621','EFO_0008622','EFO_0022851',
    'EFO_0021536','EFO_1000381','EFO_0002678','EFO_1001888','EFO_0003913',
    'EFO_0007520','EFO_1000243','EFO_0003266','EFO_0004278','EFO_0007638',
    'EFO_0700072','EFO_0010582','EFO_1001301','EFO_1000861',
]
LOWER_IS_BETTER = {"conductance"}
_S = {}  # per-worker state, set once by _init


# ── functional scorers (all take a list of bare gene symbols) ──────────────
@lru_cache(maxsize=None)
def _known_drug(ot_dir):
    import pyarrow.parquet as pq
    return pq.read_table(os.path.join(ot_dir, "known_drug")).to_pandas()


class DrugTargetScorer:
    """Fraction of genes that are drug targets (any indication, or only for `efos`)."""
    GENE_COLS = ['targetSymbol', 'target_symbol', 'approved_symbol', 'approvedSymbol',
                 'target_gene_symbol', 'gene_symbol']
    EFO_COLS = ['diseaseId', 'disease_id', 'indication_id', 'efo_id', 'diseaseEFO', 'disease_efo']

    def __init__(self, ot_dir, efos=None):
        self.ot_dir = ot_dir
        self.efos = [e.replace(":", "_") for e in efos] if efos else None
        self.targets = None

    def load(self):
        if self.targets is not None:
            return
        df = _known_drug(self.ot_dir)
        gene_col = next((c for c in self.GENE_COLS if c in df.columns), None)
        if gene_col is None:
            raise ValueError(f"No gene column in known_drug. Columns: {list(df.columns)}")
        if self.efos:
            efo_col = next((c for c in self.EFO_COLS if c in df.columns), None)
            if efo_col is None:
                raise ValueError(f"No EFO column in known_drug. Columns: {list(df.columns)}")
            pat = "|".join(re.escape(e) + "|" + re.escape(e.replace("_", ":")) for e in self.efos)
            df = df[df[efo_col].astype(str).str.contains(pat, case=False, na=False)]
        self.targets = set(df[gene_col].dropna().str.upper())

    def score(self, genes):
        self.load()
        return sum(g.upper() in self.targets for g in genes) / len(genes) if genes else np.nan

    def hits(self, genes):
        self.load()
        return [g for g in genes if g.upper() in self.targets]


class GOScorer:
    """Best GO:BP enrichment, -log10(p), via gProfiler (network call; use sparingly)."""
    def score(self, genes):
        if len(genes) < 5:
            return np.nan
        from gprofiler import GProfiler
        try:
            res = GProfiler(return_dataframe=True).profile(
                organism="hsapiens", query=list(genes), sources=["GO:BP"], no_evidences=True)
            return 0.0 if res.empty else float(-np.log10(res["p_value"].min() + 1e-300))
        except Exception:
            return np.nan


class PhenotypeScorer:
    """Fraction of genes with a pain-related phenotype in a gene/phenotype CSV."""
    KEYWORDS = ['pain', 'nociception', 'analgesia', 'hyperalgesia', 'allodynia',
                'neuropathy', 'neuralgia']

    def __init__(self, csv_path):
        self.csv_path, self.genes = csv_path, None

    def load(self):
        if self.genes is not None:
            return
        df = pd.read_csv(self.csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        gene_col = next((c for c in ['Loci', 'Gene', 'GeneSymbol'] if c in df.columns), None)
        phen_col = next((c for c in ['Phenotype', 'Trait', 'Condition'] if c in df.columns), None)
        if not gene_col or not phen_col:
            raise ValueError(f"No gene/phenotype columns in {self.csv_path}: {list(df.columns)}")
        df = df.dropna(subset=[gene_col, phen_col])
        is_pain = df[phen_col].astype(str).str.lower().apply(
            lambda p: any(k in p for k in self.KEYWORDS))
        self.genes = set(df.loc[is_pain, gene_col].astype(str).str.strip().str.upper())

    def score(self, genes):
        self.load()
        return sum(g.upper() in self.genes for g in genes) / len(genes) if genes else np.nan


# ── metrics ────────────────────────────────────────────────────────────────
def topology_metrics(G, module):
    """Edge density, mean internal degree, conductance of `module` inside G."""
    mset = {n for n in module if n in G}
    sg = G.subgraph(mset)
    n, e = sg.number_of_nodes(), sg.number_of_edges()
    cut = sum(1 for u in mset for v in G.neighbors(u) if v not in mset)
    vol = sum(G.degree(u) for u in mset)
    denom = min(vol, 2 * G.number_of_edges() - vol)
    return {"edge_density": e / (n * (n - 1) / 2) if n > 1 else 0.0,
            "mean_internal_degree": 2 * e / n if n else 0.0,
            "conductance": cut / denom if denom > 0 else 0.0}


def score_module(module, anchors, G, S, split_seeds):
    """Functional (gene-level) + optional topology metrics. A gene is a 'seed
    gene' if ANY of its cell-type copies is an anchor, otherwise a 'neighbor'."""
    genes = sorted({s6._gene_of(n) for n in module})
    seed_genes = {s6._gene_of(a) for a in anchors}
    seed_part = [g for g in genes if g in seed_genes]
    neigh_part = [g for g in genes if g not in seed_genes]
    out = {}
    for name, sc in S["scorers"].items():
        out[name] = sc.score(genes)
        if split_seeds:
            out[f"{name}__seeds"] = sc.score(seed_part) if seed_part else np.nan
        out[f"{name}__neighbors"] = sc.score(neigh_part) if neigh_part else np.nan
    if G is not None and S["topology"]:
        out.update(topology_metrics(G, module))
    return out


def mean_neighbor_ldeg(module, anchors, ldeg):
    """Mean within-layer PPI degree of the non-anchor nodes (degree-matching check)."""
    a = set(anchors)
    d = [ldeg[n] for n in module if n not in a and n in ldeg]
    return float(np.mean(d)) if d else np.nan


def finish(module, anchors, G, S, n_bridges=None, connected=None, split_seeds=False):
    mset = set(module)
    anchors_in = [a for a in anchors if a in mset]
    m = score_module(module, anchors_in, G, S, split_seeds)
    t = S["true_set"]
    m["diag_size"] = len(module)
    m["diag_jaccard_vs_true"] = len(mset & t) / len(mset | t)
    m["diag_mean_ldeg"] = mean_neighbor_ldeg(module, anchors_in, S["ldeg"])
    if n_bridges is not None:
        m["diag_n_bridges"] = n_bridges
        m["diag_connected_before_fallback"] = int(connected)
    return m


# ── matched sampling (used by BOTH the 'seeds' and 'neighbors' nulls) ──────
def layer_degrees(H):
    """Within-layer PPI degree: count only edges whose two endpoints are in the
    SAME cell-type layer (cross-cell-type identity edges don't count)."""
    deg = defaultdict(int)
    for u, v in H.edges():
        if s6._celltype_of(u) == s6._celltype_of(v):
            deg[u] += 1
            deg[v] += 1
    return {n: deg.get(n, 0) for n in H.nodes()}


def build_layer_bins(H, ldeg, bin_size):
    bins = defaultdict(list)
    for n in H.nodes():
        bins[(s6._celltype_of(n), ldeg[n] // bin_size)].append(n)
    return bins


def sample_matched(nodes, ldeg, bins, rng, bin_size, exclude=()):
    """node -> random node from the same cell-type layer with similar PPI
    degree, never one in `exclude`, never reused."""
    used, mapping = set(exclude), {}
    for n in nodes:
        if n not in ldeg:
            continue
        ct, b = s6._celltype_of(n), ldeg[n] // bin_size
        cand = []
        for off in (0, 1, -1, 2, -2, 3, -3):
            cand = [x for x in bins.get((ct, b + off), ()) if x not in used]
            if cand:
                break
        if cand:
            pick = cand[rng.integers(len(cand))]
            mapping[n] = pick
            used.add(pick)
    return mapping


# ── MultiXrank plumbing ────────────────────────────────────────────────────
@contextlib.contextmanager
def scratch(wdir, tag):
    d = wdir / "null_runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_multixrank(wdir, cfg, seeds, d, layer_paths=None):
    """Run MultiXrank with a temp seeds file (and optionally replacement
    layer files: {(multiplex, layer_idx): path relative to wdir})."""
    (d / "seeds.txt").write_text("\n".join(seeds) + "\n")
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = str((d / "seeds.txt").relative_to(wdir))
    for (name, i), rel in (layer_paths or {}).items():
        layers = cfg["multiplex"][name]["layers"]
        layers = [layers] if isinstance(layers, str) else list(layers)
        layers[i] = rel
        cfg["multiplex"][name]["layers"] = layers
    (d / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return multixrank.Multixrank(config=str(d / "config.yml"), wdir=str(wdir)).random_walk_rank()


def ranked_pool_from_result(res, excluded_ct_upper):
    """Same pool step6 builds from per_celltype_scores.csv: '<multiplex>::GENE'
    with score > 0, sorted descending."""
    if isinstance(res, list):
        raw = pd.concat(res, ignore_index=True)
    elif isinstance(res, dict):
        raw = pd.concat([v for v in res.values() if isinstance(v, pd.DataFrame)], ignore_index=True)
    else:
        raw = res.copy()
    raw.columns = [str(c).lower().strip() for c in raw.columns]
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce").fillna(0.0)
    raw = raw[raw["score"] > 0]
    if excluded_ct_upper:
        raw = raw[~raw["multiplex"].astype(str).str.strip().str.upper().isin(excluded_ct_upper)]
    gene = raw["node"].astype(str).str.strip().str.rsplit("::", n=1).str[-1].str.upper()
    raw = raw.assign(nn=raw["multiplex"].astype(str) + "::" + gene)
    return raw.sort_values("score", ascending=False).drop_duplicates("nn")["nn"].tolist()


def build_null_module(res, anchors, G, S, exclude_set):
    """step6's own bridging + LCC fallback, on graph G."""
    pool = ranked_pool_from_result(res, S["excluded_ct_upper"])
    with contextlib.redirect_stdout(io.StringIO()):
        module, n_bridges, connected, *_ = s6.build_module(
            G, anchors, pool, S["k"], S["max_bridge"], exclude_set=exclude_set)
        if not connected and not S["no_lcc_fallback"]:
            module, _ = s6.extract_lcc(module, G)
    return module, n_bridges, connected


def load_layers(wdir, cfg):
    layers = {}
    for name, spec in cfg["multiplex"].items():
        rels = [spec["layers"]] if isinstance(spec["layers"], str) else spec["layers"]
        for i, rel in enumerate(rels):
            try:
                df = pd.read_csv(wdir / rel, sep="\t", header=None, dtype=str,
                                 keep_default_na=False)
            except pd.errors.ParserError as e:
                raise ValueError(f"Cannot parse layer file {wdir / rel} as tab-separated: {e}")
            if df.shape[1] > 2:
                raise NotImplementedError(
                    f"{rel} has a weight column; the rewired null only supports unweighted layers.")
            layers[(name, i)] = df
    return layers


def rewire_layer(df, rng, swaps_per_edge):
    """Degree-preserving shuffle of ONE layer's edges."""
    g = nx.Graph()
    g.add_edges_from(zip(df[0], df[1]))
    nswap = int(swaps_per_edge * g.number_of_edges())
    nx.double_edge_swap(g, nswap=nswap, max_tries=100 * nswap, seed=int(rng.integers(2**31 - 1)))
    return list(g.edges())


# ── the nulls (each returns one dict of metrics, or None) ──────────────────
def _init(state):
    _S.update(state)


def null_seeds(i, seed_int):
    S, rng = _S, np.random.default_rng(seed_int)
    mapping = sample_matched(S["run_seeds"], S["ldeg"], S["bins"], rng, S["bin_size"],
                             exclude=S["run_seeds"])
    if len(mapping) < max(1, len(S["run_seeds"]) // 2):
        return None
    anchors = [mapping[s] for s in S["anchors"] if s in mapping]
    if not anchors:
        return None
    excluded = {mapping[s] for s in S["explicit_excluded"] if s in mapping}
    with scratch(S["wdir"], f"seeds_{i}") as d:
        res = run_multixrank(S["wdir"], S["cfg"], list(mapping.values()), d)
    module, n_br, conn = build_null_module(res, anchors, S["H"], S, excluded)
    return finish(module, anchors, S["H"], S, n_br, conn, split_seeds=True)


def null_rewired(i, seed_int):
    S, rng = _S, np.random.default_rng(seed_int)
    with scratch(S["wdir"], f"rewired_{i}") as d:
        H_rw, paths = nx.Graph(), {}
        H_rw.add_edges_from(S["bip_edges"])  # cross-cell-type links: untouched
        for (name, li), df in S["layers"].items():
            edges = rewire_layer(df, rng, S["swaps_per_edge"])
            p = d / f"{name}_{li}.tsv"
            pd.DataFrame(edges).to_csv(p, sep="\t", header=False, index=False)
            paths[(name, li)] = str(p.relative_to(S["wdir"]))
            H_rw.add_edges_from((s6._norm_node(u), s6._norm_node(v), {"cell_types": {str(name)}})
                                for u, v in edges)
        res = run_multixrank(S["wdir"], S["cfg"], S["run_seeds"], d, paths)
    with contextlib.redirect_stdout(io.StringIO()):
        H_rw = s6.filter_out_celltypes(H_rw, S["excl_ct"])
        if S["lcc_per_layer"]:
            H_rw = s6.filter_to_lcc_per_layer(H_rw)
    module, n_br, conn = build_null_module(res, S["anchors"], H_rw, S, S["explicit_excluded"])
    return finish(module, S["anchors"], H_rw, S, n_br, conn, split_seeds=False)


def null_neighbors(i, seed_int):
    S, rng = _S, np.random.default_rng(seed_int)
    need = S["true_neighbors"]
    mapping = sample_matched(need, S["ldeg"], S["bins"], rng, S["bin_size"],
                             exclude=S["anchors_in_module"])
    if len(mapping) < max(1, len(need) // 2):
        return None
    module = list(S["anchors_in_module"]) + list(mapping.values())
    return finish(module, S["anchors_in_module"], None, S, split_seeds=False)


def identity_check(args, out):
    """Run the TRUE seeds through the same MultiXrank + build_module path as the
    nulls, on the real network. If this does not reproduce the true module, the
    step 10 parameters do not match the step 6 run and the nulls are not
    comparable. Returns True if it reproduces."""
    S = _S
    print("\n[identity] true seeds -> MultiXrank -> build_module on the real network ...")
    t0 = time.time()
    with scratch(S["wdir"], "identity") as d:
        res = run_multixrank(S["wdir"], S["cfg"], S["run_seeds"], d)
    module, n_br, conn = build_null_module(res, S["anchors"], S["H"], S, S["explicit_excluded"])
    m = finish(module, S["anchors"], S["H"], S, n_br, conn, split_seeds=True)
    size, true_size = m["diag_size"], len(S["true_set"])
    jac = m["diag_jaccard_vs_true"]
    pd.DataFrame([m]).to_csv(out / "identity_check.csv", index=False)
    print(f"[identity] {time.time() - t0:.0f}s | size {size} vs true {true_size} "
          f"| Jaccard {jac:.3f} | bridges {n_br} | connected before fallback: {bool(conn)}")
    ok = jac >= args.identity_min_jaccard and abs(size - true_size) <= 0.1 * true_size
    if ok:
        print("[identity] OK: step 10 reproduces the true module; nulls are comparable.")
    else:
        print(f"[identity] FAILED (need Jaccard >= {args.identity_min_jaccard} and size within 10%).\n"
              "  The step 10 call does not rebuild the module step 6 built. Check, against the step 6 command:\n"
              "    --k  --max-bridge  --exclude-seeds  --lcc-per-layer  --exclude-ppi-seeds\n"
              "    --connectivity  --no-lcc-fallback  --exclude-celltypes  --exclude-ppi-layer\n"
              "  and that step 6 used the same scores/per-celltype scores as the MultiXrank run here.")
    return ok


# ── running + reporting ────────────────────────────────────────────────────
NULL_ID = {"seeds": 1, "rewired": 2, "neighbors": 3}


def _timed(task, i, seed_int):
    """Run one permutation; tag the row with its index and wall time."""
    t0 = time.time()
    r = task(i, seed_int)
    if r is not None:
        r["perm_i"] = i
        r["diag_seconds"] = round(time.time() - t0, 1)
    return r


def load_partial(out, tag):
    """All finished permutations for a null, across every null_<tag>_partial*.csv."""
    fs = sorted(out.glob(f"null_{tag}_partial*.csv"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df.drop_duplicates("perm_i").sort_values("perm_i").reset_index(drop=True)


def run_null(ex, task, n, tag, base_seed, out, perm_range=None, resume=False):
    """Run permutations of one null. Every finished permutation is appended to
    null_<tag>_partial[_lo-hi].csv immediately, so a timeout loses nothing.
    Permutation i always uses the same random seed (a function of --seed, the null
    and i), so --resume and --perm-range chunks are reproducible and disjoint."""
    seeds = np.random.default_rng([base_seed, NULL_ID[tag]]).integers(0, 2**31 - 1, size=n)
    lo, hi = (perm_range if perm_range else (0, n))
    hi = min(hi, n)
    todo = list(range(lo, hi))
    pf = out / (f"null_{tag}_partial.csv" if not perm_range else f"null_{tag}_partial_{lo}-{hi}.csv")
    if pf.exists() and not resume:
        raise SystemExit(f"ERROR: {pf} already exists. Use --resume to continue it, or delete it to start over.")
    if resume:
        prev = load_partial(out, tag)
        done = set(prev["perm_i"].astype(int)) if len(prev) else set()
        todo = [i for i in todo if i not in done]
        print(f"  [{tag}] resume: {len(done)} already done, {len(todo)} to run in range {lo}-{hi}")
    futs = {ex.submit(_timed, task, i, int(seeds[i])): i for i in todo}
    failed, ndone, t0 = 0, 0, time.time()
    for f in as_completed(futs):
        try:
            r = f.result()
        except Exception as e:
            print(f"  [{tag}] permutation {futs[f]} error: {type(e).__name__}: {e}", flush=True)
            r = None
        ndone += 1
        if r is None:
            failed += 1
        else:
            pd.DataFrame([r]).to_csv(pf, mode="a", header=not pf.exists(), index=False)
        el = time.time() - t0
        last = f"{r['diag_seconds']:.0f}s" if r is not None else "skipped"
        if len(todo) <= 200 or ndone % max(1, len(todo) // 20) == 0 or ndone == len(todo):
            print(f"  [{tag}] {ndone}/{len(todo)} | {el:.0f}s | ETA {el / ndone * (len(todo) - ndone):.0f}s "
                  f"| last perm {last}", flush=True)
    print(f"  [{tag}] this run: {ndone - failed} valid, {failed} failed/skipped", flush=True)
    return load_partial(out, tag)


def summarise(true_m, null_df):
    rows = []
    for k, tv in true_m.items():
        if k.startswith("diag_") or k not in null_df.columns or pd.isna(tv):
            continue
        v = null_df[k].dropna().to_numpy()
        if len(v) == 0:
            continue
        lower = k.split("__")[0] in LOWER_IS_BETTER
        extreme = int((v <= tv).sum() if lower else (v >= tv).sum())
        sd = v.std()
        rows.append({"metric": k, "true_value": tv, "null_mean": v.mean(), "null_sd": sd,
                     "z_score": (tv - v.mean()) / sd if sd > 0 else np.nan,
                     "p_value": (1 + extreme) / (1 + len(v)), "n_null": len(v)})
    return pd.DataFrame(rows)


def plot_null(summary, null_df, title, path):
    n = len(summary)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), summary.itertuples()):
        ax.axis("on")
        ax.hist(null_df[r.metric].dropna(), bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(r.true_value, color="crimson", linewidth=2.2,
                   label=f"true (p={r.p_value:.3f}, z={r.z_score:.2f})")
        ax.set_title(r.metric.replace("__", " / ").replace("_", " "), fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def report(tag, title, true_m, null_df, out):
    summary = summarise(true_m, null_df)
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if "diag_size" in null_df:
        print(f"\n  null module size: mean {null_df['diag_size'].mean():.0f} "
              f"(min {null_df['diag_size'].min()}, max {null_df['diag_size'].max()}) | "
              f"true {len(_S['true_set'])} | mean Jaccard vs true module: "
              f"{null_df['diag_jaccard_vs_true'].mean():.3f}")
    if "diag_mean_ldeg" in null_df:
        print(f"  mean within-layer degree of non-anchor genes: null {null_df['diag_mean_ldeg'].mean():.2f} "
              f"(sd {null_df['diag_mean_ldeg'].std():.2f}) | true {_S['true_neighbor_ldeg']:.2f}")
    if "diag_connected_before_fallback" in null_df:
        print(f"  connected before LCC fallback: {null_df['diag_connected_before_fallback'].mean():.0%}")
    if "diag_n_bridges" in null_df:
        print(f"  bridging nodes added: mean {null_df['diag_n_bridges'].mean():.0f} "
              f"(max {null_df['diag_n_bridges'].max()}; --max-bridge is {_S['max_bridge']})")
    null_df.to_csv(out / f"null_{tag}_permutations.csv", index=False)
    summary.to_csv(out / f"null_{tag}_summary.csv", index=False)
    plot_null(summary, null_df, title, out / f"null_{tag}.png")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(line_buffering=True)   # slurm logs are files: do not lose progress on a timeout
    ap = argparse.ArgumentParser(description="Validate a disease module against MultiXrank-built nulls")
    ap.add_argument("--wdir", required=True, help="MultiXrank wdir (config, seeds/, reference graphs)")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--scores", required=True, help="gene_rwr_scores.csv (step4)")
    ap.add_argument("--disease-module-csv", required=True, help="step6's disease_module.csv (true module)")
    ap.add_argument("--output-dir", required=True)
    # mirror the step6 run that built the true module (defaults = the current step6 run)
    ap.add_argument("--connectivity", choices=["all", "ppi"], default="all")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--max-bridge", type=int, default=30000)
    ap.add_argument("--no-lcc-fallback", action="store_true")
    ap.add_argument("--lcc-per-layer", action="store_true")
    ap.add_argument("--exclude-seeds", nargs="*", default=[],
                    help="same list as step6; if omitted, read from is_excluded_seed in --disease-module-csv")
    ap.add_argument("--exclude-celltypes", nargs="+", default=[])
    ap.add_argument("--exclude-ppi-layer", action="store_true")
    ap.add_argument("--exclude-seed-celltypes", nargs="+", default=[])
    ap.add_argument("--exclude-ppi-seeds", action="store_true")
    # identity check
    ap.add_argument("--skip-identity-check", action="store_true",
                    help="do NOT first verify that step 10 reproduces the true module (not recommended)")
    ap.add_argument("--identity-min-jaccard", type=float, default=0.9)
    ap.add_argument("--identity-only", action="store_true", help="run only the identity check, then exit")
    # run control (long nulls: checkpoint, resume, split across jobs)
    ap.add_argument("--resume", action="store_true",
                    help="continue from null_<tag>_partial*.csv, running only missing permutations")
    ap.add_argument("--perm-range", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="run only permutations LO..HI-1 of each null (split one null across jobs; "
                         "writes its own partial file; implies identity check is reused if present)")
    ap.add_argument("--from-partial", action="store_true",
                    help="run nothing: build summaries/plots from whatever partial files exist")
    # nulls
    ap.add_argument("--nulls", nargs="+", default=["seeds", "rewired", "neighbors"],
                    choices=["seeds", "rewired", "neighbors"])
    ap.add_argument("--n-perms-seeds", type=int, default=100, help="each needs a MultiXrank run (slow)")
    ap.add_argument("--n-perms-rewired", type=int, default=100, help="MultiXrank run + rewiring (slowest)")
    ap.add_argument("--n-perms-neighbors", type=int, default=1000, help="no MultiXrank (fast)")
    ap.add_argument("--swaps-per-edge", type=float, default=10)
    ap.add_argument("--bin-size", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # metrics
    ap.add_argument("--ot-data", default=None, help="OpenTargets dir containing known_drug/")
    ap.add_argument("--disease-efos", default=None, help="comma-separated; default = pain EFO list")
    ap.add_argument("--phenotype-csv", default=None)
    ap.add_argument("--go", action="store_true", help="add GO:BP enrichment (gProfiler network calls)")
    ap.add_argument("--topology", action="store_true")
    args = ap.parse_args()

    n_workers = args.n_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    wdir = Path(args.wdir).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config) if Path(args.config).is_absolute() else wdir / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    excl_ct = list(args.exclude_celltypes)
    if args.exclude_ppi_layer and "global_ppi" not in {c.lower() for c in excl_ct}:
        excl_ct.append("global_ppi")
    excl_seed_ct = list(args.exclude_seed_celltypes)
    if args.exclude_ppi_seeds and "global_ppi" not in {c.lower() for c in excl_seed_ct}:
        excl_seed_ct.append("global_ppi")

    # true module first: it also tells us which seeds step6 excluded
    mdf = pd.read_csv(args.disease_module_csv)
    mdf["node"] = mdf["node"].astype(str).str.strip()
    def _truthy(col):
        return mdf[col].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    csv_excluded = ({s6._norm_node(n) for n in mdf.loc[_truthy("is_excluded_seed"), "node"]}
                    if "is_excluded_seed" in mdf.columns else set())
    explicit_excluded = {s6._norm_node(g) for g in args.exclude_seeds}
    if not explicit_excluded and csv_excluded:
        print(f"[exclude-seeds] --exclude-seeds not given; using the {len(csv_excluded)} flagged "
              f"is_excluded_seed in {Path(args.disease_module_csv).name}")
        explicit_excluded = csv_excluded
    elif csv_excluded and csv_excluded != explicit_excluded:
        print(f"[exclude-seeds] WARNING: --exclude-seeds ({len(explicit_excluded)}) differs from "
              f"is_excluded_seed in the module csv ({len(csv_excluded)}). Using --exclude-seeds.")

    # graph + seeds: step6's own loader, so nothing can drift from the real run
    ref = wdir / ("gene_reference_graph.pkl" if args.connectivity == "all" else "gene_ppi_reference_graph.pkl")
    seeds_path = wdir / "seeds" / "seeds.txt"
    _, H_pre, anchor_seeds, _, _, _ = s6.load_inputs(
        args.scores, ref, seeds_path, per_ct_csv=None, exclude_seeds=sorted(explicit_excluded),
        exclude_celltypes=excl_ct, exclude_seed_celltypes=excl_seed_ct)
    H = s6.filter_to_lcc_per_layer(H_pre) if args.lcc_per_layer else H_pre
    all_seeds = [s6._norm_node(l) for l in open(seeds_path) if l.strip()]
    run_seeds = [s for s in all_seeds if s in H]      # everything MultiXrank was seeded with
    anchors = [s for s in anchor_seeds if s in H]     # what anchored the module

    # true module
    mask = (mdf["is_seed"] & ~_truthy("is_excluded_seed")
            if "is_excluded_seed" in mdf.columns else mdf["is_seed"])
    true_module = mdf["node"].tolist()
    anchors_in_module = mdf.loc[mask, "node"].tolist()
    anchor_set = set(anchors_in_module)
    true_neighbors = [n for n in true_module if n not in anchor_set]
    print(f"\nTrue module: {len(true_module)} nodes = {len(anchors_in_module)} anchors "
          f"+ {len(true_neighbors)} neighbors | seeds run through MultiXrank: {len(run_seeds)} "
          f"| anchors rebuilt here: {len(anchors)}")
    if len(anchors) != len(anchors_in_module):
        print(f"  WARNING: {len(anchors)} anchors rebuilt here vs {len(anchors_in_module)} in the true module. "
              "The exclude/LCC flags differ from step6; nulls will not be comparable.")

    # scorers (loaded here, before forking, so workers inherit them)
    scorers = {}
    if args.ot_data:
        efos = ([e.strip() for e in args.disease_efos.split(",") if e.strip()]
                if args.disease_efos else DEFAULT_PAIN_EFOS)
        scorers["drug_target_any"] = DrugTargetScorer(args.ot_data)
        scorers["drug_target_disease"] = DrugTargetScorer(args.ot_data, efos)
        pd.DataFrame({"efo_id": efos}).to_csv(out / "disease_efos_used.csv", index=False)
    if args.phenotype_csv:
        scorers["pain_phenotype"] = PhenotypeScorer(args.phenotype_csv)
    if args.go:
        scorers["go_bp_neglog10p"] = GOScorer()
    if not scorers:
        print("ERROR: no functional scorers (pass --ot-data, --phenotype-csv and/or --go)")
        return 1
    for sc in scorers.values():
        if hasattr(sc, "load"):
            sc.load()

    ldeg = layer_degrees(H)
    S = dict(
        H=H, ldeg=ldeg, bins=build_layer_bins(H, ldeg, args.bin_size), bin_size=args.bin_size,
        wdir=wdir, cfg=cfg, run_seeds=run_seeds, anchors=anchors,
        explicit_excluded=explicit_excluded, excl_ct=excl_ct,
        excluded_ct_upper={c.strip().upper() for c in excl_ct},
        k=args.k, max_bridge=args.max_bridge, no_lcc_fallback=args.no_lcc_fallback,
        lcc_per_layer=args.lcc_per_layer, scorers=scorers, topology=args.topology,
        true_set=set(true_module), anchors_in_module=anchors_in_module, true_neighbors=true_neighbors,
        true_neighbor_ldeg=mean_neighbor_ldeg(true_module, anchors_in_module, ldeg),
        swaps_per_edge=args.swaps_per_edge,
        layers=load_layers(wdir, cfg) if "rewired" in args.nulls else {},
        bip_edges=[(u, v) for u, v in H_pre.edges()
                   if s6._celltype_of(u) != s6._celltype_of(v)],
    )
    _init(S)   # main process too, for the identity check and report()
    print(f"Cross-cell-type (bipartite) edges kept fixed in the rewired null: {len(S['bip_edges']):,} "
          f"| PPI layers to rewire: {len(S['layers'])}")
    print(f"Step 6 parameters used for every rebuilt module: k={args.k}, max-bridge={args.max_bridge}, "
          f"lcc-per-layer={args.lcc_per_layer}, no-lcc-fallback={args.no_lcc_fallback}, "
          f"excluded seeds={len(explicit_excluded)}")

    true_m = score_module(true_module, anchors_in_module, H, S, split_seeds=True)
    true_m["diag_mean_ldeg"] = S["true_neighbor_ldeg"]
    pd.DataFrame([true_m]).T.rename(columns={0: "value"}).to_csv(out / "true_module_metrics.csv")
    print("\nTrue module metrics:")
    for k_, v in true_m.items():
        print(f"  {k_:45s} {v:.6g}" if not pd.isna(v) else f"  {k_:45s} N/A")
    if "drug_target_disease" in scorers:
        hit = scorers["drug_target_disease"].hits(sorted({s6._gene_of(n) for n in true_module}))
        pd.DataFrame({"gene": hit}).to_csv(out / "true_module_drug_target_genes.csv", index=False)

    needs_pipeline = any(n in args.nulls for n in ("seeds", "rewired")) or args.identity_only
    idf = out / "identity_check.csv"
    if needs_pipeline and not args.skip_identity_check and not args.from_partial:
        if (args.resume or args.perm_range) and idf.exists() and not args.identity_only:
            r0 = pd.read_csv(idf).iloc[0]
            ok = (r0["diag_jaccard_vs_true"] >= args.identity_min_jaccard
                  and abs(r0["diag_size"] - len(true_module)) <= 0.1 * len(true_module))
            print(f"[identity] reusing {idf.name} (Jaccard {r0['diag_jaccard_vs_true']:.3f}, "
                  f"size {int(r0['diag_size'])}): {'OK' if ok else 'FAILED'}. Delete it to re-run.")
            if not ok:
                return 2
        elif not identity_check(args, out):
            return 2
    if args.identity_only:
        return 0

    jobs = {
        "seeds": (null_seeds, args.n_perms_seeds,
                  "NULL seeds: layer/degree-matched random seeds (are my seeds special?)"),
        "rewired": (null_rewired, args.n_perms_rewired,
                    "NULL rewired: PPI edges shuffled within each cell-type layer (does the wiring matter?)"),
        "neighbors": (null_neighbors, args.n_perms_neighbors,
                      "NULL neighbors: true seeds + layer/degree-matched random neighbors (does the walk add anything?)"),
    }
    def finish_null(name, title, df):
        if df.empty:
            print(f"  [{name}] no valid permutations -- skipping report")
            return
        report(name, title, true_m, df, out)

    if args.from_partial:
        for name in args.nulls:
            _, n, title = jobs[name]
            df = load_partial(out, name)
            print(f"\n[{name}] summarising {len(df)}/{n} permutations found in partial files")
            finish_null(name, title, df)
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init, initargs=(S,),
                                 mp_context=multiprocessing.get_context("fork")) as ex:
            for name in args.nulls:
                task, n, title = jobs[name]
                print(f"\n[{name}] {n} permutations on {n_workers} workers ...")
                finish_null(name, title, run_null(ex, task, n, name, args.seed, out,
                                                  args.perm_range, args.resume))
    print(f"\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())