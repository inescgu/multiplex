"""
weaver_fast_patch.py
---------------------
Drop-in speed patch for hidef.weaver.Weaver._build().

WHY: the stock _build() calls nx.has_path(G, a, b) and nx.ancestors(G, node)
repeatedly -- once per G-node in the redundant-edge-removal step, and (far
more expensively) once per terminal-node/cluster membership in the
terminal-attachment step, plus again in the terminal-assignment-propagation
step. Each of those calls re-walks the ENTIRE graph from scratch with a
fresh BFS/DFS. None of the underlying graph G changes across these three
steps, so this is enormous, avoidable, repeated work -- it's why your job
finished graph-init (557s) and redundant-edge-removal (5955s) but then hung
indefinitely in "attaching terminal nodes to the graph...".

FIX: compute G's transitive closure ONCE (nx.transitive_closure), then
replace every has_path()/ancestors() call with an O(1) lookup against it.
This is a single-core optimization (it does not parallelize the algorithm),
but it removes the actual source of the blowup, which parallelizing the
existing repeated-BFS approach would not meaningfully fix -- you'd just be
running the same wasteful computation on more cores.

USAGE (add near the top of step11c_weave.py, before you build/weave):

    import weaver_fast_patch  # this monkeypatches hidef.weaver.Weaver._build

    ... rest of your script unchanged ...
    wv = weaver.Weaver()
    wv.weave(P, cutoff=cutoff)   # now uses the fast _build under the hood

LIMITATIONS:
- Only supports merge=False (the default -- your step11c script never
  passes merge=True, so this is fine as-is).
- Semantically IDENTICAL to the original _build(): every has_path/ancestors
  check is replaced with an equivalent transitive-closure lookup, nothing
  about which edges get kept/removed changes. Results should match the
  original exactly, just much faster.
- Memory: transitive_closure stores an edge for every reachable pair of
  G-nodes (NOT terminals -- just your ~thousands of cluster nodes after
  filtering/dedup). For n cluster-nodes this is at most O(n^2) edges;
  for n in the 10-30k range this is comfortably within your 130G allocation
  (and your job was only using ~21G anyway).
"""
import itertools
from itertools import product as iproduct
from collections import defaultdict

import numpy as np
import scipy as sp
import networkx as nx

from hidef import weaver as hidef_weaver
from hidef.weaver import istuple, denumpize
from hidef.utils import containment_indices_boolean
from hidef import LOGGER


def _build_fast(self, **kwargs):
    cutoff = kwargs.pop('cutoff', 0.75)
    merge = kwargs.pop('merge', False)
    if merge:
        raise NotImplementedError(
            "weaver_fast_patch: merge=True is not supported by the fast "
            "_build patch. Remove the patch (or extend it) if you need "
            "merge=True; step11c_weave.py does not use it, so this "
            "shouldn't affect you."
        )

    assume_levels = self.assume_levels
    terminals = self.terminals
    n_nodes = self.n_terminals
    L = self._assignment
    labels = self._labels
    levels = self._levels
    n_sets = len(L)

    # *** FIX: stash the ORIGINAL cluster-membership matrix under its own
    # name. self._assignment gets overwritten later (the "propagate
    # terminal node assignments upward" step reassigns it to a
    # differently-shaped, propagated matrix), but every hierarchy node's
    # 'index' attribute is a row index into THIS matrix, not the
    # reassigned one. Without this, extract_communities() in
    # step11c_weave.py silently drops every internal (non-terminal)
    # community -- which is exactly the bug that produced a communities
    # pickle full of nothing but singleton leaf nodes. ***
    self._L_clusters = L.copy()

    rng = range(n_sets)
    if assume_levels:
        gen = ((i, j) for i, j in iproduct(rng, rng) if levels[i] > levels[j])
    else:
        gen = ((i, j) for i, j in iproduct(rng, rng) if i != j)

    # ---- graph init: unchanged from stock hidef (this part was never the
    # bottleneck -- 557s in your run, dominated by the CI matrix multiply) ----
    LOGGER.timeit('_init')
    LOGGER.info('initializing the graph...')
    CI = containment_indices_boolean(L, L)
    G = nx.DiGraph()
    for i, j in gen:
        C = CI[i, j]
        na, nb = (i, 0), (j, 0)
        if not G.has_node(na):
            G.add_node(na, index=i, level=levels[i], label=labels[i])
        if not G.has_node(nb):
            G.add_node(nb, index=j, level=levels[j], label=labels[j])
        if C >= cutoff:
            if G.has_edge(na, nb):
                C0 = G[na][nb]['weight']
                if C > C0:
                    G.remove_edge(na, nb)
                else:
                    continue
            G.add_edge(nb, na, weight=C)
    LOGGER.report('graph initialized in %.2fs', '_init')

    roots = [node for node, indeg in G.in_degree() if indeg == 0]
    if len(roots) > 1:
        root = (-1, 0)
        G.add_node(root, index=-1, level=-1, label=True)
        for node in roots:
            G.add_edge(root, node, weight=1.)
    else:
        root = roots[0]

    # ---- THE FIX: one transitive closure, reused by every step below ----
    LOGGER.timeit('_tc')
    LOGGER.info('computing transitive closure once (replaces all repeated '
                'has_path()/ancestors() calls below with O(1) lookups)...')
    TC = nx.transitive_closure(G)
    LOGGER.report('transitive closure computed in %.2fs', '_tc')

    # ---- redundant-edge removal: was nx.ancestors(G, node) per node ----
    LOGGER.timeit('_redundancy')
    LOGGER.info('removing redundant edges...')
    redundant = []
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if len(parents) < 2:
            continue
        ancestors = set(TC.predecessors(node))  # == nx.ancestors(G, node)
        for a in parents:
            for b in ancestors:
                if a != b and TC.has_edge(a, b):
                    redundant.append((a, node))
                    break
    G.remove_edges_from(redundant)
    LOGGER.report('redundant edges removed in %.2fs', '_redundancy')

    # ---- terminal attachment: was the step your job hung in.
    # was nx.has_path(G, node, other) / nx.has_path(G, other, node) per
    # terminal-node/cluster membership, now O(1) TC lookups ----
    LOGGER.timeit('_attach')
    LOGGER.info('attaching terminal nodes to the graph...')
    X = np.arange(n_nodes)
    nodes = [node for node in G.nodes]
    attached_record = defaultdict(list)

    for node in nodes:
        n_ = node[0]
        if n_ == -1:
            continue
        x = X[L[n_]]
        for i in x:
            ter = denumpize(terminals[i])
            attached = attached_record[ter]
            skip = False
            if attached:
                for other in reversed(attached):
                    if TC.has_edge(node, other):
                        skip = True
                        break
                    elif TC.has_edge(other, node):
                        attached.remove(other)
                        G.remove_edge(other, ter)
            if not skip:
                G.add_edge(node, ter, weight=1.)
                attached.append(node)
    LOGGER.report('terminal nodes attached in %.2fs', '_attach')

    # ---- propagate terminal assignments upward: was another O(n^2)
    # has_path() loop over itertools.combinations(nodes, 2) -- this is the
    # step you would have hung in NEXT even if attach had finished ----
    LOGGER.timeit('_update')
    LOGGER.info('propagate terminal node assignments upward in the hierarchy')
    L_sp = sp.sparse.csr_matrix(L.T)
    row, col = [], []
    for v in nodes:
        row.append(v[0])
        col.append(v[0])
    for v, w in itertools.combinations(nodes, 2):
        if TC.has_edge(v, w):
            row.append(w[0])
            col.append(v[0])
    data = np.ones_like(row, dtype=int)
    cc_mat = sp.sparse.coo_matrix((data, (row, col)), shape=(L.shape[0], L.shape[0]))
    cc_mat = cc_mat.tocsr()
    L_sp_new = L_sp.dot(cc_mat) > 0
    self._assignment = L_sp_new.toarray().T
    LOGGER.report('terminal nodes propagated in %.2fs', '_update')

    in_degrees = np.array([deg for (_, deg) in G.in_degree()])
    if np.where(in_degrees == 0)[0].size > 1:
        G.add_node('root')
        for node in G.nodes():
            if G.in_degree(node) == 0:
                G.add_edge('root', node)

    self._full = G

    def node_size(node):
        if istuple(node):
            i = node[0]
            return np.count_nonzero(L[i])
        return 1

    LOGGER.timeit('_sec')
    LOGGER.info('finding secondary edges...')
    secondary = []
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if len(parents) > 1:
            nsize = node_size(node)
            pref = []
            for p in parents:
                w = G.edges()[p, node]['weight']
                psize = node_size(p)
                usize = w * nsize
                j = usize / (nsize + psize - usize)
                pref.append(j)
            ranked_edges = [((x[0], node), x[1]) for x in
                             sorted(zip(parents, pref), key=lambda x: x[1], reverse=True)]
            secondary.extend(ranked_edges[1:])
    secondary.sort(key=lambda x: x[1], reverse=True)
    self._secondary = secondary
    LOGGER.report('secondary edges found in %.2fs', '_sec')

    return G


# Apply the monkeypatch on import.
hidef_weaver.Weaver._build = _build_fast
LOGGER.info('weaver_fast_patch applied: Weaver._build now uses a single '
            'transitive-closure computation instead of repeated has_path()/'
            'ancestors() calls.')