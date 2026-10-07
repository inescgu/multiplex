#!/usr/bin/env python3
"""
Step 11c-faithful, Phase 2: the actual multi-day weave.

Takes Phase 1's consensus output (cluG_collapsed_w_len: consensus cluster
membership arrays + their real hidef_finder persistence values) and runs
the SAME weave() call hidef_finder.py's own __main__ makes:

    wv = weaver.Weaver()
    T = wv.weave(cluG_collapsed, boolean=True, levels=False, merge=True, cutoff=args.t)

then writes output via hidef_finder's own output_all(), which produces the
real .nodes/.edges/.gml files with genuine persistence in the last column
-- not an approximation.

IMPORTANT: this deliberately does NOT use weaver_fast_patch.py.
weaver_fast_patch._build_fast raises NotImplementedError for merge=True,
and hidef_finder's real pipeline requires merge=True. Faithfulness to the
official methodology and the earlier speed patch are mutually exclusive
here -- this run uses the stock, slow Weaver._build(), which is why this
is the multi-day step. There is no way to have both in one run; if you
want the speed patch's benefit, it would need to be extended to support
merge=True first, which is out of scope for "faithful reproduction."

USAGE (submit as its own sbatch job with a multi-day walltime, same as the
original weave was)
-----
    python step11c_phase2_weave.py \
        --consensus 11_hidef_results_faithful/consensus.pkl \
        --output-prefix 11_hidef_results_faithful/heterogeneous_hidef_faithful \
        --cutoff 0.75
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from provenance import write_manifest
from hidef import weaver
from hidef.hidef_finder import output_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consensus", required=True, help="Pickle from Phase 1")
    ap.add_argument("--output-prefix", required=True,
                     help="Prefix for hidef_finder's own output files: "
                          "<prefix>.nodes / .edges / .gml / .weaver")
    ap.add_argument("--cutoff", type=float, default=0.75,
                     help="hidef_finder's --t, same containment threshold used in Phase 1's "
                          "consensus() -- should match what you passed to Phase 1's --jaccard")
    ap.add_argument("--iter", action="store_true",
                     help="hidef_finder's --iter: iterate weave() until fully converged "
                          "(hidef_finder's own default is off; only enable if you specifically "
                          "want that behavior)")
    ap.add_argument("--skipgml", action="store_true")
    args = ap.parse_args()

    write_manifest(Path(args.output_prefix).parent, args, inputs=[args.consensus])
    with open(args.consensus, "rb") as f:
        data = pickle.load(f)

    cluG_collapsed_w_len = data["cluG_collapsed_w_len"]
    node_names = data["node_names"]
    print(f"Loaded {len(cluG_collapsed_w_len):,} consensus clusters from Phase 1 "
          f"({data['n_partitions']} original partitions, {len(node_names):,} nodes)\n")

    if not cluG_collapsed_w_len:
        raise ValueError("Phase 1 produced zero consensus clusters -- nothing to weave. "
                          "Re-run Phase 1 with looser --chi/--consensus-p/--minsize first.")

    cluG_collapsed = [x[0] for x in cluG_collapsed_w_len]
    len_component = [x[1] for x in cluG_collapsed_w_len]

    # hidef_finder's own __main__ prepends the trivial "everything" root
    # cluster with persistence 0 before weaving -- replicate exactly.
    cluG_collapsed.insert(0, np.ones(len(cluG_collapsed[0]), ))
    len_component.insert(0, 0)

    print(f"Weaving {len(cluG_collapsed):,} clusters (cutoff={args.cutoff}, merge=True) "
          f"-- this is the slow, multi-day step, using stock (unpatched) Weaver._build ...")

    wv = weaver.Weaver()
    T = wv.weave(cluG_collapsed, boolean=True, levels=False, merge=True, cutoff=args.cutoff)

    print("Weave complete. Writing output via hidef_finder's own output_all() "
          "(.nodes file's last column is the real persistence) ...")

    wv = output_all(wv, node_names, args.output_prefix,
                     persistence=len_component, iter=args.iter, skipgml=args.skipgml)

    print(f"\n✓ Done. Outputs written with prefix: {args.output_prefix}")
    print(f"  {args.output_prefix}.nodes  -- community, size, members, persistence (real, from hidef_finder)")
    print(f"  {args.output_prefix}.edges  -- parent/child hierarchy")
    if not args.skipgml:
        print(f"  {args.output_prefix}.gml    -- Cytoscape-loadable hierarchy")
    print(f"  {args.output_prefix}.weaver -- pickled Weaver object (so you never need to "
          f"re-weave to inspect internals again)")


if __name__ == "__main__":
    main()