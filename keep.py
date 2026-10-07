#!/usr/bin/env python3
# run for what you want to keep: python keep.py 11c_phase1 results/11c_phase1/neighbors3_20260929/ "neighbors=3 chi=5 p=75" "chosen: stable vs neighbors 5/10"
import sys, datetime
from provenance import append_results_index
step, out_dir, params, decision = sys.argv[1:5]
append_results_index("results/RESULTS_INDEX.csv", {
    "date": datetime.date.today().isoformat(), "step": step,
    "output_dir": out_dir, "key_params": params, "decision": decision})