"""
provenance.py

Put this ONE file in multiplex/ (top level). Every script -- in multiplex/
directly, or in multiplex/phen/ via the sys.path snippet below -- calls
write_manifest() as close to the top of main() as possible, right after
argparse. It writes a manifest.json into the script's own output directory
recording exactly what produced it: script name, full CLI args, git commit
(if the repo is clean or not), timestamp, and hostname/SLURM job id if
running under SLURM.

This solves the exact problem you already ran into once (the old
complete-graph persistence numbers sitting around indistinguishable from
the fixed windowed ones): six months from now, `results/11c_phase1/.../
manifest.json` tells you unambiguously whether a given consensus.pkl came
from the old or new windowing logic, at what --neighbors, and whether the
code was even committed at the time (dirty repo = treat the result as
provisional, don't cite it).

USAGE in a script that lives in multiplex/ directly:
    from provenance import write_manifest
    ...
    args = ap.parse_args()
    write_manifest(out_dir, args)

USAGE in a script under multiplex/phen/:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from provenance import write_manifest
    ...
    write_manifest(out_dir, args)
"""
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_info():
    repo_dir = Path(__file__).resolve().parent
    def run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=repo_dir, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    commit = run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"git_available": False}
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {"git_available": True, "commit": commit, "dirty": bool(dirty),
            "dirty_files": dirty.splitlines() if dirty else [],
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"])}

def _file_stamp(p):
    try:
        st = os.stat(p)
        return {"size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    except OSError:
        return {"missing": True}

def write_manifest(output_dir, args=None, script_path=None, extra=None, inputs=None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {} if args is None else (args if isinstance(args, dict) else vars(args))
    now = datetime.now(timezone.utc)
    manifest = {
        "script": script_path or sys.argv[0],
        "full_command": " ".join(sys.argv),
        "args": args_dict,
        "inputs": {str(p): _file_stamp(p) for p in (inputs or []) if p},
        "timestamp_utc": now.isoformat(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": _git_info(),
    }
    if extra:
        manifest["extra"] = extra
    if manifest["git"].get("dirty"):
        print(f"  ⚠ provenance: {len(manifest['git']['dirty_files'])} tracked file(s) modified "
              f"and uncommitted -- don't cite this run.")
    stem = Path(manifest["script"]).stem
    path = output_dir / f"manifest_{stem}_{now:%Y%m%dT%H%M%SZ}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  provenance manifest -> {path}")
    return manifest

def append_results_index(index_csv, row: dict):
    """
    Append one row to a top-level lab-notebook CSV (results/RESULTS_INDEX.csv).
    Call this manually, once, after you've LOOKED at a run's output and
    decided it's worth keeping -- not automatically from inside every
    script, or the index fills up with abandoned exploratory runs and stops
    being useful as a "what did I actually use" record.

    Suggested columns: date, step, output_dir, key_params, one_line_decision
    e.g.:
        append_results_index("results/RESULTS_INDEX.csv", {
            "date": "2026-09-29", "step": "11c_phase1",
            "output_dir": "results/11c_phase1/neighbors3_chi5_p75_20260929",
            "key_params": "neighbors=3, chi=5, consensus_p=75",
            "decision": "chosen as final -- stable vs neighbors=5,10 per sweep, "
                        "used for Phase 2 weave and all downstream enrichment",
        })
    """
    import csv
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = index_csv.exists()
    with open(index_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended to {index_csv}")
    