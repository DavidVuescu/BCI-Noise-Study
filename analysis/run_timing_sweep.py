"""Full-dataset sweep of the marker-alignment correction.

Scores every recording under both clock fits (least squares, as published, and
the upper-envelope refit) so that the recovery claim rests on the whole dataset
rather than on a hand-picked sample.

WHY A SCRIPT RATHER THAN A NOTEBOOK CELL
---------------------------------------------------------------------------
Each recording is loaded and preprocessed twice, once per fit mode, so the sweep
is roughly 336 preprocessing passes. That is tens of minutes serially, long
enough that a mid-run failure must not cost the whole run. One JSON is written
per recording as it completes and finished recordings are skipped on restart.

Both modes filter the same raw data, so hoisting the filter out of
preprocess_recording() would save about a further factor of two. That is
deliberately not done: it would mean duplicating registered preprocessing logic
outside analysis/preprocess.py, and a silent divergence between the two copies
would be far more expensive than the minutes saved. preprocess_recording()
remains the single source of truth.

CONTROL RECORDINGS ARE CONFOUNDED
---------------------------------------------------------------------------
Scoring uses each subject's saved v3 model, trained on their control epochs under
the original alignment. For a noise recording the model is fixed and only the
test data moves. For a control recording, re-alignment moves the training epochs
themselves, so a change may reflect train/test mismatch rather than signal. Rows
are marked `confounded` and should be excluded from recovery claims unless the
model is retrained on re-aligned control epochs.

Usage:
    python -m analysis.run_timing_sweep                    # all, resumable
    python -m analysis.run_timing_sweep --workers 6        # AMD 2600X: 6 cores
    python -m analysis.run_timing_sweep --subjects 19 33 04
    python -m analysis.run_timing_sweep --flagged-only     # the 11 flagged
    python -m analysis.run_timing_sweep --force            # recompute all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

# Keep BLAS single-threaded inside workers; the parallelism is across
# recordings, and nested threading oversubscribes the CPU and slows it down.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.timing_diagnostic import (  # noqa: E402
    CONDITIONS, DERIVED_PIPELINE, recording_residuals, sweep_recording,
)


def _out_dir(derived_root: Path) -> Path:
    return derived_root / DERIVED_PIPELINE / "per_recording"


def _out_path(derived_root: Path, sid: str, cond: str) -> Path:
    return _out_dir(derived_root) / f"sub-{sid}_cond-{cond}.json"


def discover(data_dir: Path, derived_root: Path) -> list[tuple[str, str]]:
    """Recordings that have raw files AND a saved v3 model to score against."""
    out = []
    for sub_dir in sorted(Path(data_dir).glob("sub-*")):
        sid = sub_dir.name.replace("sub-", "")
        if sid.startswith("pilot"):
            continue
        model = (derived_root / "classifier-v3" / f"sub-{sid}"
                 / f"sub-{sid}_model.pkl")
        if not model.exists():
            continue
        for cond in CONDITIONS:
            if (sub_dir / f"sub-{sid}_cond-{cond}_eeg.npy").exists():
                out.append((sid, cond))
    return out


def _run_one(args):
    sid, cond, data_dir, derived_root = args
    try:
        rec = sweep_recording(sid, cond, data_dir, derived_root)
        path = _out_path(Path(derived_root), sid, cond)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2))
        return sid, cond, "ok", rec
    except Exception:
        return sid, cond, "FAIL", {"error": traceback.format_exc(limit=3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full timing-correction sweep")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--derived-root", default="data/derived")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--flagged-only", action="store_true",
                    help="restrict to recordings above the residual flag")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    data_dir, derived_root = Path(args.data_dir), Path(args.derived_root)
    pairs = discover(data_dir, derived_root)

    if args.subjects:
        keep = set(args.subjects)
        pairs = [(s, c) for s, c in pairs if s in keep]

    if args.flagged_only:
        print("[sweep] screening residuals to find flagged recordings...", flush=True)
        pairs = [(s, c) for s, c in pairs
                 if recording_residuals(s, c, data_dir)["flagged"]]

    pending = pairs if args.force else [
        (s, c) for s, c in pairs if not _out_path(derived_root, s, c).exists()
    ]

    print(f"[sweep] {len(pairs)} recordings in scope, {len(pending)} to run, "
          f"{args.workers} worker(s)", flush=True)
    if not pending:
        print("[sweep] nothing to do (use --force to recompute)")
        return 0

    payload = [(s, c, str(data_dir), str(derived_root)) for s, c in pending]
    t0, done, failed = time.time(), 0, []

    def _report(sid, cond, status, rec):
        nonlocal done
        done += 1
        el = time.time() - t0
        eta = (el / done) * (len(pending) - done) / 60
        if status == "ok":
            tag = " CONFOUNDED" if rec.get("confounded") else ""
            flag = " FLAGGED" if rec.get("flagged") else ""
            print(f"[{done}/{len(pending)}] sub-{sid} {cond:9s} "
                  f"resid={rec['residual_std_ms']:7.1f}ms "
                  f"AUC {rec['auc_ols']:.3f}->{rec['auc_envelope']:.3f} "
                  f"({rec['auc_delta']:+.3f}){flag}{tag}  ETA {eta:.1f}m", flush=True)
        else:
            failed.append((sid, cond))
            print(f"[{done}/{len(pending)}] sub-{sid} {cond} FAIL", flush=True)

    if args.workers > 1 and len(payload) > 1:
        with Pool(args.workers) as pool:
            for sid, cond, status, rec in pool.imap_unordered(_run_one, payload):
                _report(sid, cond, status, rec)
    else:
        for item in payload:
            _report(*_run_one(item))

    mins = (time.time() - t0) / 60
    print(f"\n[sweep] complete in {mins:.1f} min | "
          f"{done - len(failed)} ok, {len(failed)} failed", flush=True)
    if failed:
        print("[sweep] failures:", failed)
    print(f"[sweep] results -> {_out_dir(derived_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
