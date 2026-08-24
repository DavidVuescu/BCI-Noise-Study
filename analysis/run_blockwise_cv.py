"""Runner for the blockwise (leave-one-sub-block-out) robustness analysis.

Executes analysis.classifier.train_and_evaluate_blockwise() for every subject
that has a complete set of preprocessed epochs, writing one results JSON per
subject to data/derived/classifier-v3-blockwise/.

The analysis is ~3x the cost of the v3 run (three SWLDA fits per subject instead
of one), so this is a standalone script rather than a notebook cell. It is
resumable: subjects whose results JSON already exists are skipped unless
--force is passed. That makes it safe to re-launch after an interruption
WITHOUT re-running completed subjects, which matters because the pre-commitment
in DEVIATIONS.md (2026-08-13) specifies a single run with no parameter tuning.

Usage:
    python -m analysis.run_blockwise_cv                 # all subjects, resumable
    python -m analysis.run_blockwise_cv --workers 2
    python -m analysis.run_blockwise_cv --subjects 01 02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.classifier import (  # noqa: E402
    DERIVED_PIPELINE_BLOCKWISE,
    PREPROCESSING_PIPELINE,
    train_and_evaluate_blockwise,
)

CONDITIONS = ["control", "chewing", "emi", "acoustic"]


def discover_subjects(derived_root: Path) -> list[str]:
    """Subjects with all four preprocessed condition files present."""
    preproc = derived_root / PREPROCESSING_PIPELINE
    out = []
    for sub_dir in sorted(preproc.glob("sub-*")):
        sid = sub_dir.name.replace("sub-", "")
        if all((sub_dir / f"sub-{sid}_cond-{c}_epo.fif").exists() for c in CONDITIONS):
            out.append(sid)
    return out


def _already_done(sid: str, derived_root: Path) -> bool:
    p = derived_root / DERIVED_PIPELINE_BLOCKWISE / f"sub-{sid}" / f"sub-{sid}_results.json"
    if not p.exists():
        return False
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False


def _run_one(args: tuple[str, str]) -> tuple[str, str, float, str]:
    sid, derived_root = args
    t0 = time.time()
    try:
        r = train_and_evaluate_blockwise(sid, derived_root=derived_root, save=True)
        ba = r["per_condition"]["control_heldout"]["balanced_accuracy"] * 100
        return sid, "ok", time.time() - t0, f"control {ba:.1f}%"
    except Exception:
        return sid, "FAIL", time.time() - t0, traceback.format_exc(limit=3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Blockwise CV robustness run")
    ap.add_argument("--derived-root", default="data/derived")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute subjects that already have results")
    args = ap.parse_args(argv)

    derived_root = Path(args.derived_root)
    subjects = args.subjects or discover_subjects(derived_root)

    if not args.force:
        pending = [s for s in subjects if not _already_done(s, derived_root)]
    else:
        pending = list(subjects)

    print(f"[blockwise] {len(subjects)} subjects discovered, "
          f"{len(pending)} to run, {args.workers} worker(s)", flush=True)

    t0 = time.time()
    done = 0
    payload = [(s, str(derived_root)) for s in pending]

    if args.workers > 1 and len(payload) > 1:
        with Pool(args.workers) as pool:
            for sid, status, el, msg in pool.imap_unordered(_run_one, payload):
                done += 1
                print(f"[{done}/{len(pending)}] sub-{sid} {status} "
                      f"({el:.0f}s) {msg}", flush=True)
    else:
        for item in payload:
            sid, status, el, msg = _run_one(item)
            done += 1
            print(f"[{done}/{len(pending)}] sub-{sid} {status} "
                  f"({el:.0f}s) {msg}", flush=True)

    print(f"[blockwise] complete in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
