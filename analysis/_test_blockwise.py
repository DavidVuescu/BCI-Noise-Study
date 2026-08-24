"""Equivalence test for the vectorised SWLDA forward step.

SWLDAClassifierFast replaces the parent's per-candidate statsmodels refit with
the closed-form partitioned-regression (Frisch-Waugh-Lovell) p-value. That is an
implementation change, not a methodological one, so it must produce bit-identical
selections. This test asserts that on real data.

If this test fails, the blockwise results in data/derived/classifier-v3-blockwise/
are NOT comparable to the v3 results and must be regenerated with the parent class.

Usage:
    python -m analysis._test_blockwise
    python -m analysis._test_blockwise --subject 05
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.classifier import (  # noqa: E402
    DOWNSAMPLED_HZ,
    SWLDAClassifier,
    SWLDAClassifierFast,
    _evaluate,
    _extract_features_with_blocks,
    _load_epochs,
)


def run(subject_id: str = "01", derived_root: str = "data/derived") -> bool:
    epochs = _load_epochs(subject_id, "control", Path(derived_root))
    X, y, sub_block = _extract_features_with_blocks(epochs, target_sfreq=DOWNSAMPLED_HZ)

    held = sorted(int(b) for b in np.unique(sub_block))[0]
    train, test = sub_block != held, sub_block == held
    print(f"sub-{subject_id}, hold out sub-block {held}: "
          f"{train.sum()} train / {test.sum()} test, {X.shape[1]} features")

    t0 = time.time()
    slow = SWLDAClassifier(verbose=False).fit(X[train], y[train])
    t_slow = time.time() - t0

    t0 = time.time()
    fast = SWLDAClassifierFast(verbose=False).fit(X[train], y[train])
    t_fast = time.time() - t0

    print(f"  reference (statsmodels refit) : {t_slow:8.1f} s")
    print(f"  vectorised (FWL closed form)  : {t_fast:8.1f} s "
          f"({t_slow / max(t_fast, 1e-9):.0f}x faster)")

    checks: list[tuple[str, bool, str]] = []

    same_feats = slow.selected_features_ == fast.selected_features_
    checks.append(("selected features identical", same_feats,
                   f"{len(slow.selected_features_)} vs {len(fast.selected_features_)}"))

    if same_feats:
        dcoef = float(np.max(np.abs(slow.coef_ - fast.coef_)))
        checks.append(("coefficients identical", dcoef == 0.0, f"max|diff| = {dcoef:g}"))
        dint = abs(slow.intercept_ - fast.intercept_)
        checks.append(("intercept identical", dint == 0.0, f"diff = {dint:g}"))
        dthr = abs(slow.threshold_ - fast.threshold_)
        checks.append(("threshold identical", dthr == 0.0, f"diff = {dthr:g}"))
        same_pred = bool((slow.predict(X[test]) == fast.predict(X[test])).all())
        checks.append(("held-out predictions identical", same_pred, ""))
        r_slow = _evaluate(slow, X[test], y[test])
        r_fast = _evaluate(fast, X[test], y[test])
        same_ba = r_slow["balanced_accuracy"] == r_fast["balanced_accuracy"]
        checks.append(("balanced accuracy identical", same_ba,
                       f"{r_slow['balanced_accuracy']:.10f} vs "
                       f"{r_fast['balanced_accuracy']:.10f}"))

    print()
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:32s} {detail}")
        ok &= passed

    print("\n" + ("ALL CHECKS PASSED - implementations are equivalent."
                  if ok else "FAILURE - do not use the fast path."))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="01")
    ap.add_argument("--derived-root", default="data/derived")
    args = ap.parse_args(argv)
    return 0 if run(args.subject, args.derived_root) else 1


if __name__ == "__main__":
    sys.exit(main())
