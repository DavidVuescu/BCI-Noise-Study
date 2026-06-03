"""Tests for session.py protocol-lookup logic.

Validates CSV loading, subject normalization, ordering, and target-cell
parsing without touching any hardware or running actual recordings.

Run from the project root:
    python -m src._test_session
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.session import (
    load_subject_protocol,
    _normalize_subject_id,
    _duration_for_condition,
    DEFAULT_CSV,
)


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    sys.exit(1)


def _check(condition: bool, msg: str) -> None:
    (_pass if condition else _fail)(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normalize_subject_id() -> None:
    print("\n[normalize_subject_id]")
    _check(_normalize_subject_id("sub-01") == "01", "strips 'sub-' prefix")
    _check(_normalize_subject_id("01") == "01",     "bare ID unchanged")
    _check(_normalize_subject_id("sub-10") == "10", "two-digit bare")
    _check(_normalize_subject_id("pilot01") == "pilot01", "non-numeric bare")


def test_csv_exists() -> None:
    print("\n[CSV file presence]")
    _check(DEFAULT_CSV.exists(), f"order_assignments.csv found at {DEFAULT_CSV}")


def test_load_known_subject_bare_id() -> None:
    print("\n[load_subject_protocol — bare ID]")
    rows = load_subject_protocol("01")
    _check(len(rows) == 4, f"4 rows returned (got {len(rows)})")
    _check(rows[0]["condition_order"] == 1, "first row is condition_order=1")
    _check(rows[-1]["condition_order"] == 4, "last row is condition_order=4")


def test_load_known_subject_bids_id() -> None:
    print("\n[load_subject_protocol — BIDS-prefixed ID]")
    rows_bare = load_subject_protocol("01")
    rows_bids = load_subject_protocol("sub-01")
    _check(rows_bare == rows_bids, "bare and BIDS-prefixed IDs return identical data")


def test_ordering() -> None:
    print("\n[ordering correctness]")
    # Spot-check a handful of subjects to confirm condition_order is 1→4
    for subj in ("01", "05", "15", "30", "60"):
        rows = load_subject_protocol(subj)
        orders = [r["condition_order"] for r in rows]
        _check(orders == [1, 2, 3, 4], f"sub-{subj}: order is {orders}")


def test_conditions_are_valid() -> None:
    print("\n[valid condition names]")
    valid = {"control", "chewing", "emi", "acoustic"}
    for subj in ("01", "10", "20", "40", "60"):
        rows = load_subject_protocol(subj)
        conditions = {r["condition"] for r in rows}
        _check(
            conditions == valid,
            f"sub-{subj}: conditions={conditions}",
        )


def test_target_cells_values() -> None:
    print("\n[target_cells values are from {{0, 4, 8}}]")
    allowed = {0, 4, 8}
    for subj in ("01", "02", "03"):
        rows = load_subject_protocol(subj)
        for row in rows:
            tc = row["target_cells"]
            _check(len(tc) == 3, f"sub-{subj} {row['condition']}: 3 targets (got {tc})")
            _check(set(tc) <= allowed,
                   f"sub-{subj} {row['condition']}: targets {tc} subset of {{0,4,8}}")
            _check(set(tc) == allowed,
                   f"sub-{subj} {row['condition']}: targets {tc} cover all 3 cells")


def test_target_permutation_coverage() -> None:
    """Each condition covers all three target cells (each of 0/4/8 appears at
    least once across the 4 recordings for a subject)."""
    print("\n[target permutations within a subject]")
    for subj in ("01", "25", "50"):
        rows = load_subject_protocol(subj)
        all_targets = {cell for r in rows for cell in r["target_cells"]}
        _check(
            all_targets == {0, 4, 8},
            f"sub-{subj}: all cells covered across recordings, got {all_targets}",
        )
        for r in rows:
            perms_str = str([tuple(x["target_cells"]) for x in rows])
            _check(
                len(r["target_cells"]) == 3,
                f"sub-{subj} {r['condition']}: 3 sub-block targets",
            )
        _ = perms_str  # used above


def test_unknown_subject_raises() -> None:
    print("\n[unknown subject raises ValueError]")
    try:
        load_subject_protocol("sub-999")
        _fail("expected ValueError for sub-999")
    except ValueError as e:
        _pass(f"ValueError raised: {e}")


def test_missing_csv_raises() -> None:
    print("\n[missing CSV raises FileNotFoundError]")
    try:
        load_subject_protocol("01", csv_path="nonexistent/path.csv")
        _fail("expected FileNotFoundError")
    except FileNotFoundError as e:
        _pass(f"FileNotFoundError raised: {e}")


def test_spot_check_sub01() -> None:
    """Manually verified values from the CSV for sub-01."""
    print("\n[spot-check sub-01 against known CSV values]")
    rows = load_subject_protocol("01")
    expected = [
        {"condition_order": 1, "condition": "acoustic", "target_cells": [8, 4, 0]},
        {"condition_order": 2, "condition": "control",  "target_cells": [4, 8, 0]},
        {"condition_order": 3, "condition": "chewing",  "target_cells": [4, 0, 8]},
        {"condition_order": 4, "condition": "emi",      "target_cells": [4, 8, 0]},
    ]
    for got, exp in zip(rows, expected):
        _check(got == exp, f"order {exp['condition_order']}: {got}")


def test_spot_check_sub02() -> None:
    """Manually verified values for sub-02."""
    print("\n[spot-check sub-02 against known CSV values]")
    rows = load_subject_protocol("02")
    expected = [
        {"condition_order": 1, "condition": "control",  "target_cells": [0, 4, 8]},
        {"condition_order": 2, "condition": "chewing",  "target_cells": [8, 0, 4]},
        {"condition_order": 3, "condition": "emi",      "target_cells": [0, 4, 8]},
        {"condition_order": 4, "condition": "acoustic", "target_cells": [4, 8, 0]},
    ]
    for got, exp in zip(rows, expected):
        _check(got == exp, f"order {exp['condition_order']}: {got}")


def test_no_sub_sub_prefix() -> None:
    """Passing 'sub-01' as subject_id must not produce 'sub-sub-01' filenames.

    run_recording() prepends 'sub-' internally, so the CLI and
    run_subject_protocol() must both strip the prefix before forwarding.
    We verify _normalize_subject_id does the right thing for all input forms.
    """
    print("\n[sub-sub prefix guard]")
    _check(_normalize_subject_id("sub-01") == "01",  "sub-01 -> 01")
    _check(_normalize_subject_id("sub-10") == "10",  "sub-10 -> 10")
    _check(_normalize_subject_id("01") == "01",      "01 -> 01  (unchanged)")
    _check(_normalize_subject_id("pilot01") == "pilot01", "pilot01 -> pilot01 (unchanged)")
    # Confirm the resulting directory stem would be correct
    for raw, want in [("sub-01", "sub-01"), ("01", "sub-01"), ("pilot01", "sub-pilot01")]:
        got = f"sub-{_normalize_subject_id(raw)}"
        _check(got == want, f"--subject {raw!r} -> dir stem {got!r} (want {want!r})")


def test_zero_padding_preserved() -> None:
    """Subject IDs like '01' must stay '01', not become '1'."""
    print("\n[zero-padding preservation]")
    # argparse passes CLI args as strings, so this is really an identity test
    for subj in ("01", "09", "10", "60"):
        bare = _normalize_subject_id(subj)
        _check(bare == subj, f"bare_id({subj!r}) == {subj!r}")
    for subj_bids, want in [("sub-01", "01"), ("sub-09", "09"), ("sub-10", "10")]:
        bare = _normalize_subject_id(subj_bids)
        _check(bare == want, f"bare_id({subj_bids!r}) == {want!r}")


def test_duration_per_condition() -> None:
    """Control gets 600s, all other conditions get 300s (from config)."""
    print("\n[condition-aware duration]")
    from src.config import load_config
    cfg = load_config()
    _check(
        _duration_for_condition("control", cfg) == 600.0,
        "control duration == 600s",
    )
    for cond in ("chewing", "emi", "acoustic"):
        d = _duration_for_condition(cond, cfg)
        _check(d == 300.0, f"{cond} duration == 300s (got {d})")
    # Case-insensitive (CLI lowercases, but double-check helper)
    _check(
        _duration_for_condition("Control", cfg) == 600.0,
        "Control (capital C) also gets 600s",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  session.py — protocol lookup tests")
    print("=" * 55)

    test_normalize_subject_id()
    test_no_sub_sub_prefix()
    test_zero_padding_preserved()
    test_csv_exists()
    test_load_known_subject_bare_id()
    test_load_known_subject_bids_id()
    test_ordering()
    test_conditions_are_valid()
    test_target_cells_values()
    test_target_permutation_coverage()
    test_unknown_subject_raises()
    test_missing_csv_raises()
    test_spot_check_sub01()
    test_spot_check_sub02()
    test_duration_per_condition()

    print(f"\n{'='*55}")
    print("  All tests passed.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
