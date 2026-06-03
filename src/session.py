"""
Session orchestrator: runs one recording (recorder + stimulus, integrated).

UPDATED for 3-sub-block paradigm: now accepts a sub-block target-cell
permutation and forwards it to the stimulus runner.

Protocol-aware usage (run all 4 recordings for a subject):
    python -m src.session --subject 01

    This reads protocol/order_assignments/order_assignments.csv and runs the
    4 recordings in the assigned order.  Use --start-from N to resume after
    a crash (N is 1-indexed condition_order).

Single-recording usage (still works as before):
    python -m src.session --subject 01 --condition control --targets 4,0,8

Programmatic:
    from src.session import run_recording, run_subject_protocol
    results = run_subject_protocol("01")
    # or
    meta = run_recording("01", "control", 300.0, target_cells=[4, 0, 8])
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.config import load_config
from src.recorder import Recorder
from src.stimulus import run_session as run_stimulus


PRE_ROLL_S = 1.0
POST_ROLL_S = 1.0

DEFAULT_CSV = Path("protocol/order_assignments/order_assignments.csv")

CONTROL_CONDITION = "control"


def _duration_for_condition(condition: str, cfg: dict) -> float:
    """Return the configured stimulus duration for this condition.

    Control gets the longer duration (training data); all others share the
    standard per-condition duration.  Both values come from config.yaml
    session.duration_control_s / session.duration_per_condition_s.
    """
    if condition.lower() == CONTROL_CONDITION:
        return float(cfg["session"].get("duration_control_s", 600.0))
    return float(cfg["session"].get("duration_per_condition_s", 300.0))


# ---------------------------------------------------------------------------
# Protocol lookup
# ---------------------------------------------------------------------------

def _normalize_subject_id(subject_id: str) -> str:
    """Return bare ID, stripping 'sub-' prefix if present."""
    return subject_id[4:] if subject_id.startswith("sub-") else subject_id


def load_subject_protocol(
    subject_id: str,
    csv_path: str | Path = DEFAULT_CSV,
) -> list[dict]:
    """Load the ordered recording protocol for one subject from the CSV.

    Accepts subject_id as either bare ('01') or BIDS-prefixed ('sub-01').

    Returns a list of 4 dicts sorted by condition_order:
        [
          {"condition_order": 1, "condition": "acoustic", "target_cells": [8, 4, 0]},
          ...
        ]

    Raises:
        FileNotFoundError: CSV not found.
        ValueError: Subject not present in CSV.
    """
    bare_id = _normalize_subject_id(subject_id)
    csv_subject = f"sub-{bare_id}"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Order-assignments CSV not found: {csv_path.absolute()}")

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["subject_id"] == csv_subject:
                rows.append({
                    "condition_order": int(row["condition_order"]),
                    "condition": row["condition"],
                    "target_cells": [
                        int(row["sub_block_1_target"]),
                        int(row["sub_block_2_target"]),
                        int(row["sub_block_3_target"]),
                    ],
                })

    if not rows:
        raise ValueError(
            f"Subject {csv_subject!r} not found in {csv_path}. "
            f"Available IDs run sub-01 through sub-60."
        )

    rows.sort(key=lambda r: r["condition_order"])
    return rows


# ---------------------------------------------------------------------------
# Single recording
# ---------------------------------------------------------------------------

def run_recording(
    subject_id: str,
    condition: str,
    duration_s: float,
    target_cells: list[int] | None = None,
    practice: bool = False,
    windowed: bool = True,
    seed: int | None = None,
    config_path: str | Path = "config.yaml",
) -> dict:
    """Run one full recording: pre-roll + stimulus + post-roll, save everything.

    Args:
        target_cells: Sub-block target-cell permutation, e.g. [4, 0, 8].
            If None, stimulus falls back to config defaults. For practice,
            the stimulus ignores this and uses center cell only.
    """
    cfg = load_config(config_path)

    refresh_hz = cfg["stimulus"]["monitor_refresh_hz"]
    frames_per_flash = cfg["stimulus"]["flash_frames"] + cfg["stimulus"]["isi_frames"]
    flash_period_s = frames_per_flash / refresh_hz
    n_flashes = int(duration_s / flash_period_s)

    raw_dir = Path(cfg["output"]["raw_dir"])

    if practice:
        effective_targets = [cfg["stimulus"]["target_cell"]]
        targets_label = f"[practice: {effective_targets[0]}]"
    elif target_cells is not None:
        effective_targets = target_cells
        targets_label = str(effective_targets)
    else:
        effective_targets = cfg["stimulus"]["sub_blocks"]["target_cells"]
        targets_label = f"{effective_targets} (config default)"

    print(f"\n{'='*60}")
    print(f"Session: subject={subject_id}  condition={condition}  "
          f"{'[PRACTICE]' if practice else ''}")
    print(f"  Target duration:  {duration_s:.1f}s")
    print(f"  Flash period:     {flash_period_s*1000:.1f}ms")
    print(f"  Planned flashes:  {n_flashes}")
    print(f"  Sub-block targets: {targets_label}")
    print(f"  Pre-roll: {PRE_ROLL_S}s  Post-roll: {POST_ROLL_S}s")
    print(f"{'='*60}\n")

    print("Starting recorder thread...")
    recorder = Recorder(cfg, max_duration_s=duration_s + PRE_ROLL_S + POST_ROLL_S + 90)
    recorder.start()

    print(f"Pre-roll: {PRE_ROLL_S}s of baseline EEG before stimulus starts...")
    time.sleep(PRE_ROLL_S)

    stats = recorder.stats()
    if stats["samples_received"] < int(PRE_ROLL_S * cfg["device"]["sampling_rate"] * 0.5):
        recorder.stop()
        raise RuntimeError(
            f"Recorder received only {stats['samples_received']} samples in "
            f"{PRE_ROLL_S}s pre-roll. Is UnicornUDP streaming?"
        )
    print(f"  Pre-roll OK: {stats['samples_received']} samples received.")

    print("\nRunning stimulus...")
    stim_meta = run_stimulus(
        config=cfg,
        output_dir=raw_dir,
        subject_id=subject_id,
        condition=condition,
        n_flashes=n_flashes,
        target_cells=target_cells,
        practice=practice,
        windowed=windowed,
        seed=seed,
    )
    print(f"Stimulus complete: {stim_meta['n_flashes_delivered']} flashes, "
          f"{stim_meta['late_frames']} late frames.")
    if stim_meta.get("reported_counts"):
        print(f"  Reported counts (sub-block: reported vs expected):")
        for rc in stim_meta["reported_counts"]:
            print(f"    block {rc['sub_block_index']} target={rc['target_cell']}: "
                  f"{rc['reported_count']} vs {rc['expected_count']}")

    print(f"\nPost-roll: {POST_ROLL_S}s to capture final ERP tail...")
    time.sleep(POST_ROLL_S)

    print("Stopping recorder...")
    recorder.stop()
    rec_stats = recorder.stats()
    print(f"  Total samples: {rec_stats['samples_received']}")
    print(f"  Dropped:       {rec_stats['samples_dropped']} "
          f"({100*rec_stats['dropout_rate']:.3f}%)")

    if practice:
        print("\n[PRACTICE] Nothing saved to disk.")
        return {"stimulus": stim_meta, "recorder_stats": rec_stats}

    print("\nSaving recorder data...")
    rec_meta = recorder.save(
        output_dir=raw_dir,
        subject_id=subject_id,
        condition=condition,
    )

    subject_dir = raw_dir / f"sub-{subject_id}"
    print(f"\nFiles in {subject_dir}/")
    stem = f"sub-{subject_id}_cond-{condition}"
    for suffix in ["eeg.npy", "acqtime.npy", "timestamps.npy",
                   "meta.json", "markers.csv", "session.json"]:
        path = subject_dir / f"{stem}_{suffix}"
        marker = "✓" if path.exists() else "✗"
        print(f"  {marker}  {path.name}")

    wall_anchor = rec_meta["wall_clock_anchor_unix"]
    stim_start = stim_meta["start_time_unix"]
    stim_stop = stim_meta["stop_time_unix"]
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"  Recorder anchor (first sample wall time): {wall_anchor:.6f}")
    print(f"  Stimulus first event (offset from anchor): "
          f"{stim_start - wall_anchor:.3f}s")
    print(f"  Stimulus last event (offset from anchor):  "
          f"{stim_stop - wall_anchor:.3f}s")
    print(f"  Recorder ran for: {rec_stats['elapsed_s']:.3f}s")

    return {"stimulus": stim_meta, "recorder_meta": rec_meta, "recorder_stats": rec_stats}


# ---------------------------------------------------------------------------
# Full-protocol runner (all 4 recordings for one subject)
# ---------------------------------------------------------------------------

def run_subject_protocol(
    subject_id: str,
    duration_s: float | None = None,
    practice: bool = False,
    windowed: bool = True,
    seed: int | None = None,
    config_path: str | Path = "config.yaml",
    csv_path: str | Path = DEFAULT_CSV,
    start_from: int = 1,
) -> list[dict]:
    """Run all 4 recordings for a subject in the assigned protocol order.

    Reads condition order and sub-block target cells from the order-assignments
    CSV.  Keeps the 4 recordings as separate run_recording() calls with a
    manual ENTER gate between them (so the experimenter can take a break,
    swap gel, etc.).

    Args:
        subject_id:  Bare ID ('01') or BIDS-prefixed ('sub-01').
        duration_s:  Override stimulus duration for every recording.  If None
                     (default), duration is read from config per condition:
                     control gets duration_control_s (600s), all others get
                     duration_per_condition_s (300s).
        start_from:  1-indexed condition_order to resume from — use this if
                     the session was interrupted mid-run (e.g. start_from=3
                     skips recordings 1 and 2).
        (other args passed through to run_recording)

    Returns:
        List of result dicts, one per completed recording, each containing
        {"recording": <protocol row>, "result": <run_recording return value>}.
    """
    protocol = load_subject_protocol(subject_id, csv_path)
    cfg = load_config(config_path)
    bare_id = _normalize_subject_id(subject_id)

    print(f"\n{'#'*60}")
    print(f"# SUBJECT PROTOCOL: sub-{bare_id}")
    if start_from > 1:
        print(f"# Resuming from recording {start_from}")
    print(f"#")
    for rec in protocol:
        rec_dur = duration_s if duration_s is not None else _duration_for_condition(rec["condition"], cfg)
        skip = " [SKIP]" if rec["condition_order"] < start_from else ""
        print(f"#   {rec['condition_order']}.  {rec['condition']:<12} "
              f"{rec_dur:.0f}s  targets={rec['target_cells']}{skip}")
    print(f"{'#'*60}\n")

    results: list[dict] = []
    for rec in protocol:
        if rec["condition_order"] < start_from:
            continue

        rec_dur = duration_s if duration_s is not None else _duration_for_condition(rec["condition"], cfg)
        order = rec["condition_order"]
        total = len(protocol)
        print(f"\n{'#'*60}")
        print(f"# Recording {order}/{total}: {rec['condition'].upper()}  ({rec_dur:.0f}s)")
        print(f"# Sub-block targets: {rec['target_cells']}")
        print(f"{'#'*60}")

        result = run_recording(
            subject_id=bare_id,
            condition=rec["condition"],
            duration_s=rec_dur,
            target_cells=rec["target_cells"],
            practice=practice,
            windowed=windowed,
            seed=seed,
            config_path=config_path,
        )
        results.append({"recording": rec, "result": result})

        if order < total:
            input(f"\n  Recording {order}/{total} complete. "
                  f"Press ENTER when ready for the next recording... ")

    print(f"\n{'#'*60}")
    print(f"# PROTOCOL COMPLETE for sub-{bare_id}")
    print(f"# {len(results)} recordings saved.")
    print(f"{'#'*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_targets(s: str) -> list[int]:
    """Parse '4,0,8' into [4, 0, 8]. Used by --targets CLI arg."""
    try:
        result = [int(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--targets must be comma-separated integers, got: {s!r}"
        )
    if not result:
        raise argparse.ArgumentTypeError("--targets cannot be empty")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run BCI noise study recordings for one subject.\n\n"
            "Without --condition: runs all 4 recordings in protocol order "
            "(reads order_assignments.csv automatically).\n"
            "With --condition: runs a single recording for that condition."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True,
                        help="Subject ID, e.g. '01' or 'sub-01'")
    parser.add_argument("--condition", default=None,
                        help="Condition tag (control/chewing/emi/acoustic). "
                             "Omit to run all 4 in protocol order.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stimulus duration in seconds. If omitted, auto-selected "
                             "from config: control=600s, all others=300s.")
    parser.add_argument("--targets", type=_parse_targets, default=None,
                        help="Comma-separated sub-block target cells, e.g. 4,0,8. "
                             "Only relevant with --condition; if omitted, looked up "
                             "from order_assignments.csv.")
    parser.add_argument("--start-from", type=int, default=1, metavar="N",
                        help="Resume full-protocol run from condition_order N "
                             "(1-indexed). Only used when --condition is omitted.")
    parser.add_argument("--practice", action="store_true",
                        help="Run as practice (no files saved, single center sub-block)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Fullscreen mode (default windowed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for sequence generation (optional)")
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help=f"Path to order_assignments CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    windowed = not args.fullscreen
    subject_id = _normalize_subject_id(args.subject)  # strip 'sub-' if present
    condition = args.condition.lower() if args.condition else None

    if condition is not None:
        # Single-recording mode
        targets = args.targets
        if targets is None:
            # Auto-lookup targets from CSV
            try:
                protocol = load_subject_protocol(subject_id, args.csv)
                match = next((r for r in protocol if r["condition"] == condition), None)
                if match:
                    targets = match["target_cells"]
                    print(f"[CSV lookup] targets for {condition}: {targets}")
                else:
                    print(f"[CSV lookup] condition {condition!r} not found for "
                          f"this subject; falling back to config default.")
            except (FileNotFoundError, ValueError) as e:
                print(f"[CSV lookup] {e}  -- falling back to config default.")

        # Auto-select duration from config if not explicitly given
        if args.duration is not None:
            duration_s = args.duration
        else:
            cfg = load_config()
            duration_s = _duration_for_condition(condition, cfg)
            print(f"[auto duration] {condition} -> {duration_s:.0f}s")

        run_recording(
            subject_id=subject_id,
            condition=condition,
            duration_s=duration_s,
            target_cells=targets,
            practice=args.practice,
            windowed=windowed,
            seed=args.seed,
        )
    else:
        # Full-protocol mode
        run_subject_protocol(
            subject_id=subject_id,
            duration_s=args.duration,  # None = auto per condition
            practice=args.practice,
            windowed=windowed,
            seed=args.seed,
            csv_path=args.csv,
            start_from=args.start_from,
        )


if __name__ == "__main__":
    main()
