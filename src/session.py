"""
Session orchestrator: runs one recording (recorder + stimulus, integrated).

UPDATED for 3-sub-block paradigm: now accepts a sub-block target-cell
permutation and forwards it to the stimulus runner.

Usage (programmatic):
    from src.session import run_recording
    meta = run_recording(
        subject_id="pilot01",
        condition="control",
        duration_s=600,
        target_cells=[4, 0, 8],     # sub-block target permutation
        practice=False,
        windowed=True,
    )

Usage (command line):
    python -m src.session --subject pilot01 --condition control \\
        --duration 600 --targets 4,0,8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import load_config
from src.recorder import Recorder
from src.stimulus import run_session as run_stimulus


PRE_ROLL_S = 1.0
POST_ROLL_S = 1.0


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
        (other args unchanged)
    """
    cfg = load_config(config_path)

    refresh_hz = cfg["stimulus"]["monitor_refresh_hz"]
    frames_per_flash = cfg["stimulus"]["flash_frames"] + cfg["stimulus"]["isi_frames"]
    flash_period_s = frames_per_flash / refresh_hz
    n_flashes = int(duration_s / flash_period_s)

    raw_dir = Path(cfg["output"]["raw_dir"])

    # Resolve the effective target_cells for the header print (informational)
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

    # ---- Start the recorder ----
    print("Starting recorder thread...")
    # Generous buffer: stimulus duration + pre/post-roll + ~10s per sub-block for
    # rest gates and count prompts + paranoia margin.
    recorder = Recorder(cfg, max_duration_s=duration_s + PRE_ROLL_S + POST_ROLL_S + 90)
    recorder.start()

    # ---- Pre-roll ----
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

    # ---- Run the stimulus ----
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

    # ---- Post-roll ----
    print(f"\nPost-roll: {POST_ROLL_S}s to capture final ERP tail...")
    time.sleep(POST_ROLL_S)

    # ---- Stop the recorder ----
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

    print(f"\nFiles in {raw_dir}/")
    stem = f"sub-{subject_id}_cond-{condition}"
    for suffix in ["eeg.npy", "acqtime.npy", "timestamps.npy",
                   "meta.json", "markers.csv", "session.json"]:
        path = raw_dir / f"{stem}_{suffix}"
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


# ---- CLI ----

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
    parser = argparse.ArgumentParser(description="Run one BCI noise study recording.")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g. pilot01)")
    parser.add_argument("--condition", required=True,
                        help="Condition tag (control/chewing/emi/acoustic)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Stimulus duration in seconds (default 300)")
    parser.add_argument("--targets", type=_parse_targets, default=None,
                        help="Comma-separated sub-block target cells, e.g. 4,0,8. "
                             "If omitted, uses config default.")
    parser.add_argument("--practice", action="store_true",
                        help="Run as practice (no files saved, single center sub-block)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Fullscreen mode (default windowed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for sequence generation (optional)")
    args = parser.parse_args()

    run_recording(
        subject_id=args.subject,
        condition=args.condition,
        duration_s=args.duration,
        target_cells=args.targets,
        practice=args.practice,
        windowed=not args.fullscreen,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
