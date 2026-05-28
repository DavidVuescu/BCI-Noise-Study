"""
Session orchestrator: runs one recording (recorder + stimulus, integrated).

This is the entry point for a single subject × single condition recording.
It starts the UDP recorder, waits for pre-roll, runs the stimulus,
post-rolls, stops the recorder, and saves everything under matching
filenames.

Usage (programmatic):
    from src.session import run_recording
    meta = run_recording(
        subject_id="pilot01",
        condition="control",
        duration_s=600,  # 10 min for control, 300 for noise conditions
        practice=False,
        windowed=True,   # False for real recordings
    )

Usage (command line):
    python -m src.session --subject pilot01 --condition control --duration 600
    python -m src.session --subject pilot01 --condition control --practice
    python -m src.session --subject pilot01 --condition control --duration 600 --fullscreen
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import load_config
from src.recorder import Recorder
from src.stimulus import run_session as run_stimulus


# Pre/post-roll: recorder runs this long before stimulus starts and after
# it ends. Pre-roll guarantees the recorder is receiving before the first
# marker is emitted (which prevents negative acquisition times in analysis).
# Post-roll captures the tail of the last ERP (~600ms response window) so
# we don't truncate the response to the final flash.
PRE_ROLL_S = 1.0
POST_ROLL_S = 1.0


def run_recording(
    subject_id: str,
    condition: str,
    duration_s: float,
    practice: bool = False,
    windowed: bool = True,
    seed: int | None = None,
    config_path: str | Path = "config.yaml",
) -> dict:
    """Run one full recording: pre-roll + stimulus + post-roll, save everything.

    Args:
        subject_id:  Identifier (e.g. "pilot01"). Used in filenames.
        condition:   One of the conditions in config['session']['conditions'].
                     For practice runs, this is just a tag (no validation).
        duration_s:  Target stimulus duration in seconds. We compute n_flashes
                     from this and the per-flash period (flash + isi).
        practice:    If True, run the paradigm but write NOTHING to disk.
                     Recorder still runs (subjects benefit from being recorded
                     during practice too, even if we throw the data away).
        windowed:    Pygame windowed (dev) vs fullscreen (real). Default True.
        seed:        Optional RNG seed for sequence generation.
        config_path: Where to load config from.

    Returns:
        Combined session metadata dict.
    """
    cfg = load_config(config_path)

    # ---- Compute n_flashes from duration ----
    # Per-flash period = flash phase + ISI phase, both in frames.
    refresh_hz = cfg["stimulus"]["monitor_refresh_hz"]
    frames_per_flash = cfg["stimulus"]["flash_frames"] + cfg["stimulus"]["isi_frames"]
    flash_period_s = frames_per_flash / refresh_hz
    n_flashes = int(duration_s / flash_period_s)

    # ---- Output paths ----
    raw_dir = Path(cfg["output"]["raw_dir"])

    print(f"\n{'='*60}")
    print(f"Session: subject={subject_id}  condition={condition}  "
          f"{'[PRACTICE]' if practice else ''}")
    print(f"  Target duration: {duration_s:.1f}s")
    print(f"  Flash period:    {flash_period_s*1000:.1f}ms")
    print(f"  Planned flashes: {n_flashes}")
    print(f"  Pre-roll: {PRE_ROLL_S}s  Post-roll: {POST_ROLL_S}s")
    print(f"{'='*60}\n")

    # ---- Start the recorder ----
    print("Starting recorder thread...")
    recorder = Recorder(cfg, max_duration_s=duration_s + PRE_ROLL_S + POST_ROLL_S + 30)
    recorder.start()

    # ---- Pre-roll: let the recorder warm up ----
    # The recorder is now accumulating samples in the background. We sleep
    # for PRE_ROLL_S so that wall_clock_anchor (first sample's time) is
    # strictly before any stimulus marker time.
    print(f"Pre-roll: {PRE_ROLL_S}s of baseline EEG before stimulus starts...")
    time.sleep(PRE_ROLL_S)

    # Verify the recorder is actually receiving before we start the stimulus.
    # If it's stuck (e.g. UnicornUDP not streaming), better to know now.
    stats = recorder.stats()
    if stats["samples_received"] < int(PRE_ROLL_S * cfg["device"]["sampling_rate"] * 0.5):
        recorder.stop()
        raise RuntimeError(
            f"Recorder received only {stats['samples_received']} samples in "
            f"{PRE_ROLL_S}s pre-roll. Is UnicornUDP streaming?"
        )
    print(f"  Pre-roll OK: {stats['samples_received']} samples received.")

    # ---- Run the stimulus ----
    # This blocks until the stimulus completes (or is aborted via ESC).
    # Recorder continues in the background throughout.
    print("\nRunning stimulus...")
    stim_meta = run_stimulus(
        config=cfg,
        output_dir=raw_dir,
        subject_id=subject_id,
        condition=condition,
        n_flashes=n_flashes,
        practice=practice,
        windowed=windowed,
        seed=seed,
    )
    print(f"Stimulus complete: {stim_meta['n_flashes_delivered']} flashes, "
          f"{stim_meta['late_frames']} late frames.")

    # ---- Post-roll: capture the tail of the last ERP ----
    print(f"\nPost-roll: {POST_ROLL_S}s to capture final ERP tail...")
    time.sleep(POST_ROLL_S)

    # ---- Stop the recorder ----
    print("Stopping recorder...")
    recorder.stop()
    rec_stats = recorder.stats()
    print(f"  Total samples: {rec_stats['samples_received']}")
    print(f"  Dropped:       {rec_stats['samples_dropped']} "
          f"({100*rec_stats['dropout_rate']:.3f}%)")

    # ---- Save (unless practice) ----
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

    # ---- Alignment summary ----
    # Compute the relationship between stimulus and recorder clocks.
    # This is a sanity check on alignment, not a correction.
    wall_anchor = rec_meta["wall_clock_anchor_unix"]
    stim_start = stim_meta["start_time_unix"]
    stim_stop = stim_meta["stop_time_unix"]
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"  Recorder anchor (first sample wall time): {wall_anchor:.6f}")
    print(f"  Stimulus first flash (offset from anchor): "
          f"{stim_start - wall_anchor:.3f}s "
          f"(should be ≥ {PRE_ROLL_S}s pre-roll, less drawing overhead)")
    print(f"  Stimulus last flash (offset from anchor):  "
          f"{stim_stop - wall_anchor:.3f}s")
    print(f"  Recorder ran for: {rec_stats['elapsed_s']:.3f}s")

    return {"stimulus": stim_meta, "recorder_meta": rec_meta, "recorder_stats": rec_stats}


# ---- Command line entry point ----

def main():
    parser = argparse.ArgumentParser(description="Run one BCI noise study recording.")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g. pilot01)")
    parser.add_argument("--condition", required=True,
                        help="Condition tag (control/chewing/emi/audio)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Stimulus duration in seconds (default 300)")
    parser.add_argument("--practice", action="store_true",
                        help="Run as practice (no files saved)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Fullscreen mode (default windowed)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for sequence generation (optional)")
    args = parser.parse_args()

    run_recording(
        subject_id=args.subject,
        condition=args.condition,
        duration_s=args.duration,
        practice=args.practice,
        windowed=not args.fullscreen,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
