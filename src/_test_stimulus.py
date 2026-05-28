"""Standalone test of the stimulus runner. No headset, no recorder.

Runs a SHORT block (60 flashes, ~15 seconds) in windowed mode so you can
visually verify the paradigm and check the marker CSV. Press ESC to abort.
"""
from src.config import load_config
from src.stimulus import run_session


def main():
    cfg = load_config()
    meta = run_session(
        config=cfg,
        output_dir="data/raw",
        subject_id="stimtest",
        condition="dev",
        n_flashes=60,         # short — about 14 seconds
        practice=False,        # write the files so we can inspect them
        windowed=True,         # don't go fullscreen during dev
        seed=42,
    )

    print("\n=== STIMULUS TEST SUMMARY ===")
    print(f"  flashes delivered: {meta['n_flashes_delivered']} / {meta['n_flashes_planned']}")
    print(f"  expected target count (to be reported by subject): "
          f"{meta['expected_target_count']}")
    print(f"  total frames: {meta['total_frames']}")
    print(f"  late frames: {meta['late_frames']}  "
          f"({100*meta['late_frames']/max(meta['total_frames'],1):.2f}%)")
    print(f"  duration: {meta['duration_s']:.2f}s")
    print(f"  aborted: {meta['aborted']}")
    print(f"\n  Files written:")
    print(f"    data/raw/sub-stimtest_cond-dev_markers.csv")
    print(f"    data/raw/sub-stimtest_cond-dev_session.json")


if __name__ == "__main__":
    main()
