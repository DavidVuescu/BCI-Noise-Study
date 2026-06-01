"""Standalone test of the 3-sub-block stimulus runner. No headset, no recorder.

Runs a SHORT block (90 flashes total → 30 per sub-block, ~7s each) in
windowed mode. Tests rest gates, count prompts, rotating borders, and the
new marker columns. Press ESC to abort.
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
        n_flashes=90,                 # 30 per sub-block
        target_cells=[4, 0, 8],       # a test permutation
        practice=False,
        windowed=True,
        seed=42,
    )
    print("\n=== STIMULUS TEST SUMMARY ===")
    print(f"  flashes delivered: {meta['n_flashes_delivered']} / {meta['n_flashes_planned']}")
    print(f"  sub-blocks: {meta['n_sub_blocks']}  order: {meta['target_cells_order']}")
    print(f"  late frames: {meta['late_frames']} "
          f"({100*meta['late_frames']/max(meta['total_frames'],1):.2f}%)")
    print(f"  aborted: {meta['aborted']}")
    print(f"\n  Reported counts per sub-block:")
    for rc in meta["reported_counts"]:
        print(f"    block {rc['sub_block_index']} (target cell {rc['target_cell']}): "
              f"you reported {rc['reported_count']}, expected {rc['expected_count']}")

if __name__ == "__main__":
    main()
