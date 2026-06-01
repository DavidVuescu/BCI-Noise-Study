"""Sanity test for the 3-sub-block flash sequence generator. No hardware."""
import random
from src.sequence import generate_sequence, sequence_stats


def main():
    seed = 42
    target_cells = [0, 4, 8]  # a fixed permutation for this test
    events = generate_sequence(
        n_flashes_total=2570,        # ~control recording size
        n_cells=9,
        target_cells=target_cells,
        n_faces=60,
        rng=random.Random(seed),
    )

    stats = sequence_stats(events, n_cells=9)

    print(f"=== 3-SUB-BLOCK SEQUENCE TEST (seed={seed}) ===\n")
    print(f"  n_flashes:                  {stats['n_flashes']}")
    print(f"  n_sub_blocks:               {stats['n_sub_blocks']}")
    print(f"  target_count (total):       {stats['target_count']}")
    print(f"  target_ratio:               {stats['target_ratio']:.4f}  (expect ~0.122)")
    print(f"  immediate_repeats_in_block: {stats['immediate_repeats_within_block']}  (must be 0)")
    print(f"  per_cell range:             {stats['min_per_cell']} - {stats['max_per_cell']}")
    print(f"\n  Per sub-block:")
    for sb_idx in sorted(stats["sub_blocks"]):
        sb = stats["sub_blocks"][sb_idx]
        print(f"    block {sb_idx}: target=cell{sb['target_cell']}  "
              f"flashes={sb['n_flashes']}  targets={sb['target_count']} "
              f"({sb['target_count']/sb['n_flashes']:.3f})")

    # Hard assertions
    assert stats["immediate_repeats_within_block"] == 0, "No-repeat constraint violated"
    assert stats["n_sub_blocks"] == 3, "Expected exactly 3 sub-blocks"
    assert abs(stats["target_ratio"] - 1/9) < 0.03, "Target ratio drifted too far"

    # Verify each sub-block's targets land ONLY on that block's target cell
    for e in events:
        if e["is_target"]:
            assert e["cell"] == e["sub_block_target_cell"], \
                f"Target flash at seq={e['seq']} has cell != sub_block_target_cell"

    # Verify seq is globally unique and contiguous
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(events))), "seq values not contiguous 0..n-1"

    # Verify sub-block boundaries are contiguous and ordered
    boundaries = [e["sub_block_index"] for e in events]
    assert boundaries == sorted(boundaries), "Sub-blocks not in contiguous order"

    print(f"\n  ✓ All assertions passed.")


if __name__ == "__main__":
    main()
