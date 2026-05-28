"""Sanity test for the flash sequence generator.

Generates a typical sequence and verifies the constraints hold and the
oddball ratio is close to expected. No hardware needed.
"""
import random
from src.sequence import generate_sequence, sequence_stats


def main():
    # Match the planned paradigm: 3x3 grid, center target, 60 faces.
    # 1300 flashes = ~5 min at our planned 233ms-per-flash rate.
    seed = 42  # Deterministic for testing
    events = generate_sequence(
        n_flashes=1300,
        n_cells=9,
        target_cell=4,
        n_faces=60,
        rng=random.Random(seed),
    )

    stats = sequence_stats(events, n_cells=9)

    print(f"=== SEQUENCE GENERATOR TEST (seed={seed}) ===\n")
    print(f"  n_flashes:           {stats['n_flashes']}")
    print(f"  target_count:        {stats['target_count']}")
    print(f"  target_ratio:        {stats['target_ratio']:.4f}")
    print(f"  expected ratio:      {stats['expected_target_ratio']:.4f}  (1/9)")
    print(f"  immediate_repeats:   {stats['immediate_repeats']}  (must be 0)")
    print(f"  per_cell range:      {stats['min_per_cell']} - {stats['max_per_cell']}")
    print(f"  per_cell counts:     {stats['per_cell_counts']}")

    # Hard assertions: these MUST hold or the generator is broken.
    assert stats["immediate_repeats"] == 0, "Generator violated no-repeat constraint"
    assert abs(stats["target_ratio"] - 1/9) < 0.02, "Target ratio drifted too far from 1/9"

    # Show the first 10 events so you can eyeball the structure
    print(f"\n  First 10 events:")
    for e in events[:10]:
        marker = " <-- TARGET" if e["is_target"] else ""
        print(f"    seq={e['seq']:4d}  cell={e['cell']}  face={e['face_id']:02d}{marker}")

    print(f"\n  ✓ All assertions passed.")


if __name__ == "__main__":
    main()
