"""Smoke test for the loader. Loads all four pilot recordings, prints a
summary of each, and runs a few sanity checks on alignment."""
from analysis.loader import load_recording


def main():
    subject = "pilot-self-day0"
    conditions = ["control", "chewing", "emi", "acoustic"]

    for cond in conditions:
        print(f"\n=== {cond.upper()} ===")
        rec = load_recording(subject, cond)
        print(f"  duration:      {rec.duration_s:.2f}s")
        print(f"  n samples:     {len(rec.raw.times)}")
        print(f"  n markers:     {len(rec.markers)}")
        print(f"  n target:      {rec.markers['is_target'].sum()}")
        print(f"  sub-block targets: "
              f"{rec.meta['session']['target_cells_order']}")
        print(f"  sample range of markers: "
              f"{rec.markers['sample'].min()} to {rec.markers['sample'].max()} "
              f"(EEG has {len(rec.raw.times)} samples)")
        # Sanity: every marker should fall within the EEG range
        max_sample = len(rec.raw.times) - 1
        assert (rec.markers["sample"] >= 0).all(), \
            f"{cond}: some markers have negative sample index"
        assert (rec.markers["sample"] <= max_sample).all(), \
            f"{cond}: some markers exceed EEG length"
        # Sanity: counts reported should match expected (from your discipline)
        reported = rec.meta["session"]["reported_counts"]
        for sb in reported:
            expected = sb["expected_count"]
            actually = sb["reported_count"]
            print(f"    sub-block {sb['sub_block_index']} "
                  f"(target={sb['target_cell']}): "
                  f"reported {actually} vs expected {expected}")

    print("\n  ✓ All loaders ran without alignment errors.")


if __name__ == "__main__":
    main()
