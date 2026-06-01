"""Smoke test for preprocessing. Runs all four pilot recordings through the
pipeline and prints rejection summaries."""
from analysis.loader import load_recording
from analysis.preprocess import preprocess_recording


def main():
    subject = "pilot-self-day0"
    conditions = ["control", "chewing", "emi", "acoustic"]

    for cond in conditions:
        print(f"\n=== {cond.upper()} ===")
        rec = load_recording(subject, cond)
        result = preprocess_recording(rec, save=True)
        log = result.rejection_log

        print(f"  planned:              {log['n_planned']}")
        print(f"  boundary dropped:     {log['n_boundary_dropped']}")
        print(f"  amplitude dropped:    {log['n_amplitude_dropped']}")
        print(f"  kept:                 {log['n_kept']}")
        print(f"  total rejection rate: {log['rejection_rate']*100:.1f}%")
        print(f"  per sub-block (planned → kept):")
        for sb_idx, sb in log["per_subblock"].items():
            print(f"    block {sb_idx}: {sb['planned']:4d} → {sb['kept']:4d} "
                  f"({sb['rejection_rate']*100:.1f}% rejected)")

        # Sanity: target/nontarget counts among kept
        meta = result.epochs.metadata
        n_target_kept = int(meta["is_target"].sum())
        n_nontarget_kept = len(meta) - n_target_kept
        print(f"  kept targets:         {n_target_kept}")
        print(f"  kept non-targets:     {n_nontarget_kept}")

        # Verify §4 exclusion criterion: 20% rejection threshold
        if log["rejection_rate"] > 0.20:
            print(f"  ⚠  Exceeds 20% rejection threshold (registered exclusion criterion)")
        else:
            print(f"  ✓  Within 20% rejection threshold")


if __name__ == "__main__":
    main()
