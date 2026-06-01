"""
Generate the per-subject randomization table for the BCI Noise Study.

Produces protocol/order_assignments/order_assignments.csv with one row per (subject, recording),
columns matching pre-registration §4:
    subject_id, condition_order, condition,
    sub_block_1_target, sub_block_2_target, sub_block_3_target

Randomization layers:
1. Condition order per subject: 4x4 Latin square cycled through the cohort,
   guaranteeing balanced position counts across all 4 conditions.
2. Sub-block target-cell permutation per (subject, condition): uniform random
   over the 6 permutations of {0, 4, 8}, seeded reproducibly per (subject, condition).

Both layers derive from a single MASTER_SEED, hardcoded in this script and
documented in the registered protocol. Regenerating the CSV from this seed
produces byte-identical output, which is the auditability property pre-
registration requires.
"""
from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path
import hashlib


# ---- Master seed -------------------------------------------------------
# Hardcoded so the CSV is reproducible. Change requires re-running and
# logging as a deviation in DEVIATIONS.md. DO NOT change after data
# collection has started for any subject in the cohort.
MASTER_SEED = 20260601  # date-derived for documentation; arbitrary otherwise

# ---- Study parameters --------------------------------------------------
CONDITIONS = ["control", "chewing", "emi", "acoustic"]
TARGET_CELLS = [0, 4, 8]
N_SUBJECTS = 60
SUBJECT_ID_PREFIX = "sub-"
SUBJECT_ID_WIDTH = 2  # sub-01, sub-02, ... sub-50

OUTPUT_PATH = Path("protocol/order_assignments/order_assignments.csv")


def build_latin_square(items: list[str], seed: int) -> list[list[str]]:
    """Build a 4x4 Latin square over `items`, with row order shuffled by seed.

    A Latin square here is a square arrangement where each item appears
    exactly once in each row and exactly once in each column. We construct
    it by cyclic shifts of a base ordering, then shuffle the row order
    deterministically for additional randomization.

    Returns: list of `len(items)` rows, each a permutation of items.
    """
    n = len(items)
    # Cyclic-shift construction: row i is items shifted left by i positions.
    # This guarantees the Latin-square property by construction.
    base_rows = [items[i:] + items[:i] for i in range(n)]
    # Shuffle the row order so position-i isn't always the same cyclic shift.
    rng = random.Random(seed)
    rng.shuffle(base_rows)
    return base_rows


def derive_subject_seed(master_seed: int, subject_index: int, layer: str) -> int:
    """Derive a deterministic per-subject seed for a given randomization layer.

    Uses hashlib (not Python's built-in hash) because the built-in hash for
    strings is salted per-process — each Python invocation produces different
    values, which would break reproducibility. hashlib is deterministic across
    processes, machines, and Python versions.

    The result is the same every time this function is called with the same
    arguments, which is the audit-trail property pre-registration requires.
    """
    key = f"{master_seed}|{subject_index}|{layer}|salt_v1".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    # Take the first 8 bytes of the SHA-256 digest as an unsigned 64-bit int.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build the Latin square for condition ordering (Layer 1).
    # Seed only with the master seed; the same square serves the whole cohort.
    latin_square = build_latin_square(CONDITIONS, seed=MASTER_SEED)
    print(f"Latin square ({len(latin_square)} rows over {CONDITIONS}):")
    for row in latin_square:
        print(f"  {row}")

    # Pre-compute all 6 permutations of {0, 4, 8} for sub-block targets.
    all_sub_block_perms = list(itertools.permutations(TARGET_CELLS))

    rows = []
    for s_idx in range(N_SUBJECTS):
        subject_id = f"{SUBJECT_ID_PREFIX}{s_idx + 1:0{SUBJECT_ID_WIDTH}d}"

        # Layer 1: which Latin-square row does this subject use?
        # Cycle through the square rows; subject 1 -> row 0, subject 5 -> row 0, etc.
        square_row = latin_square[s_idx % len(latin_square)]

        # Layer 2: per-condition sub-block permutation.
        # Independent seed per (subject, condition).
        for cond_position, condition in enumerate(square_row, start=1):
            sb_seed = derive_subject_seed(MASTER_SEED, s_idx, f"subblock_{condition}")
            sb_rng = random.Random(sb_seed)
            sb_perm = sb_rng.choice(all_sub_block_perms)

            rows.append({
                "subject_id": subject_id,
                "condition_order": cond_position,
                "condition": condition,
                "sub_block_1_target": sb_perm[0],
                "sub_block_2_target": sb_perm[1],
                "sub_block_3_target": sb_perm[2],
            })

    # Write the CSV
    fieldnames = [
        "subject_id", "condition_order", "condition",
        "sub_block_1_target", "sub_block_2_target", "sub_block_3_target",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"  ({N_SUBJECTS} subjects x {len(CONDITIONS)} conditions = "
          f"{N_SUBJECTS * len(CONDITIONS)} recordings)")

    # ---- Sanity checks (informational, don't gate the write) -----------
    # 1. Every subject has all 4 conditions exactly once
    from collections import Counter
    subj_conds = {}
    for r in rows:
        subj_conds.setdefault(r["subject_id"], []).append(r["condition"])
    for sid, conds in subj_conds.items():
        assert sorted(conds) == sorted(CONDITIONS), \
            f"{sid} has wrong condition set: {conds}"

    # 2. Condition-in-position balance across cohort
    pos_counts = Counter((r["condition_order"], r["condition"]) for r in rows)
    print(f"\nCondition × position counts (each cell should be ~{N_SUBJECTS // 4}):")
    print(f"  {'pos':>4}  " + "  ".join(f"{c:>9}" for c in CONDITIONS))
    for pos in range(1, len(CONDITIONS) + 1):
        line = f"  {pos:>4}  " + "  ".join(
            f"{pos_counts.get((pos, c), 0):>9}" for c in CONDITIONS
        )
        print(line)

    # 3. Sub-block permutation distribution
    perm_counts = Counter(
        (r["sub_block_1_target"], r["sub_block_2_target"], r["sub_block_3_target"])
        for r in rows
    )
    print(f"\nSub-block permutation counts (each should be ~{len(rows) // 6}):")
    for perm, count in sorted(perm_counts.items()):
        print(f"  {perm}: {count}")

    # 4. Show the first few rows for a sanity eyeball
    print(f"\nFirst 8 rows of the CSV:")
    print(f"  {'subject':>8}  {'pos':>3}  {'condition':>10}  blocks")
    for r in rows[:8]:
        print(f"  {r['subject_id']:>8}  {r['condition_order']:>3}  "
              f"{r['condition']:>10}  "
              f"[{r['sub_block_1_target']}, {r['sub_block_2_target']}, {r['sub_block_3_target']}]")


if __name__ == "__main__":
    main()
