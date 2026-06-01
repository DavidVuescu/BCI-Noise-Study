"""
Flash sequence generator for the P300 oddball paradigm.

Generates a sequence structured as THREE sub-blocks, each with its own
fixed target cell drawn from a study-wide set (default {0, 4, 8}). Within
a sub-block, flashes follow the single-cell oddball logic (no immediate
repeat, faces sampled uniformly). The sub-block target rotates per the
permutation passed in, which decorrelates target-cell identity from
retinotopic position and from within-recording temporal drift.

Each flash event is a dict with:
    seq:                   int, 0-indexed flash number across the WHOLE recording
    cell:                  int, 0-8 (grid index, row-major)
    face_id:               int, 1-60 (matches faceNN.jpg)
    is_target:             bool, True iff cell == this sub-block's target
    sub_block_index:       int, 0/1/2 — which sub-block this flash belongs to
    sub_block_target_cell: int — the target cell active during this sub-block

Constraints (unchanged from single-block version, enforced WITHIN each sub-block):
    - No cell flashes twice in immediate succession (refractory confound).
    - Target probability per flash is ~1/n_cells (true oddball ratio).
    - Faces sampled uniformly with replacement from the pool.

The no-repeat constraint resets at sub-block boundaries: the first flash of
sub-block 1 may legitimately be the same cell as the last flash of sub-block
0, because the rest screen + re-anchoring between them breaks any refractory
chaining. (In practice this is harmless either way; we simply don't carry
last_cell across the boundary.)
"""
from __future__ import annotations

import random
from typing import TypedDict


class FlashEvent(TypedDict):
    seq: int
    cell: int
    face_id: int
    is_target: bool
    sub_block_index: int
    sub_block_target_cell: int


def _generate_sub_block(
    start_seq: int,
    n_flashes: int,
    n_cells: int,
    target_cell: int,
    sub_block_index: int,
    n_faces: int,
    rng: random.Random,
) -> list[FlashEvent]:
    """Generate one sub-block's worth of flashes.

    Identical single-cell oddball logic to the original generator, plus
    the two sub-block context fields. seq values continue from start_seq
    so that seq is globally unique across the whole recording.
    """
    events: list[FlashEvent] = []
    last_cell: int | None = None

    for i in range(n_flashes):
        # Pick a cell, rejecting immediate repeat (within this sub-block).
        while True:
            cell = rng.randrange(n_cells)
            if cell != last_cell:
                break

        face_id = rng.randrange(1, n_faces + 1)

        events.append({
            "seq": start_seq + i,
            "cell": cell,
            "face_id": face_id,
            "is_target": cell == target_cell,
            "sub_block_index": sub_block_index,
            "sub_block_target_cell": target_cell,
        })
        last_cell = cell

    return events


def generate_sequence(
    n_flashes_total: int,
    n_cells: int,
    target_cells: list[int],
    n_faces: int,
    rng: random.Random | None = None,
) -> list[FlashEvent]:
    """Generate a full 3-sub-block flash sequence.

    Args:
        n_flashes_total: Total flashes across all sub-blocks. Split as evenly
            as possible across len(target_cells) sub-blocks (remainder goes
            to the earlier sub-blocks).
        n_cells:        Grid size (9 for 3x3).
        target_cells:   Ordered list of target cells, one per sub-block. The
            ORDER is the per-(subject,condition) permutation; this function
            takes it as given and does not shuffle it. Length determines the
            number of sub-blocks (3 in this study).
        n_faces:        Size of face pool (60).
        rng:            Optional seeded Random for reproducibility.

    Returns:
        List of FlashEvent dicts, length == n_flashes_total, with globally
        unique seq values and per-flash sub-block context.
    """
    if rng is None:
        rng = random.Random()

    n_sub_blocks = len(target_cells)
    if n_sub_blocks < 1:
        raise ValueError("Need at least one target cell / sub-block")
    for tc in target_cells:
        if tc < 0 or tc >= n_cells:
            raise ValueError(f"target cell {tc} out of range [0, {n_cells})")
    if n_cells < 2:
        raise ValueError("Need at least 2 cells to avoid trivial repeats")

    # Split total flashes across sub-blocks as evenly as possible.
    # Remainder distributed to the earliest sub-blocks (difference of at
    # most 1 flash between sub-blocks — negligible for analysis).
    base = n_flashes_total // n_sub_blocks
    remainder = n_flashes_total % n_sub_blocks
    per_block = [base + (1 if i < remainder else 0) for i in range(n_sub_blocks)]

    events: list[FlashEvent] = []
    seq_cursor = 0
    for sb_index, (target_cell, n_fl) in enumerate(zip(target_cells, per_block)):
        block_events = _generate_sub_block(
            start_seq=seq_cursor,
            n_flashes=n_fl,
            n_cells=n_cells,
            target_cell=target_cell,
            sub_block_index=sb_index,
            n_faces=n_faces,
            rng=rng,
        )
        events.extend(block_events)
        seq_cursor += n_fl

    return events


def sequence_stats(events: list[FlashEvent], n_cells: int) -> dict:
    """Summary statistics for a generated sequence, including per-sub-block.

    Used for sanity-checking and stimulus startup logging.
    """
    n = len(events)
    if n == 0:
        return {"n_flashes": 0}

    target_count = sum(1 for e in events if e["is_target"])
    per_cell_counts = [0] * n_cells
    for e in events:
        per_cell_counts[e["cell"]] += 1

    # Per-sub-block breakdown
    sub_blocks: dict[int, dict] = {}
    for e in events:
        sb = e["sub_block_index"]
        if sb not in sub_blocks:
            sub_blocks[sb] = {
                "target_cell": e["sub_block_target_cell"],
                "n_flashes": 0,
                "target_count": 0,
            }
        sub_blocks[sb]["n_flashes"] += 1
        if e["is_target"]:
            sub_blocks[sb]["target_count"] += 1

    # No-repeat check, scoped WITHIN sub-blocks (we don't count a same-cell
    # flash across a sub-block boundary as a violation).
    repeats = 0
    for i in range(1, n):
        same_cell = events[i]["cell"] == events[i - 1]["cell"]
        same_block = events[i]["sub_block_index"] == events[i - 1]["sub_block_index"]
        if same_cell and same_block:
            repeats += 1

    return {
        "n_flashes": n,
        "target_count": target_count,
        "target_ratio": target_count / n,
        "expected_target_ratio": 1.0 / n_cells,
        "per_cell_counts": per_cell_counts,
        "immediate_repeats_within_block": repeats,  # must be 0
        "min_per_cell": min(per_cell_counts),
        "max_per_cell": max(per_cell_counts),
        "n_sub_blocks": len(sub_blocks),
        "sub_blocks": sub_blocks,
    }
