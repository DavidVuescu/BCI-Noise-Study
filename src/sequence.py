"""
Flash sequence generator for the P300 oddball paradigm.

Produces a pre-computed list of flash events for a recording. Pure function,
no pygame, no I/O — easy to test in isolation.

Each flash event is a dict with:
    seq:         int, 0-indexed flash number within the recording
    cell:        int, 0-8 (grid index, row-major)
    face_id:     int, 1-60 (matches faceNN.jpg)
    is_target:   bool, True iff cell == target_cell

Constraints enforced during generation:
    - No cell flashes twice in immediate succession (refractory confound).
    - Target probability per flash is 1/n_cells (true oddball ratio).
    - Faces are sampled uniformly with replacement from the pool.
"""
from __future__ import annotations

import random
from typing import TypedDict


class FlashEvent(TypedDict):
    seq: int
    cell: int
    face_id: int
    is_target: bool


def generate_sequence(
    n_flashes: int,
    n_cells: int,
    target_cell: int,
    n_faces: int,
    rng: random.Random | None = None,
) -> list[FlashEvent]:
    """Generate a flash sequence with no-immediate-repeat constraint.

    Args:
        n_flashes:   Total number of flashes to generate.
        n_cells:     Grid size (9 for 3x3).
        target_cell: 0-indexed cell that counts as 'target' (4 for center of 3x3).
        n_faces:     Size of face pool (60 for faceNN.jpg, NN in 01..60).
        rng:         Optional seeded Random instance for reproducibility.
                     Defaults to random.Random() (non-deterministic).

    Returns:
        List of FlashEvent dicts of length n_flashes.

    Why no-immediate-repeat: if the same cell flashes twice in a row, the
    second flash's evoked response is suppressed by neural refractoriness
    of the first. This contaminates ERP averages with a systematic
    amplitude reduction that has nothing to do with attention or noise.
    Standard oddball protocols exclude immediate repeats.

    Why sample faces with replacement: with 60 faces and ~1300 flashes per
    block, every face will be shown ~22 times on average. Sampling with
    replacement is simpler and gives proper face novelty per flash without
    artifacts from a forced shuffle.
    """
    if rng is None:
        rng = random.Random()

    if target_cell < 0 or target_cell >= n_cells:
        raise ValueError(f"target_cell {target_cell} out of range [0, {n_cells})")
    if n_cells < 2:
        raise ValueError("Need at least 2 cells to avoid trivial repeats")

    events: list[FlashEvent] = []
    last_cell: int | None = None

    for seq in range(n_flashes):
        # Pick a cell, rejecting immediate repeat.
        # With n_cells=9, the rejection rate is 1/9, so this loop terminates
        # quickly in expectation (~1.13 picks per flash on average).
        while True:
            cell = rng.randrange(n_cells)
            if cell != last_cell:
                break

        # Pick a face uniformly from the pool. face_id is 1-indexed to
        # match the file naming convention (face01.jpg .. faceNN.jpg).
        face_id = rng.randrange(1, n_faces + 1)

        events.append({
            "seq": seq,
            "cell": cell,
            "face_id": face_id,
            "is_target": cell == target_cell,
        })
        last_cell = cell

    return events


def sequence_stats(events: list[FlashEvent], n_cells: int) -> dict:
    """Compute summary statistics for a generated sequence.

    Useful for sanity-checking that constraints held and the oddball
    ratio came out as expected. Used in tests and in stimulus startup
    logging.
    """
    n = len(events)
    if n == 0:
        return {"n_flashes": 0}

    target_count = sum(1 for e in events if e["is_target"])
    per_cell_counts = [0] * n_cells
    for e in events:
        per_cell_counts[e["cell"]] += 1

    # Verify no-immediate-repeat constraint held
    repeats = sum(
        1 for i in range(1, n)
        if events[i]["cell"] == events[i - 1]["cell"]
    )

    return {
        "n_flashes": n,
        "target_count": target_count,
        "target_ratio": target_count / n,
        "expected_target_ratio": 1.0 / n_cells,
        "per_cell_counts": per_cell_counts,
        "immediate_repeats": repeats,  # should always be 0
        "min_per_cell": min(per_cell_counts),
        "max_per_cell": max(per_cell_counts),
    }
