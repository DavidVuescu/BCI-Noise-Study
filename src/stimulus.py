"""
Pygame stimulus runner for the P300 face-oddball paradigm.

Renders a 3x3 grid of dark cells. On each flash, a randomly-chosen face
appears in a randomly-chosen cell for N frames, then blanks for M frames.
The target cell (center) has a persistent thin border so the subject
knows where to fixate.

Timing model:
    - Frame-locked: stimulus durations are integer multiples of frame period.
    - Markers are timestamped with time.time() IMMEDIATELY AFTER
      pygame.display.flip() returns (which blocks until vsync on a normal
      vsync-enabled setup). This is the moment closest to physical photon
      onset that we can measure without external hardware.
    - Per-frame timing is monitored. Frames exceeding budget are logged
      as 'late frames' in session metadata.

Marker schema (one row per flash, written to CSV):
    wall_time, seq, cell, face_id, is_target, frame_index

The marker CSV is paired with a session JSON containing run-level info:
    practice flag, expected target count, total flashes, late frame count,
    start/stop wall times, config snapshot.
"""
from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pygame

from src.sequence import generate_sequence, sequence_stats


# ---- Pygame setup --------------------------------------------------------

def _init_pygame(windowed: bool, monitor_refresh_hz: int):
    """Initialize pygame and create the display surface.

    Args:
        windowed: If True, create a windowed display (for development).
                  If False, fullscreen (for real recordings).
        monitor_refresh_hz: Used to size windowed mode; fullscreen takes
                            the native resolution.

    Returns:
        (screen, screen_size) where screen is the pygame Surface and
        screen_size is (width, height).
    """
    pygame.init()
    pygame.mouse.set_visible(False)

    if windowed:
        # Reasonable windowed size for development
        size = (1024, 768)
        flags = 0
    else:
        # Fullscreen on the primary display. SCALED gives us a logical
        # surface at the requested size, scaled to native resolution by
        # pygame — avoids resolution mismatch surprises.
        size = (1280, 720)
        flags = pygame.FULLSCREEN | pygame.SCALED

    # vsync=1 asks the driver to sync display.flip() to vertical blank.
    # This is essential for our timing model — without it, flip() returns
    # immediately and our timestamps are decoupled from actual photons.
    screen = pygame.display.set_mode(size, flags, vsync=1)
    pygame.display.set_caption("BCI Noise Study - Stimulus")

    return screen, size


def _load_faces(face_dir: Path, n_faces: int) -> list[pygame.Surface]:
    """Load all face images into pygame Surfaces, indexed 1..n_faces.

    Returns a list where index i corresponds to faceNN.jpg with NN = i+1.
    (i.e. faces[0] is face01.jpg). Surfaces are converted to the display
    format up front so blitting is fast.
    """
    faces = []
    for i in range(1, n_faces + 1):
        path = face_dir / f"face{i:02d}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing face image: {path}")
        img = pygame.image.load(str(path))
        # .convert() matches the display's pixel format for fast blitting.
        # Skipping this can cost milliseconds per blit, which adds up in a
        # frame-tight loop.
        img = img.convert()
        faces.append(img)
    return faces


def _compute_cell_rects(
    screen_size: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    cell_size_px: int,
    spacing_px: int = 40,
) -> list[pygame.Rect]:
    """Compute the pixel rectangle for each grid cell, row-major.

    Returns a list of length grid_rows*grid_cols. Index = row*cols + col.
    """
    grid_w = grid_cols * cell_size_px + (grid_cols - 1) * spacing_px
    grid_h = grid_rows * cell_size_px + (grid_rows - 1) * spacing_px
    x0 = (screen_size[0] - grid_w) // 2
    y0 = (screen_size[1] - grid_h) // 2

    rects = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = x0 + col * (cell_size_px + spacing_px)
            y = y0 + row * (cell_size_px + spacing_px)
            rects.append(pygame.Rect(x, y, cell_size_px, cell_size_px))
    return rects


# ---- Drawing primitives --------------------------------------------------

def _draw_grid_baseline(
    screen: pygame.Surface,
    cell_rects: list[pygame.Rect],
    target_cell: int,
    bg_color: tuple[int, int, int],
    cell_color: tuple[int, int, int],
    target_border_color: tuple[int, int, int] = (180, 180, 180),
    target_border_width: int = 2,
) -> None:
    """Draw the resting grid: background + 9 dark cells + target border.

    Called every frame as the base layer. Flash overlay (the face) is
    drawn on top of this.
    """
    screen.fill(bg_color)
    for i, rect in enumerate(cell_rects):
        pygame.draw.rect(screen, cell_color, rect)
        if i == target_cell:
            pygame.draw.rect(screen, target_border_color, rect, target_border_width)


# ---- Main runner ---------------------------------------------------------

def run_session(
    config: dict,
    output_dir: str | Path,
    subject_id: str,
    condition: str,
    n_flashes: int,
    practice: bool = False,
    windowed: bool = True,
    seed: int | None = None,
) -> dict:
    """Run one stimulus block and write markers + session metadata.

    Args:
        config: Loaded config dict (see config.yaml).
        output_dir: Where to write markers + session JSON. Ignored if practice.
        subject_id, condition: Used in output filenames.
        n_flashes: Total flashes for this block.
        practice: If True, run the paradigm but write nothing to disk.
        windowed: If True, windowed mode (for dev). False for real recordings.
        seed: Optional seed for sequence generation. If None, non-deterministic.

    Returns:
        Session metadata dict (also written to JSON unless practice).
    """
    cfg_stim = config["stimulus"]
    cfg_dev = config["device"]

    grid_rows = cfg_stim["grid_rows"]
    grid_cols = cfg_stim["grid_cols"]
    n_cells = grid_rows * grid_cols
    target_cell = cfg_stim["target_cell"]
    flash_frames = cfg_stim["flash_frames"]
    isi_frames = cfg_stim["isi_frames"]
    cell_size_px = cfg_stim["cell_size_px"]
    bg_color = tuple(cfg_stim["background_color"])
    cell_color = tuple(cfg_stim["cell_color"])
    face_dir = Path(cfg_stim["face_dir"])
    refresh_hz = cfg_stim["monitor_refresh_hz"]
    frame_period_ms = 1000.0 / refresh_hz
    frame_budget_ms = frame_period_ms * 1.5  # late if frame takes >1.5x budget

    # Pre-compute the sequence. Doing this up front means the per-frame
    # loop does ZERO decision-making — it just plays a script. Predictable
    # timing, no surprise latency from RNG calls in the hot path.
    rng = random.Random(seed)
    sequence = generate_sequence(
        n_flashes=n_flashes,
        n_cells=n_cells,
        target_cell=target_cell,
        n_faces=60,
        rng=rng,
    )
    seq_stats = sequence_stats(sequence, n_cells)
    expected_target_count = seq_stats["target_count"]

    # Initialize pygame
    screen, screen_size = _init_pygame(windowed, refresh_hz)
    cell_rects = _compute_cell_rects(screen_size, grid_rows, grid_cols, cell_size_px)
    faces = _load_faces(face_dir, n_faces=60)
    clock = pygame.time.Clock()

    # State for the run
    markers: list[dict] = []
    late_frames = 0
    aborted = False
    frame_index = 0  # monotonic frame counter for the whole session
    start_time_wall = time.time()
    start_time_iso = datetime.now(timezone.utc).isoformat()

    try:
        # Pre-flight: 1 second of just the grid so the subject orients
        # to the target cell before flashes start.
        for _ in range(refresh_hz):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    aborted = True
                    raise KeyboardInterrupt
            _draw_grid_baseline(screen, cell_rects, target_cell, bg_color, cell_color)
            pygame.display.flip()
            clock.tick(refresh_hz)
            frame_index += 1

        # Main paradigm loop
        for flash in sequence:
            # ---- FLASH PHASE: face visible for flash_frames frames ----
            face_surface = faces[flash["face_id"] - 1]
            cell_rect = cell_rects[flash["cell"]]
            # Center the face within its cell (faces are 100x100, cells are 100x100)
            face_pos = (
                cell_rect.x + (cell_rect.width - face_surface.get_width()) // 2,
                cell_rect.y + (cell_rect.height - face_surface.get_height()) // 2,
            )

            for f in range(flash_frames):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        aborted = True
                        raise KeyboardInterrupt

                _draw_grid_baseline(screen, cell_rects, target_cell, bg_color, cell_color)
                screen.blit(face_surface, face_pos)

                pygame.display.flip()
                # Marker timestamp IMMEDIATELY after flip() on the FIRST frame
                # of the flash (the actual stimulus onset).
                if f == 0:
                    marker_ts = time.time()
                    markers.append({
                        "wall_time": marker_ts,
                        "seq": flash["seq"],
                        "cell": flash["cell"],
                        "face_id": flash["face_id"],
                        "is_target": flash["is_target"],
                        "frame_index": frame_index,
                    })

                elapsed_ms = clock.tick(refresh_hz)
                if elapsed_ms > frame_budget_ms:
                    late_frames += 1
                frame_index += 1

            # ---- ISI PHASE: blank grid for isi_frames frames ----
            for f in range(isi_frames):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        aborted = True
                        raise KeyboardInterrupt

                _draw_grid_baseline(screen, cell_rects, target_cell, bg_color, cell_color)
                pygame.display.flip()
                elapsed_ms = clock.tick(refresh_hz)
                if elapsed_ms > frame_budget_ms:
                    late_frames += 1
                frame_index += 1

    except KeyboardInterrupt:
        # ESC or window close — clean exit, partial data still gets saved
        # below (so we can debug interrupted runs).
        pass
    finally:
        pygame.quit()

    stop_time_wall = time.time()

    # Build session metadata
    meta = {
        "subject_id": subject_id,
        "condition": condition,
        "practice": practice,
        "aborted": aborted,
        "start_time_iso_utc": start_time_iso,
        "start_time_unix": start_time_wall,
        "stop_time_unix": stop_time_wall,
        "duration_s": stop_time_wall - start_time_wall,
        "n_flashes_planned": n_flashes,
        "n_flashes_delivered": len(markers),
        "expected_target_count": expected_target_count,
        "late_frames": late_frames,
        "total_frames": frame_index,
        "monitor_refresh_hz": refresh_hz,
        "flash_ms": flash_frames * frame_period_ms,
        "isi_ms": isi_frames * frame_period_ms,
        "sequence_stats": seq_stats,
        "seed": seed,
        "config_snapshot": config,
    }

    if practice:
        print(f"[practice] {len(markers)} flashes, {late_frames} late frames. "
              "Nothing saved.")
        return meta

    # Persist markers + session metadata
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject_id}_cond-{condition}"
    markers_path = output_dir / f"{stem}_markers.csv"
    session_path = output_dir / f"{stem}_session.json"

    with open(markers_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wall_time", "seq", "cell", "face_id", "is_target", "frame_index"],
        )
        writer.writeheader()
        writer.writerows(markers)

    with open(session_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
