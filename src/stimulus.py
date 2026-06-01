"""
Pygame stimulus runner for the P300 face-oddball paradigm.

THREE-SUB-BLOCK VERSION. Each recording is divided into three sub-blocks,
each with its own fixed target cell drawn from a study-wide set (default
{0,4,8}). The target order is a per-(subject,condition) permutation passed
in by the caller. Between sub-blocks, a rest/acknowledge gate updates the
target border and waits for the subject to press SPACE. After each
sub-block, the subject is prompted to report their counted number of
target flashes, which is logged.

Timing model (unchanged): frame-locked; markers timestamped with
time.time() immediately after pygame.display.flip() on the FIRST frame of
each flash.

Marker schema (one row per flash):
    wall_time, seq, cell, face_id, is_target,
    sub_block_index, sub_block_target_cell, frame_index
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


# ---- Pygame setup (unchanged) -------------------------------------------

def _init_pygame(windowed: bool, _refresh_hz: int):
    pygame.init()
    pygame.mouse.set_visible(False)
    if windowed:
        size = (1024, 768)
        flags = 0
    else:
        size = (1280, 720)
        flags = pygame.FULLSCREEN | pygame.SCALED
    screen = pygame.display.set_mode(size, flags, vsync=1)
    pygame.display.set_caption("BCI Noise Study - Stimulus")
    return screen, size


def _load_faces(face_dir: Path, n_faces: int) -> list[pygame.Surface]:
    faces = []
    for i in range(1, n_faces + 1):
        path = face_dir / f"face{i:02d}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Missing face image: {path}")
        faces.append(pygame.image.load(str(path)).convert())
    return faces


def _compute_cell_rects(screen_size, grid_rows, grid_cols, cell_size_px, spacing_px=40):
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


def _draw_grid_baseline(screen, cell_rects, target_cell, bg_color, cell_color,
                        target_border_color=(180, 180, 180), target_border_width=2):
    """Draw resting grid: bg + cells + border on the CURRENT target cell.

    NOTE: target_cell is now a parameter that changes per sub-block, not a
    fixed constant. This is the key change that makes the border follow the
    rotating target.
    """
    screen.fill(bg_color)
    for i, rect in enumerate(cell_rects):
        pygame.draw.rect(screen, cell_color, rect)
        if i == target_cell:
            pygame.draw.rect(screen, target_border_color, rect, target_border_width)


# ---- NEW: helper screens ------------------------------------------------

def _check_quit(events) -> bool:
    """Return True if a quit/escape event is present in the event list."""
    for event in events:
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            return True
    return False


def _show_message_and_wait_space(
    screen, cell_rects, target_cell, bg_color, cell_color,
    refresh_hz, lines: list[str], min_wait_s: float,
):
    """Show the grid (with new target border) + instruction text, then block
    until the subject presses SPACE. SPACE is ignored until min_wait_s has
    elapsed, enforcing a minimum rest.

    Returns 'aborted' True/False.

    This is the rest/acknowledge GATE between sub-blocks. The keypress —
    the subject's own action — is what resumes stimulus, which is what
    guarantees attentional re-anchoring (per pre-registration §2).
    """
    font = pygame.font.SysFont(None, 36)
    small = pygame.font.SysFont(None, 28)
    start = time.perf_counter()
    space_allowed_msg_shown = False

    while True:
        events = pygame.event.get()
        if _check_quit(events):
            return True
        elapsed = time.perf_counter() - start
        if elapsed >= min_wait_s:
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return False  # acknowledged, proceed

        # Draw grid with the NEW target border so the subject sees where to look
        _draw_grid_baseline(screen, cell_rects, target_cell, bg_color, cell_color)
        # Overlay instruction text
        y = 60
        for line in lines:
            surf = font.render(line, True, (220, 220, 220))
            screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, y))
            y += 44
        # Footer: only show "press SPACE" once rest minimum has passed
        if elapsed >= min_wait_s:
            footer = small.render("Press SPACE to begin", True, (140, 200, 140))
        else:
            remaining = int(min_wait_s - elapsed) + 1
            footer = small.render(f"Rest... ({remaining}s)", True, (160, 160, 160))
        screen.blit(footer, (screen.get_width() // 2 - footer.get_width() // 2,
                             screen.get_height() - 80))
        pygame.display.flip()
        pygame.time.Clock().tick(refresh_hz)


def _prompt_count(screen, bg_color, refresh_hz) -> tuple[int | None, bool]:
    """Freeze on a prompt; subject types digits, ENTER submits.

    Returns (count, aborted). count is None if aborted or empty submission.
    Typed digits accumulate; BACKSPACE deletes; ENTER submits.
    """
    font = pygame.font.SysFont(None, 40)
    small = pygame.font.SysFont(None, 28)
    typed = ""

    while True:
        events = pygame.event.get()
        if _check_quit(events):
            return None, True
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return (int(typed) if typed else None), False
                elif event.key == pygame.K_BACKSPACE:
                    typed = typed[:-1]
                elif event.unicode.isdigit():
                    typed += event.unicode

        screen.fill(bg_color)
        q = font.render("How many target flashes did you count?", True, (220, 220, 220))
        screen.blit(q, (screen.get_width() // 2 - q.get_width() // 2, 200))
        entry = font.render(typed if typed else "_", True, (140, 200, 140))
        screen.blit(entry, (screen.get_width() // 2 - entry.get_width() // 2, 280))
        hint = small.render("Type the number, then press ENTER", True, (160, 160, 160))
        screen.blit(hint, (screen.get_width() // 2 - hint.get_width() // 2, 360))
        pygame.display.flip()
        pygame.time.Clock().tick(refresh_hz)


# ---- Main runner --------------------------------------------------------

def run_session(
    config: dict,
    output_dir: str | Path,
    subject_id: str,
    condition: str,
    n_flashes: int,
    target_cells: list[int] | None = None,
    practice: bool = False,
    windowed: bool = True,
    seed: int | None = None,
) -> dict:
    """Run one three-sub-block stimulus block; write markers + session JSON.

    Args:
        target_cells: Ordered list of per-sub-block target cells (the
            permutation). If None, falls back to config's sub_blocks.target_cells
            in default order. For practice, a single-element list [center] is
            used regardless (practice is one sub-block, per pre-reg §2).
    """
    cfg_stim = config["stimulus"]
    grid_rows = cfg_stim["grid_rows"]
    grid_cols = cfg_stim["grid_cols"]
    n_cells = grid_rows * grid_cols
    flash_frames = cfg_stim["flash_frames"]
    isi_frames = cfg_stim["isi_frames"]
    cell_size_px = cfg_stim["cell_size_px"]
    bg_color = tuple(cfg_stim["background_color"])
    cell_color = tuple(cfg_stim["cell_color"])
    face_dir = Path(cfg_stim["face_dir"])
    refresh_hz = cfg_stim["monitor_refresh_hz"]
    frame_period_ms = 1000.0 / refresh_hz
    frame_budget_ms = frame_period_ms * 1.5
    sb_cfg = cfg_stim["sub_blocks"]
    rest_min_s = sb_cfg["rest_min_s"]

    # Determine sub-block target cells.
    if practice:
        # Practice is a SINGLE sub-block at center (pre-reg §2).
        target_cells_eff = [cfg_stim["target_cell"]]
    elif target_cells is not None:
        target_cells_eff = list(target_cells)
    else:
        target_cells_eff = list(sb_cfg["target_cells"])

    # Build the sequence across all sub-blocks.
    rng = random.Random(seed)
    sequence = generate_sequence(
        n_flashes_total=n_flashes,
        n_cells=n_cells,
        target_cells=target_cells_eff,
        n_faces=60,
        rng=rng,
    )
    seq_stats = sequence_stats(sequence, n_cells)

    # Group flashes by sub-block so we can run them with gates between.
    flashes_by_block: dict[int, list] = {}
    for ev in sequence:
        flashes_by_block.setdefault(ev["sub_block_index"], []).append(ev)
    n_sub_blocks = len(flashes_by_block)

    # Init pygame
    screen, screen_size = _init_pygame(windowed, refresh_hz)
    cell_rects = _compute_cell_rects(screen_size, grid_rows, grid_cols, cell_size_px)
    faces = _load_faces(face_dir, n_faces=60)
    clock = pygame.time.Clock()

    markers: list[dict] = []
    reported_counts: list[dict] = []   # NEW: per-sub-block reported counts
    late_frames = 0
    aborted = False
    frame_index = 0
    start_time_wall = time.time()
    start_time_iso = datetime.now(timezone.utc).isoformat()

    try:
        for sb_index in sorted(flashes_by_block):
            block_flashes = flashes_by_block[sb_index]
            sb_target = block_flashes[0]["sub_block_target_cell"]

            # ---- REST/ACKNOWLEDGE GATE (between sub-blocks; also before first) ----
            # For the very first sub-block we still show it, so the subject
            # orients to the target cell before stimulus. Wording differs
            # slightly for first vs subsequent, but the gate is the same.
            if sb_index == 0:
                lines = [
                    "Look at the cell with the border.",
                    "Silently count every face that flashes INSIDE it.",
                ]
            else:
                lines = [
                    "New target cell.",
                    "Look at the cell now bordered.",
                    "Count silently every face that flashes inside it.",
                ]
            aborted = _show_message_and_wait_space(
                screen, cell_rects, sb_target, bg_color, cell_color,
                refresh_hz, lines, min_wait_s=(0.5 if sb_index == 0 else rest_min_s),
            )
            if aborted:
                raise KeyboardInterrupt

            # ---- FLASH LOOP for this sub-block (the validated inner loop) ----
            for flash in block_flashes:
                face_surface = faces[flash["face_id"] - 1]
                cell_rect = cell_rects[flash["cell"]]
                face_pos = (
                    cell_rect.x + (cell_rect.width - face_surface.get_width()) // 2,
                    cell_rect.y + (cell_rect.height - face_surface.get_height()) // 2,
                )

                # Flash phase
                for f in range(flash_frames):
                    if _check_quit(pygame.event.get()):
                        aborted = True
                        raise KeyboardInterrupt
                    _draw_grid_baseline(screen, cell_rects, sb_target, bg_color, cell_color)
                    screen.blit(face_surface, face_pos)
                    pygame.display.flip()
                    if f == 0:
                        marker_ts = time.time()
                        markers.append({
                            "wall_time": marker_ts,
                            "seq": flash["seq"],
                            "cell": flash["cell"],
                            "face_id": flash["face_id"],
                            "is_target": flash["is_target"],
                            "sub_block_index": flash["sub_block_index"],
                            "sub_block_target_cell": flash["sub_block_target_cell"],
                            "frame_index": frame_index,
                        })
                    elapsed_ms = clock.tick(refresh_hz)
                    if elapsed_ms > frame_budget_ms:
                        late_frames += 1
                    frame_index += 1

                # ISI phase
                for f in range(isi_frames):
                    if _check_quit(pygame.event.get()):
                        aborted = True
                        raise KeyboardInterrupt
                    _draw_grid_baseline(screen, cell_rects, sb_target, bg_color, cell_color)
                    pygame.display.flip()
                    elapsed_ms = clock.tick(refresh_hz)
                    if elapsed_ms > frame_budget_ms:
                        late_frames += 1
                    frame_index += 1

            # ---- COUNT PROMPT after this sub-block ----
            expected = sum(1 for fl in block_flashes if fl["is_target"])
            reported, aborted = _prompt_count(screen, bg_color, refresh_hz)
            if aborted:
                raise KeyboardInterrupt
            reported_counts.append({
                "sub_block_index": sb_index,
                "target_cell": sb_target,
                "expected_count": expected,
                "reported_count": reported,
            })

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()

    stop_time_wall = time.time()

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
        "n_sub_blocks": n_sub_blocks,
        "target_cells_order": target_cells_eff,
        "reported_counts": reported_counts,
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

    # Files go into a per-subject subfolder: data/raw/sub-<id>/
    output_dir = Path(output_dir) / f"sub-{subject_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject_id}_cond-{condition}"
    with open(output_dir / f"{stem}_markers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "wall_time", "seq", "cell", "face_id", "is_target",
            "sub_block_index", "sub_block_target_cell", "frame_index",
        ])
        writer.writeheader()
        writer.writerows(markers)
    with open(output_dir / f"{stem}_session.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
