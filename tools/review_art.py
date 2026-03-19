"""
ALBart Art Review Tool — Milestone 2

Cycles through all processed 32x32 album covers, showing:
  - Left:  original downloaded art (reference)
  - Right: LED panel mockup (32x32 pixels with gap, ~128mm equivalent)
  - Bottom: track metadata and controls

Controls:
  ←  /  →    Previous / Next track
  F          Flag / unflag current track
  G          Jump to a specific index (prompts in terminal)
  Q          Quit and save flag list

On exit, writes data/flagged_tracks.txt with flagged track IDs and titles.

Usage:
    python tools/review_art.py [--flagged-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

# Project root is one level up from this file
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

# ── Layout constants ────────────────────────────────────────────────────────

LED_COUNT      = 32
LED_CELL       = 16    # pixels per LED cell (matches sim_scale)
LED_FACE       = 13    # LED lit area within cell (gap = cell - face = 3px)
LED_PANEL_PX   = LED_COUNT * LED_CELL          # 512px — ≈128mm at 96 DPI
BEZEL          = 20    # dark bezel around LED panel

REF_SIZE       = 400   # original art reference panel size
GAP            = 32    # horizontal gap between ref and LED panel
INFO_HEIGHT    = 110   # bottom info bar height

WIN_W = REF_SIZE + GAP + BEZEL + LED_PANEL_PX + BEZEL
WIN_H = max(REF_SIZE, BEZEL + LED_PANEL_PX + BEZEL) + INFO_HEIGHT

# ── Colors ──────────────────────────────────────────────────────────────────

C_BG         = (18, 18, 18)
C_BEZEL      = (30, 30, 30)
C_BEZEL_EDGE = (50, 50, 50)
C_LED_OFF    = (8,  8,  8)
C_INFO_BG    = (24, 24, 24)
C_INFO_LINE  = (45, 45, 45)
C_WHITE      = (230, 230, 230)
C_DIM        = (120, 120, 120)
C_FLAG_ON    = (220,  60,  60)
C_FLAG_OFF   = (60,  60,  60)
C_HINT       = (80,  80,  80)
C_INDEX      = (90,  90,  90)


def load_tracks() -> list[dict]:
    conn = get_connection(DB_PATH)
    rows = conn.execute(
        """SELECT track_id, title, artist, album, art_path_original, art_path_32
           FROM tracks
           WHERE art_path_32 IS NOT NULL
           ORDER BY title"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_original(art_path: str, size: int) -> pygame.Surface:
    path = DATA_DIR / art_path
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((size, size), Image.LANCZOS)
            data = np.array(img)
        return pygame.surfarray.make_surface(data.swapaxes(0, 1))
    except Exception:
        surf = pygame.Surface((size, size))
        surf.fill((40, 40, 40))
        return surf


def load_art_32(art_path: str) -> np.ndarray:
    path = DATA_DIR / art_path
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return np.zeros((32, 32, 3), dtype=np.uint8)


def render_led_panel(screen: pygame.Surface, pixels: np.ndarray, x: int, y: int) -> None:
    """Render the 32x32 pixel array as an LED panel with bezel."""
    # Bezel background
    bezel_rect = pygame.Rect(x, y, BEZEL + LED_PANEL_PX + BEZEL, BEZEL + LED_PANEL_PX + BEZEL)
    pygame.draw.rect(screen, C_BEZEL, bezel_rect, border_radius=8)
    pygame.draw.rect(screen, C_BEZEL_EDGE, bezel_rect, width=1, border_radius=8)

    # LED grid
    panel_x = x + BEZEL
    panel_y = y + BEZEL

    # Fill panel background (LED-off color visible in gaps)
    panel_rect = pygame.Rect(panel_x, panel_y, LED_PANEL_PX, LED_PANEL_PX)
    pygame.draw.rect(screen, C_LED_OFF, panel_rect)

    offset = (LED_CELL - LED_FACE) // 2
    for row in range(LED_COUNT):
        for col in range(LED_COUNT):
            r, g, b = int(pixels[row, col, 0]), int(pixels[row, col, 1]), int(pixels[row, col, 2])
            lx = panel_x + col * LED_CELL + offset
            ly = panel_y + row * LED_CELL + offset
            pygame.draw.rect(screen, (r, g, b), (lx, ly, LED_FACE, LED_FACE))


def render_info_bar(
    screen: pygame.Surface,
    track: dict,
    index: int,
    total: int,
    flagged: bool,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    font_hint: pygame.font.Font,
    y: int,
) -> None:
    bar_rect = pygame.Rect(0, y, WIN_W, INFO_HEIGHT)
    pygame.draw.rect(screen, C_INFO_BG, bar_rect)
    pygame.draw.line(screen, C_INFO_LINE, (0, y), (WIN_W, y))

    # Index
    idx_surf = font_hint.render(f"{index + 1} / {total}", True, C_INDEX)
    screen.blit(idx_surf, (16, y + 10))

    # Flag indicator
    flag_color = C_FLAG_ON if flagged else C_FLAG_OFF
    flag_surf = font_sm.render("⚑ FLAGGED" if flagged else "⚑ flag", True, flag_color)
    screen.blit(flag_surf, (WIN_W - flag_surf.get_width() - 16, y + 10))

    # Title
    title = track["title"]
    if len(title) > 55:
        title = title[:52] + "…"
    title_surf = font_lg.render(title, True, C_WHITE)
    screen.blit(title_surf, (16, y + 32))

    # Artist / album
    meta = f"{track['artist']}  ·  {track['album']}"
    if len(meta) > 75:
        meta = meta[:72] + "…"
    meta_surf = font_sm.render(meta, True, C_DIM)
    screen.blit(meta_surf, (16, y + 60))

    # Keyboard hints
    hints = "← → navigate    F flag/unflag    Q quit"
    hint_surf = font_hint.render(hints, True, C_HINT)
    screen.blit(hint_surf, (16, y + 84))


def render_ref_label(
    screen: pygame.Surface, font: pygame.font.Font, x: int, y: int
) -> None:
    lbl = font.render("original", True, C_HINT)
    screen.blit(lbl, (x + (REF_SIZE - lbl.get_width()) // 2, y))


def render_panel_label(
    screen: pygame.Surface, font: pygame.font.Font, x: int, y: int
) -> None:
    lbl = font.render("32 × 32  LED panel  (~128 mm)", True, C_HINT)
    screen.blit(lbl, (x + (BEZEL + LED_PANEL_PX + BEZEL - lbl.get_width()) // 2, y))


def save_flagged(flagged_ids: set[str], tracks: list[dict]) -> None:
    out = DATA_DIR / "flagged_tracks.txt"
    flagged_tracks = [t for t in tracks if t["track_id"] in flagged_ids]
    with open(out, "w") as f:
        for t in flagged_tracks:
            f.write(f"{t['track_id']}\t{t['artist']} — {t['title']}\n")
    print(f"Saved {len(flagged_tracks)} flagged tracks to {out}")


def load_existing_flags() -> set[str]:
    out = DATA_DIR / "flagged_tracks.txt"
    if not out.exists():
        return set()
    flags = set()
    with open(out) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if parts:
                flags.add(parts[0])
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="ALBart art review tool")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only show previously flagged tracks")
    args = parser.parse_args()

    tracks = load_tracks()
    if not tracks:
        print("No processed tracks found. Run the pipeline first.")
        sys.exit(1)

    flagged_ids = load_existing_flags()

    if args.flagged_only:
        tracks = [t for t in tracks if t["track_id"] in flagged_ids]
        if not tracks:
            print("No flagged tracks found.")
            sys.exit(0)

    print(f"Loaded {len(tracks)} tracks. Use ←→ to navigate, F to flag, Q to quit.")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("ALBart — Art Review")
    clock = pygame.time.Clock()

    font_lg   = pygame.font.SysFont("helveticaneue,arial,sans-serif", 20, bold=True)
    font_sm   = pygame.font.SysFont("helveticaneue,arial,sans-serif", 16)
    font_hint = pygame.font.SysFont("helveticaneue,arial,sans-serif", 13)

    index = 0
    running = True

    # Cache surfaces for current track
    cached_index = -1
    ref_surf = None
    pixels_32 = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                    index = (index + 1) % len(tracks)
                elif event.key == pygame.K_LEFT:
                    index = (index - 1) % len(tracks)
                elif event.key in (pygame.K_f,):
                    tid = tracks[index]["track_id"]
                    if tid in flagged_ids:
                        flagged_ids.discard(tid)
                    else:
                        flagged_ids.add(tid)
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Load assets if track changed
        if index != cached_index:
            track = tracks[index]
            ref_surf = load_original(track["art_path_original"], REF_SIZE)
            pixels_32 = load_art_32(track["art_path_32"])
            cached_index = index

        track = tracks[index]

        # ── Draw ────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        label_y = 8
        content_y = label_y + font_hint.get_height() + 6

        # Reference art
        render_ref_label(screen, font_hint, 0, label_y)
        screen.blit(ref_surf, (0, content_y))

        # LED panel
        panel_x = REF_SIZE + GAP
        render_panel_label(screen, font_hint, panel_x, label_y)
        render_led_panel(screen, pixels_32, panel_x, content_y)

        # Info bar
        info_y = max(REF_SIZE, BEZEL + LED_PANEL_PX + BEZEL) + content_y
        render_info_bar(
            screen, track, index, len(tracks),
            track["track_id"] in flagged_ids,
            font_lg, font_sm, font_hint, info_y,
        )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    save_flagged(flagged_ids, tracks)


if __name__ == "__main__":
    main()
