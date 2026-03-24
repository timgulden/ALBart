"""CLI entry point for the ALBart music map visualization.

Runs as a separate process alongside the main runtime.  Receives embedding
payloads via UDP from the main process and renders the music map.

Usage:
    python -m albart.runtime.run_map
    python -m albart.runtime.run_map --view neighborhood --mode half
    python -m albart.runtime.run_map --view umap --mode full --trail-minutes 30

Views:
    neighborhood — 3D perspective neighborhood (default)
    umap         — 2D UMAP global map with Voronoi labels

Display modes:
    half  — 1920×1080 (fits a 1080p monitor)
    full  — 3840×2160 (4K TV via HDMI)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import time

from albart.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Canvas sizes for each mode
_MODE_CANVAS: dict[str, tuple[int, int]] = {
    "half": (1920, 1080),
    "full": (3840, 2160),
}
_MODE_THUMB: dict[str, int] = {
    "half": 10,
    "full": 20,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="ALBart music map visualization")
    parser.add_argument(
        "--view", choices=["neighborhood", "umap"], default="neighborhood",
        help="Visualization: neighborhood (3D perspective, default) or umap (2D global map)",
    )
    parser.add_argument(
        "--mode", choices=["half", "full"], default="half",
        help="Display mode: half=1080p (default), full=4K",
    )
    parser.add_argument(
        "--port", type=int, default=57001,
        help="UDP port to receive embeddings on (default: 57001)",
    )
    parser.add_argument(
        "--trail-minutes", type=float, default=1.5,
        help="Trail fade duration in minutes (default: 1.5)",
    )
    parser.add_argument(
        "--no-voronoi-lines", action="store_true",
        help="Hide Voronoi region borders and labels (umap view only)",
    )
    args = parser.parse_args()

    config = load_config()
    rt = config["runtime"]
    nb = config.get("neighborhood", {})

    canvas_size = _MODE_CANVAS[args.mode]
    trail_secs  = args.trail_minutes * 60.0

    logger.info(
        "Starting music map — view=%s  mode=%s  canvas=%dx%d  port=%d",
        args.view, args.mode, canvas_size[0], canvas_size[1], args.port,
    )

    if args.view == "neighborhood":
        from albart.runtime.neighborhood_display import NeighborhoodDisplay

        display = NeighborhoodDisplay(
            canvas_size              = canvas_size,
            neighborhood_k           = int(nb.get("k", 100)),
            base_thumb_px            = int(nb.get("base_thumb_px", 80)),
            focal_length             = float(nb.get("focal_length", 2.0)),
            z_near                   = float(nb.get("z_near", 0.5)),
            zoom                     = float(nb.get("zoom", 1.5)),
            lerp_speed               = float(nb.get("lerp_speed", 0.8)),
            recompute_threshold      = float(nb.get("recompute_threshold", 0.15)),
            recompute_consecutive    = int(nb.get("recompute_consecutive", 3)),
            trail_max_seconds        = trail_secs,
            brightness_k             = float(rt["brightness_k"]),
            brightness_floor         = float(rt["brightness_floor"]),
            brightness_power         = float(rt["brightness_power"]),
            sphere_base_radius       = int(nb.get("sphere_base_radius", 50)),
        )
    else:
        from albart.runtime.map_display import MapDisplay

        display = MapDisplay(
            canvas_size       = canvas_size,
            thumb_size        = _MODE_THUMB[args.mode],
            trail_max_seconds = trail_secs,
            brightness_k      = float(rt["brightness_k"]),
            brightness_floor   = float(rt["brightness_floor"]),
            brightness_power   = float(rt["brightness_power"]),
            show_voronoi      = not args.no_voronoi_lines,
        )

    # Bind UDP socket (non-blocking)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    logger.info("Listening for embeddings on UDP port %d", args.port)

    fps = float(rt.get("display_fps", 30))
    frame_duration = 1.0 / fps

    try:
        while True:
            t0 = time.monotonic()

            # Drain UDP — process only the latest embedding per frame
            latest = None
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    latest = pickle.loads(data)
                except BlockingIOError:
                    break
                except Exception as e:
                    logger.warning("UDP receive error: %s", e)
                    break

            if latest is not None:
                display.update(
                    latest["raw"],
                    latest["top1"], latest["d_min_raw"],
                )

            display.render(frame_duration)

            sleep = frame_duration - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Music map shutting down.")
    finally:
        sock.close()
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
