"""CLI entry point for the ALBart runtime.

Usage:
    python -m albart.runtime.run

Startup sequence:
  1. Open display window
  2. Begin loading CLAP model in a background thread
  3. Show ripple startup animation until model is ready
  4. Fade animation to black
  5. Start audio capture, embedding worker, display loop
"""

from __future__ import annotations

import argparse
import logging
import pickle
import queue
import socket
import sys
import threading

from albart.pipeline.embedder import load_model
from albart.runtime.agc import AGCWorker
from albart.runtime.audio import AudioBuffer
from albart.runtime.embedder import EmbeddingWorker
from albart.runtime.lookup import DatabaseTrackLookup
from albart.runtime.loop import DisplayLoop
from albart.runtime.startup import run_startup
from albart.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_device(name: str) -> int:
    import sounddevice as sd
    try:
        return int(name)
    except ValueError:
        devs = sd.query_devices()
        matches = [i for i, d in enumerate(devs)
                   if name.lower() in d["name"].lower() and d["max_input_channels"] > 0]
        if not matches:
            print(f"No input device matching '{name}'. Available inputs:")
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    print(f"  [{i}] {d['name']}")
            sys.exit(1)
        print(f"Using device [{matches[0]}]: {sd.query_devices(matches[0])['name']}")
        return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="ALBart runtime")
    parser.add_argument("--device", type=str, default=None,
                        help="Input audio device name or index (default: system default)")
    args = parser.parse_args()

    config = load_config()
    rt = config["runtime"]

    # Resolve audio device: CLI arg > config > system default
    device_name = args.device or rt.get("audio_device")
    device = resolve_device(device_name) if device_name is not None else None

    # --- Display backend (only place these are imported) ---
    display_mode = rt.get("display_mode", "sim")
    if display_mode == "sim":
        from albart.runtime.display_sim import SimDisplay
        display = SimDisplay(scale=rt.get("sim_scale", 16))
    elif display_mode == "hub75":
        from albart.runtime.display_hub75 import Hub75Display
        display = Hub75Display()
    else:
        raise ValueError(f"Unknown display_mode: {display_mode!r}")

    logger.info("Starting ALBart runtime (display_mode=%s)", display_mode)

    # --- Start audio capture immediately so the buffer fills during startup ---
    audio_buffer = AudioBuffer(buffer_length_seconds=rt["buffer_length_seconds"])
    audio_buffer.start(device=device)

    # --- AGC: adjust macOS input volume to keep RMS near target (macOS only) ---
    agc = AGCWorker(
        audio_buffer=audio_buffer,
        target_rms=float(rt.get("agc_target_rms", 0.20)),
        interval_seconds=float(rt.get("agc_interval_seconds", 10.0)),
    )
    if rt.get("agc_enabled", True):
        agc.start()

    # --- Load CLAP model in background while startup animation plays ---
    model_ready = threading.Event()
    model_store: dict = {}

    def _load():
        try:
            m, p, d = load_model()
            model_store["model"] = m
            model_store["processor"] = p
            model_store["device"] = d
        except Exception as e:
            logger.error("Model load failed: %s", e)
            model_store["error"] = e
        finally:
            model_ready.set()

    threading.Thread(target=_load, daemon=True, name="ModelLoader").start()

    # Ready when BOTH model is loaded AND the audio buffer has 30s of real audio.
    # The buffer fills in real time (~30s), which is typically longer than model load.
    all_ready = threading.Event()

    def _watch_ready():
        model_ready.wait()
        while not audio_buffer.buffer_full:
            threading.Event().wait(timeout=0.25)
        all_ready.set()

    threading.Thread(target=_watch_ready, daemon=True, name="ReadyWatcher").start()
    run_startup(display, all_ready, fps=rt.get("display_fps", 30))

    if "error" in model_store:
        raise RuntimeError("CLAP model failed to load") from model_store["error"]

    # --- UDP broadcast to map display + DJ process ---
    map_port = rt.get("map_broadcast_port", 0)
    dj_port  = rt.get("dj_broadcast_port", 57002)
    map_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _broadcast_map(emb: "np.ndarray", top1_id: str, d_min_raw: float) -> None:
        if not map_port and not dj_port:
            return
        try:
            data = pickle.dumps({
                "raw": emb, "top1": top1_id, "d_min_raw": d_min_raw,
            })
            if map_port:
                map_sock.sendto(data, ("127.0.0.1", map_port))
            if dj_port:
                map_sock.sendto(data, ("127.0.0.1", dj_port))
        except Exception as e:
            logger.debug("Map broadcast error: %s", e)

    # --- Runtime components ---
    embedding_queue: queue.Queue = queue.Queue()

    from albart.effects.database import DatabaseClient, DatabaseConfig
    lookup = DatabaseTrackLookup(DatabaseClient(DatabaseConfig()))

    worker = EmbeddingWorker(
        audio_buffer=audio_buffer,
        result_queue=embedding_queue,
        interval_seconds=rt["embedding_interval_seconds"],
        alpha=rt.get("embedding_alpha", 1.0),
        norm_target=float(rt.get("norm_target_raw", 0.0)),
        model=model_store["model"],
        processor=model_store["processor"],
        device=model_store["device"],
    )
    worker.start()

    loop = DisplayLoop(
        display=display,
        lookup=lookup,
        embedding_queue=embedding_queue,
        config=config,
        map_broadcast=_broadcast_map,
    )

    try:
        loop.run()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
    finally:
        worker.stop()
        agc.stop()
        audio_buffer.stop()
        display.close()
        map_sock.close()


if __name__ == "__main__":
    main()
