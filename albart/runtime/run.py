"""CLI entry point for the ALBart runtime.

Usage:
    python -m albart.runtime.run

Startup sequence:
  1. Open display window
  2. Begin loading CLAP model in a background thread
  3. Show ripple startup animation until model is ready
  4. Fade animation to black
  5. Start audio capture, FAISS lookup, embedding worker, display loop
"""

from __future__ import annotations

import logging
import queue
import threading

from albart.pipeline.embedder import load_model
from albart.runtime.audio import AudioBuffer
from albart.runtime.embedder import EmbeddingWorker
from albart.runtime.lookup import TrackLookup
from albart.runtime.loop import DisplayLoop
from albart.runtime.startup import run_startup
from albart.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    rt = config["runtime"]

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
    run_startup(display, model_ready, fps=rt.get("display_fps", 30))

    if "error" in model_store:
        raise RuntimeError("CLAP model failed to load") from model_store["error"]

    # --- Runtime components ---
    embedding_queue: queue.Queue = queue.Queue()

    audio_buffer = AudioBuffer(buffer_length_seconds=rt["buffer_length_seconds"])
    audio_buffer.start()

    lookup = TrackLookup()

    worker = EmbeddingWorker(
        audio_buffer=audio_buffer,
        result_queue=embedding_queue,
        interval_seconds=rt["embedding_interval_seconds"],
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
    )

    try:
        loop.run()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
    finally:
        worker.stop()
        audio_buffer.stop()
        display.close()


if __name__ == "__main__":
    main()
