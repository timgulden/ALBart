"""CLI entry point for the ALBart runtime.

Usage:
    python -m albart.runtime.run
"""

import logging
import queue

from albart.runtime.audio import AudioBuffer
from albart.runtime.embedder import EmbeddingWorker
from albart.runtime.lookup import TrackLookup
from albart.runtime.loop import DisplayLoop
from albart.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    rt = config["runtime"]

    # Select and initialize display backend (only place these are imported)
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

    embedding_queue: queue.Queue = queue.Queue()

    audio_buffer = AudioBuffer(
        buffer_length_seconds=rt["buffer_length_seconds"]
    )
    audio_buffer.start()

    lookup = TrackLookup()

    worker = EmbeddingWorker(
        audio_buffer=audio_buffer,
        result_queue=embedding_queue,
        interval_seconds=rt["embedding_interval_seconds"],
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
