"""Runtime CLAP embedder: loads model once, runs in a background thread."""

import logging
import queue
import threading
import time

import numpy as np

from albart.pipeline.embedder import embed_audio, load_model
from albart.runtime.audio import AudioBuffer

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """
    Background thread that periodically reads the audio buffer,
    computes a CLAP embedding, and posts it to a result queue.
    """

    def __init__(
        self,
        audio_buffer: AudioBuffer,
        result_queue: queue.Queue,
        interval_seconds: float = 10.0,
    ) -> None:
        self.audio_buffer = audio_buffer
        self.result_queue = result_queue
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        logger.info("Loading CLAP model for runtime...")
        self.model, self.processor, self.device = load_model()
        logger.info("CLAP model ready on %s", self.device)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="EmbeddingWorker")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        logger.info("EmbeddingWorker started (interval=%.1fs)", self.interval)
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                audio = self.audio_buffer.read()
                embedding = embed_audio(audio, self.model, self.processor, self.device)
                self.result_queue.put(embedding)
                elapsed = time.monotonic() - t0
                logger.debug("Embedding computed in %.2fs", elapsed)
            except Exception as e:
                logger.error("Embedding error: %s", e)

            # Sleep for the remainder of the interval
            sleep_time = self.interval - (time.monotonic() - t0)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
