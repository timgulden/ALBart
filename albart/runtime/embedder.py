"""Runtime CLAP embedder: loads model once, runs in a background thread."""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from albart.pipeline.embedder import SAMPLE_RATE, embed_audio, load_model
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
        alpha: float = 1.0,
        model=None,
        processor=None,
        device: str | None = None,
    ) -> None:
        self.audio_buffer = audio_buffer
        self.result_queue = result_queue
        self.interval = interval_seconds
        self.alpha = alpha  # EMA weight for current embedding; 1.0 = no averaging
        self._ema: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if model is not None:
            self.model, self.processor, self.device = model, processor, device
            logger.info("EmbeddingWorker using pre-loaded CLAP model on %s", device)
        else:
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

    CHUNK_SAMPLES = 10 * SAMPLE_RATE   # 10s per chunk (processor max_length_s for unfused models)
    N_CHUNKS = 3                        # embed 3 chunks from the 30s buffer, then average

    def _run(self) -> None:
        logger.info("EmbeddingWorker started (interval=%.1fs)", self.interval)
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                audio = self.audio_buffer.read()
                # Ensure we have 30s; pad with silence at the start if buffer not full yet
                total = self.N_CHUNKS * self.CHUNK_SAMPLES
                if len(audio) < total:
                    audio = np.pad(audio, (total - len(audio), 0))
                else:
                    audio = audio[-total:]

                # Embed 3 × 10s chunks and average — mirrors how the index is built
                chunk_embs = []
                for i in range(self.N_CHUNKS):
                    chunk = audio[i * self.CHUNK_SAMPLES:(i + 1) * self.CHUNK_SAMPLES]
                    chunk_embs.append(embed_audio(chunk, self.model, self.processor, self.device))
                avg = np.mean(chunk_embs, axis=0).astype(np.float32)
                # Renormalize: mean of unit vectors is not a unit vector.
                avg_norm = np.linalg.norm(avg)
                embedding = (avg / (avg_norm + 1e-8)).astype(np.float32)
                if self._ema is None or self.alpha >= 1.0:
                    self._ema = embedding
                else:
                    raw_ema = self.alpha * embedding + (1.0 - self.alpha) * self._ema
                    ema_norm = np.linalg.norm(raw_ema)
                    self._ema = (raw_ema / (ema_norm + 1e-8)).astype(np.float32)
                self.result_queue.put(self._ema.copy())
                elapsed = time.monotonic() - t0
                logger.debug(
                    "Embedding computed in %.2fs  alpha=%.2f  ema=%s",
                    elapsed, self.alpha, "fresh" if self.alpha >= 1.0 else "blended",
                )
            except Exception as e:
                logger.error("Embedding error: %s", e)

            # Sleep for the remainder of the interval.
            # interval=0 means continuous: recompute immediately after finishing.
            sleep_time = self.interval - (time.monotonic() - t0)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
