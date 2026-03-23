"""Runtime CLAP embedder: loads model once, runs in a background thread.

Produces DualEmbedding pairs (raw + norm) that feed DualTrackLookup.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from albart.pipeline.embedder import SAMPLE_RATE, embed_audio, load_model
from albart.runtime.audio import AudioBuffer

logger = logging.getLogger(__name__)


@dataclass
class DualEmbedding:
    """A pair of CLAP embeddings for the same audio at two normalization levels."""
    raw:  np.ndarray   # (512,) float32 — no RMS normalization
    norm: np.ndarray   # (512,) float32 — RMS-normalized to norm_target


def _avg_normalize(embs: list[np.ndarray]) -> np.ndarray:
    """Average a list of unit vectors and renormalize (mean of unit vecs ≠ unit vec)."""
    avg = np.mean(embs, axis=0).astype(np.float32)
    return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)


class EmbeddingWorker:
    """
    Background thread that periodically reads the audio buffer,
    computes a DualEmbedding, and posts it to a result queue.
    """

    CHUNK_SAMPLES = 10 * SAMPLE_RATE   # 10s per chunk
    N_CHUNKS = 3                        # embed 3 chunks from the 30s buffer, then average

    def __init__(
        self,
        audio_buffer: AudioBuffer,
        result_queue: queue.Queue,
        interval_seconds: float = 10.0,
        alpha: float = 1.0,
        norm_target_raw: float = 0.0,
        norm_target_norm: float = 0.12,
        model=None,
        processor=None,
        device: str | None = None,
        on_embedding=None,
    ) -> None:
        self.audio_buffer = audio_buffer
        self.result_queue = result_queue
        self.interval = interval_seconds
        self.alpha = alpha
        self.norm_target_raw  = norm_target_raw
        self.norm_target_norm = norm_target_norm
        self._ema_raw:  np.ndarray | None = None
        self._ema_norm: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_embedding = on_embedding  # optional callable(DualEmbedding)

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

                # Embed 3 × 10s chunks at both normalization levels
                chunk_embs_raw  = []
                chunk_embs_norm = []
                for i in range(self.N_CHUNKS):
                    chunk = audio[i * self.CHUNK_SAMPLES:(i + 1) * self.CHUNK_SAMPLES]
                    chunk_embs_raw.append(
                        embed_audio(chunk, self.model, self.processor, self.device,
                                    norm_target=self.norm_target_raw)
                    )
                    chunk_embs_norm.append(
                        embed_audio(chunk, self.model, self.processor, self.device,
                                    norm_target=self.norm_target_norm)
                    )

                emb_raw  = _avg_normalize(chunk_embs_raw)
                emb_norm = _avg_normalize(chunk_embs_norm)

                # EMA smoothing (independent for each path)
                if self._ema_raw is None or self.alpha >= 1.0:
                    self._ema_raw  = emb_raw
                    self._ema_norm = emb_norm
                else:
                    raw_ema  = self.alpha * emb_raw  + (1.0 - self.alpha) * self._ema_raw
                    norm_ema = self.alpha * emb_norm + (1.0 - self.alpha) * self._ema_norm
                    self._ema_raw  = (raw_ema  / (np.linalg.norm(raw_ema)  + 1e-8)).astype(np.float32)
                    self._ema_norm = (norm_ema / (np.linalg.norm(norm_ema) + 1e-8)).astype(np.float32)

                emb = DualEmbedding(raw=self._ema_raw.copy(), norm=self._ema_norm.copy())
                self.result_queue.put(emb)
                if self._on_embedding is not None:
                    try:
                        self._on_embedding(emb)
                    except Exception as cb_err:
                        logger.debug("on_embedding callback error: %s", cb_err)
                elapsed = time.monotonic() - t0
                logger.debug(
                    "DualEmbedding computed in %.2fs  alpha=%.2f",
                    elapsed, self.alpha,
                )
            except Exception as e:
                logger.error("Embedding error: %s", e)

            sleep_time = self.interval - (time.monotonic() - t0)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
