"""RoomEar service: microphone → CLAP embedding → UDP broadcast.

A lightweight standalone service that captures ambient audio, computes
512D CLAP embeddings, and publishes them via UDP for consumption by
the DJ engine and/or MapView.

Reuses AudioBuffer, EmbeddingWorker, and AGCWorker from albart.runtime.
"""

from __future__ import annotations

import logging
import pickle
import queue
import socket
import threading
import time

import numpy as np

from albart.pipeline.embedder import load_model
from albart.runtime.agc import AGCWorker
from albart.runtime.audio import AudioBuffer
from albart.runtime.embedder import EmbeddingWorker

logger = logging.getLogger(__name__)


class RoomEarService:
    """Standalone ambient audio embedding service.

    Captures microphone input, computes CLAP embeddings every
    ``interval`` seconds, and broadcasts the raw 512D vector via UDP.
    """

    def __init__(
        self,
        config: dict,
        device: int | None = None,
        port: int = 57002,
    ) -> None:
        self._config = config
        self._audio_device = device
        self._port = port

        rt = config.get("runtime", {})
        self._buffer_length = rt.get("buffer_length_seconds", 30)
        self._interval = rt.get("embedding_interval_seconds", 10)
        self._alpha = rt.get("embedding_alpha", 1.0)
        self._norm_target = float(rt.get("norm_target_raw", 0.0))
        self._agc_enabled = rt.get("agc_enabled", True)
        self._agc_target_rms = float(rt.get("agc_target_rms", 0.20))
        self._agc_interval = float(rt.get("agc_interval_seconds", 10.0))

        # Components (created in run())
        self._audio_buffer: AudioBuffer | None = None
        self._agc: AGCWorker | None = None
        self._worker: EmbeddingWorker | None = None
        self._sock: socket.socket | None = None

    def _on_embedding(self, emb: np.ndarray) -> None:
        """Callback from EmbeddingWorker — broadcast via UDP."""
        if self._sock is None:
            return
        try:
            data = pickle.dumps({
                "raw": emb,
                "top1": None,
                "d_min_raw": 0.0,
                "source": "roomear",
            })
            self._sock.sendto(data, ("127.0.0.1", self._port))
            logger.debug("Broadcast embedding to port %d", self._port)
        except Exception as e:
            logger.debug("Broadcast error: %s", e)

    def run(self) -> None:
        """Start all components and block until interrupted."""
        logger.info("Starting RoomEar service (port=%d)", self._port)

        # UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Audio capture — start immediately so buffer fills during model load
        self._audio_buffer = AudioBuffer(
            buffer_length_seconds=self._buffer_length,
        )
        self._audio_buffer.start(device=self._audio_device)

        # AGC (macOS only)
        self._agc = AGCWorker(
            audio_buffer=self._audio_buffer,
            target_rms=self._agc_target_rms,
            interval_seconds=self._agc_interval,
        )
        if self._agc_enabled:
            self._agc.start()

        # Load CLAP model in background
        logger.info("Loading CLAP model...")
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

        # Wait for model + buffer
        logger.info("Waiting for CLAP model and audio buffer to fill...")
        model_ready.wait()
        if "error" in model_store:
            raise RuntimeError("CLAP model failed to load") from model_store["error"]

        while not self._audio_buffer.buffer_full:
            time.sleep(0.25)

        logger.info("Ready — starting embedding worker")

        # Embedding worker (queue is unused — we broadcast via callback)
        embedding_queue: queue.Queue = queue.Queue()
        self._worker = EmbeddingWorker(
            audio_buffer=self._audio_buffer,
            result_queue=embedding_queue,
            interval_seconds=self._interval,
            alpha=self._alpha,
            norm_target=self._norm_target,
            model=model_store["model"],
            processor=model_store["processor"],
            device=model_store["device"],
            on_embedding=self._on_embedding,
        )
        self._worker.start()

        logger.info("RoomEar running — broadcasting to 127.0.0.1:%d every %.0fs",
                     self._port, self._interval)

        try:
            while True:
                time.sleep(1.0)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down RoomEar...")
        finally:
            if self._worker:
                self._worker.stop()
            if self._agc:
                self._agc.stop()
            if self._audio_buffer:
                self._audio_buffer.stop()
            if self._sock:
                self._sock.close()
