"""Mic capture with a thread-safe circular buffer."""

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000


class AudioBuffer:
    """
    A circular buffer that receives audio from a sounddevice InputStream
    callback and allows safe reads from other threads.
    """

    def __init__(self, buffer_length_seconds: float = 10.0) -> None:
        self.sample_rate = SAMPLE_RATE
        self.n_samples = int(buffer_length_seconds * SAMPLE_RATE)
        self._buffer = np.zeros(self.n_samples, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning("Audio stream status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        n = len(mono)
        with self._lock:
            end = self._write_pos + n
            if end <= self.n_samples:
                self._buffer[self._write_pos:end] = mono
            else:
                first = self.n_samples - self._write_pos
                self._buffer[self._write_pos:] = mono[:first]
                self._buffer[: n - first] = mono[first:]
            self._write_pos = end % self.n_samples

    def start(self) -> None:
        """Open and start the audio input stream."""
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Audio capture started (%.1fs buffer @ %dHz)",
            self.n_samples / SAMPLE_RATE,
            SAMPLE_RATE,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped")

    def read(self) -> np.ndarray:
        """
        Return a copy of the current buffer contents in chronological order
        (oldest sample first).
        """
        with self._lock:
            pos = self._write_pos
            return np.concatenate([self._buffer[pos:], self._buffer[:pos]])
