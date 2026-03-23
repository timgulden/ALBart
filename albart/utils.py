"""Shared utilities: device selection, config loading, audio preprocessing."""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (ALBart/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def get_device() -> str:
    """Return the best available torch device: 'mps', or 'cpu'."""
    if torch.backends.mps.is_available():
        logger.debug("Using MPS device (Apple Silicon)")
        return "mps"
    logger.debug("Using CPU device")
    return "cpu"


def preprocess_audio(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """
    Preprocess audio to improve robustness to room acoustics and mic coloration.

    Steps:
      1. RMS normalize to a fixed target level — removes level differences
         between clean studio previews and quiet room mic recordings.
      2. Hard limit at 0.02 — strips amplitude dynamics, keeping only the
         temporal/spectral structure of when energy is present.  Empirically
         the most effective single step for room-mic robustness.
      3. Low-pass filter at 4kHz — band-limits to the range least affected by
         room reflections, speaker coloration, and mic frequency response.

    Applied identically on both the pipeline (reference embeddings) and the
    runtime/query side so the two embedding spaces remain aligned.

    Empirically reduces L2 distance between a clean preview and a room-mic
    recording of the same track from ~0.46 to ~0.09 (80% improvement) while
    preserving inter-track discrimination across diverse genres.
    """
    import scipy.signal

    # 1. RMS normalize
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
    audio = (audio * (0.1 / rms)).astype(np.float32)

    # 2. Hard limit
    audio = np.clip(audio, -0.02, 0.02).astype(np.float32)

    # 3. Low-pass at 4kHz
    sos = scipy.signal.butter(8, 4000, btype="low", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, audio).astype(np.float32)


def compress_and_normalize(
    audio: np.ndarray,
    sr: int = 48000,
    threshold: float = 0.3,
    ratio: float = 4.0,
    frame_ms: int = 10,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Dynamic range compression + RMS normalization for larger_clap_music.

    Designed to handle wide loudness variation across genres (e.g. BTS RMS=0.70
    vs Bach RMS=0.06) and to recover clipped/distorted recordings.

    Steps:
      1. Frame-based RMS envelope detection (default 10ms frames).
      2. Soft-knee gain reduction above threshold at ratio:1 — reduces loud peaks
         without the spectral damage of hard limiting.
      3. Final global RMS normalization to target_rms — aligns all genres and
         makes the embedding space agnostic to absolute playback level.

    Applied identically on the pipeline (reference) and runtime (query) sides so
    the two embedding spaces remain aligned.
    """
    frame_samples = int(sr * frame_ms / 1000)
    audio = audio.astype(np.float32)
    n = len(audio)
    gain = np.ones(n, dtype=np.float32)

    # Frame-based gain computation
    for start in range(0, n, frame_samples):
        frame = audio[start : start + frame_samples]
        frame_rms = float(np.sqrt(np.mean(frame ** 2))) + 1e-8
        if frame_rms > threshold:
            # Gain reduction: target = threshold + (frame_rms - threshold) / ratio
            target_level = threshold + (frame_rms - threshold) / ratio
            gain[start : start + frame_samples] = target_level / frame_rms

    compressed = (audio * gain).astype(np.float32)

    # Final RMS normalize
    global_rms = float(np.sqrt(np.mean(compressed ** 2))) + 1e-8
    return (compressed * (target_rms / global_rms)).astype(np.float32)


def compress_lp4k(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """
    Norm-path preprocessing for dual-index: DRC + low-pass at 4kHz.

    Complements preprocess_audio (the raw path) by using dynamic range
    compression instead of hard limiting.  Empirically the best norm-path
    strategy for dual-index RRF fusion with clap-htsat-unfused.
    """
    import scipy.signal
    audio = compress_and_normalize(audio)
    sos = scipy.signal.butter(8, 4000, btype="low", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, audio).astype(np.float32)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return the config.yaml as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
