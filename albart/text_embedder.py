"""Lazy-loading CLAP text embedder with auto-unload.

Loads the CLAP model on first use, caches it for 5 minutes of idleness,
then releases memory.  Thread-safe.

Usage:
    from albart.text_embedder import embed_texts
    embs = embed_texts(["jazz piano", "ambient electronic"])
    # Returns (N, 512) float32 L2-normalized embeddings
"""

from __future__ import annotations

import gc
import logging
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_processor = None
_device = None
_last_used: float = 0.0
_idle_timeout: float = 300.0  # 5 minutes
_cleanup_started = False


def _load() -> None:
    global _model, _processor, _device
    from albart.pipeline.embedder import MODEL_ID
    from albart.utils import get_device
    from transformers import ClapModel, ClapProcessor

    _device = get_device()
    logger.info("Loading CLAP model for text embedding on %s...", _device)
    _processor = ClapProcessor.from_pretrained(MODEL_ID)
    _model = ClapModel.from_pretrained(MODEL_ID).to(_device)
    _model.eval()
    logger.info("CLAP text embedder ready")


def _unload() -> None:
    global _model, _processor, _device
    logger.info("Unloading CLAP text embedder (idle timeout)")
    _model = None
    _processor = None
    _device = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _start_cleanup_thread() -> None:
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True

    def _cleanup():
        while True:
            time.sleep(60)
            with _lock:
                if (_model is not None
                        and time.monotonic() - _last_used > _idle_timeout):
                    _unload()

    t = threading.Thread(target=_cleanup, daemon=True, name="CLAPCleanup")
    t.start()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of text strings via CLAP. Returns (N, 512) float32.

    Loads the model on first call. Auto-unloads after 5 min idle.
    """
    global _last_used

    with _lock:
        if _model is None:
            _load()
            _start_cleanup_thread()
        _last_used = time.monotonic()

        inputs = _processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            features = _model.get_text_features(**inputs)

        embs = features.cpu().numpy().astype(np.float32)
        # L2-normalize (matches audio embedding normalization)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= np.maximum(norms, 1e-8)

        return embs
