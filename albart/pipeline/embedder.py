"""CLAP inference and FAISS index construction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

# faiss is imported lazily inside build_and_save_index / load_index only.
# Importing it at module level causes a BLAS conflict with the fused CLAP
# model's mel-spectrogram processor on macOS, producing a segfault.

from albart.utils import DATA_DIR, get_device, preprocess_audio

logger = logging.getLogger(__name__)

MODEL_ID = "laion/larger_clap_music"
EMBEDDING_DIM = 512
SAMPLE_RATE = 48000

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.npy"


def load_model(device: str | None = None):
    """Load CLAP model and processor. Returns (model, processor, device).

    Always runs on CPU: the fused model triggers a Metal shader race when used
    on MPS alongside pygame's Metal renderer.  CPU is also the Pi 5 target.
    """
    device = "cpu"
    logger.info("Loading CLAP model %s on %s", MODEL_ID, device)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def embed_audio(
    audio: np.ndarray,
    model: ClapModel,
    processor: ClapProcessor,
    device: str,
) -> np.ndarray:
    """
    Compute a CLAP embedding for a mono float32 audio array at 48kHz.
    Returns a (512,) float32 numpy array.

    For the fused model (enable_fusion=True), truncation="fusion" is required —
    it produces 4 mel-spectrograms (3 random crops + 1 global) for the forward pass.
    """
    fusion = getattr(model.config.audio_config, "enable_fusion", False)
    hidden = getattr(model.config.audio_config, "hidden_size", 768)
    if not fusion:
        if hidden <= 768:
            # Tiny unfused model: full preprocessing (RMS + hard limit + LP)
            audio = preprocess_audio(audio, sr=SAMPLE_RATE)
        # Larger unfused models (hidden > 768): pass raw audio — any preprocessing
        # degrades embeddings for the music-specific larger model.
    kwargs = {"truncation": "fusion"} if fusion else {}
    inputs = processor(
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        **kwargs,
    )
    # Move float tensors to device; skip bool tensors (e.g. is_longer for fused
    # model) — calling .to() on them causes a segfault on some torch builds.
    inputs = {k: v.to(device) if v.dtype != torch.bool else v
              for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_audio_features(**inputs)

    vec = embedding.cpu().numpy().squeeze().astype(np.float32)
    # L2-normalize so FlatL2 distance ≡ cosine distance (CLAP contrastive objective).
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8)


def build_and_save_index(
    embeddings: np.ndarray,
    track_ids: list[str],
) -> None:
    """
    Build a FlatL2 FAISS index from embeddings and save index + ID map.
    embeddings: (N, 512) float32 array
    track_ids: list of N track_id strings (parallel to embeddings)
    """
    import faiss  # lazy — avoids BLAS conflict with CLAP processor on macOS

    assert embeddings.shape == (len(track_ids), EMBEDDING_DIM), (
        f"Expected ({len(track_ids)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    )

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(str(FAISS_IDS_PATH), np.array(track_ids))
    logger.info(
        "Saved FAISS index (%d vectors) to %s", len(track_ids), FAISS_INDEX_PATH
    )


def load_index() -> tuple:
    """Load FAISS index and track ID map. Returns (index, track_ids array)."""
    import faiss  # lazy — avoids BLAS conflict with CLAP processor on macOS

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    track_ids = np.load(str(FAISS_IDS_PATH), allow_pickle=True)
    logger.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index, track_ids
