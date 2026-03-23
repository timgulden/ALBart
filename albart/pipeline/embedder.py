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

from albart.utils import DATA_DIR, get_device, preprocess_audio, compress_and_normalize

logger = logging.getLogger(__name__)

MODEL_ID = "laion/larger_clap_music"
EMBEDDING_DIM = 512
SAMPLE_RATE = 48000

# Default (legacy) single-index paths
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH   = DATA_DIR / "faiss_ids.npy"

# Dual-index paths (raw = no normalization; norm = RMS-normalized)
FAISS_RAW_INDEX_PATH = DATA_DIR / "faiss_raw.index"
FAISS_RAW_IDS_PATH   = DATA_DIR / "faiss_raw_ids.npy"
FAISS_NORM_INDEX_PATH = DATA_DIR / "faiss_norm.index"
FAISS_NORM_IDS_PATH   = DATA_DIR / "faiss_norm_ids.npy"


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
    norm_target: float = 0.0,
) -> np.ndarray:
    """
    Compute a CLAP embedding for a mono float32 audio array at 48kHz.
    Returns a (512,) float32 numpy array.

    norm_target: RMS normalization target applied before CLAP processing.
        0.0 (default) = no normalization (raw path).
        0.12 = normalize to RMS 0.12 (blurs mel-spectrogram, more robust
               to room acoustics for tracks recorded at high volume).

    For the fused model (enable_fusion=True), truncation="fusion" is required —
    it produces 4 mel-spectrograms (3 random crops + 1 global) for the forward pass.
    """
    fusion = getattr(model.config.audio_config, "enable_fusion", False)
    hidden = getattr(model.config.audio_config, "hidden_size", 768)
    if not fusion:
        if hidden <= 768:
            # Tiny unfused model: full preprocessing (RMS + hard limit + LP)
            audio = preprocess_audio(audio, sr=SAMPLE_RATE)
        else:
            # Larger music model: optional RMS normalization.
            if norm_target > 0:
                rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
                audio = (audio * (norm_target / rms)).astype(np.float32)
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
    index_path: Path | None = None,
    ids_path: Path | None = None,
) -> None:
    """
    Build a FlatL2 FAISS index from embeddings and save index + ID map.
    embeddings: (N, 512) float32 array
    track_ids:  list of N track_id strings (parallel to embeddings)
    index_path: override for the .index file (default: FAISS_INDEX_PATH)
    ids_path:   override for the _ids.npy file (default: FAISS_IDS_PATH)
    """
    import faiss  # lazy — avoids BLAS conflict with CLAP processor on macOS

    index_path = index_path or FAISS_INDEX_PATH
    ids_path   = ids_path   or FAISS_IDS_PATH

    assert embeddings.shape == (len(track_ids), EMBEDDING_DIM), (
        f"Expected ({len(track_ids)}, {EMBEDDING_DIM}), got {embeddings.shape}"
    )

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    np.save(str(ids_path), np.array(track_ids))
    logger.info("Saved FAISS index (%d vectors) to %s", len(track_ids), index_path)


def load_index(
    index_path: Path | None = None,
    ids_path: Path | None = None,
) -> tuple:
    """Load FAISS index and track ID map. Returns (index, track_ids array).

    index_path: override for the .index file (default: FAISS_INDEX_PATH)
    ids_path:   override for the _ids.npy file (default: FAISS_IDS_PATH)
    """
    import faiss  # lazy — avoids BLAS conflict with CLAP processor on macOS

    index_path = index_path or FAISS_INDEX_PATH
    ids_path   = ids_path   or FAISS_IDS_PATH

    index = faiss.read_index(str(index_path))
    track_ids = np.load(str(ids_path), allow_pickle=True)
    logger.info("Loaded FAISS index with %d vectors from %s", index.ntotal, index_path)
    return index, track_ids
