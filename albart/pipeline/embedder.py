"""CLAP inference and FAISS index construction."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from albart.utils import DATA_DIR, get_device

logger = logging.getLogger(__name__)

MODEL_ID = "laion/clap-htsat-unfused"
EMBEDDING_DIM = 512
SAMPLE_RATE = 48000

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.npy"


def load_model(device: str | None = None):
    """Load CLAP model and processor. Returns (model, processor, device)."""
    if device is None:
        device = get_device()
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
    """
    inputs = processor(
        audios=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_audio_features(**inputs)

    return embedding.cpu().numpy().squeeze().astype(np.float32)


def build_and_save_index(
    embeddings: np.ndarray,
    track_ids: list[str],
) -> None:
    """
    Build a FlatL2 FAISS index from embeddings and save index + ID map.
    embeddings: (N, 512) float32 array
    track_ids: list of N track_id strings (parallel to embeddings)
    """
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


def load_index() -> tuple[faiss.Index, np.ndarray]:
    """Load FAISS index and track ID map. Returns (index, track_ids array)."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    track_ids = np.load(str(FAISS_IDS_PATH), allow_pickle=True)
    logger.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index, track_ids
