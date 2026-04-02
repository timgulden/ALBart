"""CLAP audio inference and embedding storage."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from albart.utils import get_device, preprocess_audio, compress_and_normalize, compress_lp4k

logger = logging.getLogger(__name__)

MODEL_ID = "laion/clap-htsat-unfused"
EMBEDDING_DIM = 512
SAMPLE_RATE = 48000


def load_model(allow_mps: bool = False):
    """Load CLAP model and processor. Returns (model, processor, device).

    allow_mps: if True, use MPS when available (safe for pipeline / sweep tools
               that do not run alongside pygame's Metal renderer).
               Default False keeps CPU for the runtime display loop, where MPS
               triggers a Metal shader race with pygame.
    """
    device = get_device() if allow_mps else "cpu"
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
    """Compute a CLAP embedding for a mono float32 audio array at 48kHz.

    Returns a L2-normalised (512,) float32 numpy array.

    norm_target: 0.0 = raw preprocessing path (RMS→0.1, hard limit, LP4k).
                 >0  = norm preprocessing path (DRC + LP4k).
    """
    fusion = getattr(model.config.audio_config, "enable_fusion", False)
    hidden = getattr(model.config.audio_config, "hidden_size", 768)
    if not fusion:
        if hidden <= 768:
            if norm_target > 0:
                audio = compress_lp4k(audio, sr=SAMPLE_RATE)
            else:
                audio = preprocess_audio(audio, sr=SAMPLE_RATE)
        else:
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
    inputs = {k: v.to(device) if v.dtype != torch.bool else v
              for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_audio_features(**inputs)

    vec = embedding.cpu().numpy().squeeze().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8)


def embed_track(
    audio: np.ndarray,
    model: ClapModel,
    processor: ClapProcessor,
    device: str,
    norm_target: float = 0.12,
    chunk_seconds: int = 10,
) -> np.ndarray | None:
    """Embed a track of any length: chunk, embed each, average, normalize.

    Splits the audio into non-overlapping chunks of ``chunk_seconds``.
    Discards a final partial chunk shorter than ``chunk_seconds``.
    Returns None if the audio is shorter than one full chunk.

    This is the single authoritative implementation of the chunk-average
    strategy.  All callers (pipeline, ingestion, runtime) should use this
    rather than rolling their own loop.

    Args:
        audio: mono float32 array at 48 kHz (any length).
        model: loaded CLAP model.
        processor: CLAP processor.
        device: torch device string.
        norm_target: preprocessing normalization (0.12 = compress+LP4k).
        chunk_seconds: length of each chunk in seconds.

    Returns:
        L2-normalized (512,) float32 embedding, or None if too short.
    """
    chunk_samples = chunk_seconds * SAMPLE_RATE
    min_chunk = int(chunk_samples * 0.9)  # pad chunks that are ≥90% full

    if len(audio) < min_chunk:
        return None  # track shorter than one usable chunk

    chunk_embs = []
    offset = 0
    while offset < len(audio):
        remaining = len(audio) - offset
        if remaining < min_chunk:
            break  # discard final fragment under 90% of chunk size
        chunk = audio[offset:offset + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunk_embs.append(
            embed_audio(chunk, model, processor, device, norm_target=norm_target)
        )
        offset += chunk_samples

    if not chunk_embs:
        return None

    avg = np.mean(chunk_embs, axis=0).astype(np.float32)
    return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)


def save_embeddings_to_db(
    track_ids: list[str],
    embeddings: np.ndarray,
) -> None:
    """Store 512D embeddings in PostgreSQL.

    Args:
        track_ids: list of N track_id strings.
        embeddings: (N, 512) float32 array (L2-normalised).
    """
    from albart.pipeline.database import get_db

    db = get_db()
    for i, tid in enumerate(track_ids):
        db.upsert_embedding(tid, embeddings[i])
    logger.info("Saved %d embeddings to PostgreSQL", len(track_ids))
