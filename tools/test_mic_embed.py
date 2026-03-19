"""
Quick sanity check: mic capture → CLAP embedding.

Records 10 seconds of audio, runs CLAP inference, reports timing and
embedding stats. No display or FAISS involved.

Usage:
    python tools/test_mic_embed.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd

from albart.pipeline.embedder import embed_audio, load_model

SAMPLE_RATE = 48000
RECORD_SECONDS = 10


def main() -> None:
    # --- Device info ---
    print("Available input devices:")
    devices = sd.query_devices()
    default_input = sd.query_devices(kind="input")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◀ default" if d["name"] == default_input["name"] else ""
            print(f"  [{i}] {d['name']} ({d['max_input_channels']}ch){marker}")
    print()

    # --- Record ---
    print(f"Recording {RECORD_SECONDS}s from default mic... (make some noise)")
    try:
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:
        print(f"ERROR: mic capture failed: {e}")
        sys.exit(1)

    audio = audio.flatten()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Captured {len(audio):,} samples  RMS={rms:.4f}  peak={peak:.4f}")

    if rms < 1e-5:
        print("WARNING: audio looks like silence — check mic permissions and input level")
    print()

    # --- Load model ---
    print("Loading CLAP model (first load may take 10-15s)...")
    t0 = time.monotonic()
    model, processor, device = load_model()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s on {device}")
    print()

    # --- Embed ---
    print("Computing embedding...")
    t0 = time.monotonic()
    embedding = embed_audio(audio, model, processor, device)
    elapsed = time.monotonic() - t0

    print(f"Embedding computed in {elapsed:.2f}s")
    print(f"Shape: {embedding.shape}  dtype: {embedding.dtype}")
    print(f"Norm:  {float(np.linalg.norm(embedding)):.4f}")
    print(f"Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print()
    print("All good — mic and CLAP inference are working.")


if __name__ == "__main__":
    main()
