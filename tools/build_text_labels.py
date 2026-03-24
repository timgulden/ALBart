"""Pre-compute CLAP text embeddings for music genre/mood vocabulary.

Embeds ~200 short music descriptors using the same CLAP model used for audio.
Saves to data/text_labels.npz with keys:
  labels:     (M,) array of strings
  embeddings: (M, 512) float32 array

Usage:
    python tools/build_text_labels.py
    python tools/build_text_labels.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.utils import DATA_DIR

OUTPUT_PATH = DATA_DIR / "text_labels.npz"

# Music descriptor vocabulary — short, specific phrases that CLAP can
# distinguish.  Each becomes a point in the same 512-D space as the
# audio embeddings.
VOCABULARY = [
    # Rock / Metal
    "hard rock guitar",
    "classic rock",
    "heavy metal",
    "punk rock",
    "alternative rock",
    "grunge",
    "progressive rock",
    "psychedelic rock",
    "garage rock",
    "blues rock",
    "soft rock",
    "arena rock anthem",
    "surf rock",
    "southern rock",
    "indie rock",
    # Pop
    "pop music",
    "synth pop",
    "dance pop",
    "power ballad",
    "teen pop",
    "art pop",
    "dream pop",
    "indie pop",
    "chamber pop",
    # Electronic
    "electronic music",
    "ambient electronic",
    "techno",
    "house music",
    "drum and bass",
    "dubstep",
    "trance music",
    "chillout lounge",
    "glitch electronic",
    "synthwave",
    "IDM experimental electronic",
    "downtempo",
    # Hip Hop / R&B
    "hip hop rap",
    "old school hip hop",
    "trap beat",
    "R&B soul",
    "neo soul",
    "funk music",
    "disco",
    # Jazz
    "jazz",
    "bebop jazz",
    "cool jazz",
    "jazz fusion",
    "smooth jazz",
    "free jazz avant-garde",
    "jazz piano trio",
    "big band swing",
    "jazz vocal",
    "latin jazz",
    # Blues
    "blues guitar",
    "delta blues",
    "electric blues",
    "blues harmonica",
    # Classical
    "classical orchestra",
    "classical piano",
    "string quartet",
    "opera soprano",
    "opera baritone",
    "baroque harpsichord",
    "romantic symphony",
    "modern classical",
    "minimalist classical",
    "classical guitar",
    "chamber music",
    "choral music",
    "organ music",
    # Folk / Country / Acoustic
    "folk acoustic guitar",
    "country music",
    "bluegrass banjo",
    "country western",
    "singer songwriter",
    "americana",
    "celtic folk",
    "world folk music",
    # World / Latin / African
    "bossa nova",
    "samba",
    "reggae",
    "afrobeat",
    "flamenco",
    "indian classical sitar",
    "middle eastern music",
    "gamelan",
    "caribbean calypso",
    "salsa",
    "tango",
    # Ambient / New Age / Soundtrack
    "ambient drone",
    "nature sounds",
    "meditation music",
    "new age",
    "film soundtrack",
    "cinematic orchestral",
    "dark ambient",
    "space ambient",
    "environmental soundscape",
    # Experimental / Avant-garde
    "experimental noise",
    "musique concrete",
    "avant-garde",
    "sound collage",
    "drone music",
    "industrial music",
    "post-punk",
    # Vocal / Choral
    "a cappella vocal",
    "vocal harmony",
    "choir singing",
    "gregorian chant",
    "spoken word",
    # Instrument-centric
    "solo piano",
    "acoustic guitar fingerpicking",
    "electric guitar solo",
    "violin solo",
    "cello solo",
    "saxophone solo",
    "trumpet solo",
    "flute solo",
    "drum solo percussion",
    "bass guitar groove",
    "synthesizer",
    "harp music",
    "accordion",
    "mandolin",
    "harmonica",
    # Mood / Energy
    "energetic upbeat",
    "calm relaxing",
    "melancholy sad",
    "dark brooding",
    "joyful happy",
    "romantic love song",
    "angry aggressive",
    "mysterious eerie",
    "epic dramatic",
    "gentle lullaby",
    "party dance",
    "nostalgic",
    "dreamy ethereal",
    "intense powerful",
    "peaceful serene",
    # Era / Style
    "1950s rock and roll",
    "1960s psychedelia",
    "1970s funk",
    "1980s new wave",
    "1990s alternative",
    "early 2000s pop",
    "vintage recording",
    "lo-fi",
    "high fidelity studio",
    # Texture / Production
    "heavy distortion",
    "clean acoustic",
    "reverb drenched",
    "minimalist sparse",
    "dense layered production",
    "raw live recording",
    "polished studio pop",
    "warm analog",
    "crisp digital",
    # Rhythm
    "fast tempo",
    "slow tempo ballad",
    "syncopated rhythm",
    "waltz three four time",
    "march rhythm",
    "shuffle groove",
    "polyrhythm",
    # Specific genres
    "musical theater",
    "gospel music",
    "ska",
    "dub",
    "trip hop",
    "shoegaze",
    "post rock",
    "math rock",
    "emo",
    "death metal",
    "black metal",
    "doom metal",
    "power metal",
    "thrash metal",
    "jazz rap",
    "electro swing",
    "vaporwave",
    "lo-fi hip hop",
    "noise rock",
    "krautrock",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute CLAP text embeddings for music vocabulary"
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output exists")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        data = np.load(str(OUTPUT_PATH), allow_pickle=True)
        print(f"Already exists: {OUTPUT_PATH} ({len(data['labels'])} labels)")
        print("Use --force to rebuild.")
        return

    print(f"Embedding {len(VOCABULARY)} music descriptors...")

    from transformers import ClapModel, ClapProcessor
    from albart.pipeline.embedder import MODEL_ID
    from albart.utils import get_device

    device = get_device()
    print(f"Loading CLAP model {MODEL_ID} on {device}...")
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Embed all labels in one batch
    print("Computing text embeddings...")
    inputs = processor(text=VOCABULARY, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # L2-normalize (same as audio embeddings)
    embeddings = text_features.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Norm range: [{norms.min():.3f}, {norms.max():.3f}]")

    np.savez(
        str(OUTPUT_PATH),
        labels=np.array(VOCABULARY),
        embeddings=embeddings,
    )
    print(f"Saved → {OUTPUT_PATH}")

    # Quick sanity check: find nearest label for a few known tracks
    from albart.pipeline.embedder import FAISS_RAW_IDS_PATH
    from albart.pipeline.database import DB_PATH, get_connection

    track_emb = np.load(str(DATA_DIR / "embeddings_raw.npy")).astype(np.float32)
    track_ids = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)

    conn = get_connection(DB_PATH)
    db = {r["track_id"]: r for r in conn.execute(
        "SELECT track_id, title, artist FROM tracks"
    ).fetchall()}
    conn.close()

    print("\nSanity check — nearest text label for 10 random tracks:")
    rng = np.random.default_rng(42)
    for idx in rng.choice(len(track_ids), size=10, replace=False):
        tid = str(track_ids[idx])
        emb = track_emb[idx]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        # Cosine similarity = dot product of L2-normalized vectors
        sims = embeddings @ emb_norm
        best = int(np.argmax(sims))
        row = db.get(tid)
        name = f"{row['title']} — {row['artist']}" if row else tid
        print(f"  {name:50s} → {VOCABULARY[best]}")


if __name__ == "__main__":
    main()
