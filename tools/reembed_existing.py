"""Re-embed all already-embedded tracks using 3x10s chunks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
from tqdm import tqdm
from albart.pipeline import embedder, database
from albart.utils import DATA_DIR

CHUNK_SAMPLES = 10 * 48000

conn = database.get_connection()
model, processor, device = embedder.load_model()

rows = conn.execute(
    "SELECT track_id, preview_path FROM tracks "
    "WHERE embedding_status='ok' AND preview_path IS NOT NULL"
).fetchall()

print(f"Re-embedding {len(rows)} tracks with 3x10s chunks...")

new_embeddings = []
new_ids = []

for row in tqdm(rows, desc="Re-embedding"):
    track_id, preview_path = row[0], row[1]
    path = DATA_DIR / preview_path
    try:
        audio, _ = librosa.load(str(path), sr=48000, mono=True, dtype="float32")
        for i in range(3):
            start = i * CHUNK_SAMPLES
            chunk = audio[start:start + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            emb = embedder.embed_audio(chunk, model, processor, device)
            new_embeddings.append(emb)
            new_ids.append(track_id)
    except Exception as e:
        print(f"Error on {track_id}: {e}")

# Replace old single-chunk entries with new 3-chunk ones
EMBEDDINGS_STORE = DATA_DIR / "embeddings.npy"
IDS_STORE        = DATA_DIR / "faiss_ids.npy"

prev_embeddings = np.load(str(EMBEDDINGS_STORE))
prev_ids = np.load(str(IDS_STORE), allow_pickle=True).tolist()
new_id_set = set(new_ids)

keep = [i for i, tid in enumerate(prev_ids) if tid not in new_id_set]
all_embeddings = np.concatenate(
    [prev_embeddings[keep], np.stack(new_embeddings)], axis=0
).astype(np.float32)
all_ids = [prev_ids[i] for i in keep] + new_ids

np.save(str(EMBEDDINGS_STORE), all_embeddings)
embedder.build_and_save_index(all_embeddings, all_ids)
print(f"Done. Index: {len(all_ids)} vectors / {len(set(all_ids))} unique tracks.")
