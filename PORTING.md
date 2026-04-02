# Porting ALBart to a Non-Spotify Music Source

ALBart was built with Spotify as its music source, but the architecture
cleanly separates music-source concerns from the core navigation,
embedding, and visualization systems.  This guide explains what you need
to build to run ALBart against a different music library and player.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Pure Logic (core/)                       │
│  Navigation, orbit, mood, sampling — no I/O, no player      │
│  dependency.  Works with track IDs (opaque strings) and      │
│  PlaybackSnapshot (a simple struct).                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Engine (engine.py)                       │
│  Poll loop: reads playback state, calls pure logic,          │
│  dispatches commands.  Talks to the player exclusively       │
│  through the PlaybackClient protocol.                        │
└──────┬──────────┬───────────────────┬──────────────────────┘
       │          │                   │
  PlaybackClient  DatabaseClient  BroadcastClient
  (your player)   (PostgreSQL)    (UDP → MapView)
```

The engine never imports Spotify.  It accepts any object that satisfies
the `PlaybackClient` protocol.  The pure logic layer (`core/`) doesn't
even know a player exists — it works entirely with immutable state
snapshots and command objects.

---

## What You Need to Build

### 1. PlaybackClient (required)

Defined in `albart/effects/playback.py`.  Six methods:

```python
from albart.effects.playback import PlaybackClient
from albart.core.state import PlaybackSnapshot
import time

class MyPlayerClient:
    """Wraps your music player for the ALBart engine."""

    def poll_playback(self) -> PlaybackSnapshot:
        """Called every ~5s.  Return current playback state."""
        # Query your player for what's currently playing
        return PlaybackSnapshot(
            current_track_id="your-track-id",  # your ID scheme
            progress_ms=12000,                  # ms into the track
            duration_ms=240000,                 # total track length
            is_playing=True,
            volume=75,                          # 0-100, or -1 if unknown
            snapshot_time=time.monotonic(),      # when you polled
        )

    def play_track(self, track_id: str) -> bool:
        """Start playing a specific track.  Return True on success."""
        # Tell your player to play this track
        ...

    def resume(self) -> bool:
        """Resume paused playback."""
        ...

    def pause(self) -> bool:
        """Pause playback."""
        ...

    def seek(self, position_ms: int) -> bool:
        """Seek to a position in the current track."""
        ...

    def set_volume(self, volume: int) -> bool:
        """Set volume (0-100)."""
        ...
```

The `track_id` is an opaque string.  It can be anything — a file path,
a database primary key, a MusicBrainz ID, a hash.  The engine passes
it around but never parses it.  The only requirement is that the same
string consistently identifies the same track.

**PlaybackSnapshot** is a frozen Pydantic model with these fields:

| Field | Type | Purpose |
|---|---|---|
| `current_track_id` | `str \| None` | Track currently playing (None if nothing) |
| `progress_ms` | `int` | Playback position in milliseconds |
| `duration_ms` | `int` | Total track duration in milliseconds |
| `is_playing` | `bool` | Whether audio is actively playing |
| `volume` | `int` | Volume 0-100, or -1 if unknown |
| `snapshot_time` | `float` | `time.monotonic()` when you polled |

The engine uses `snapshot_time` to estimate elapsed time between polls,
so it should be set to the moment you read the player state, not when
you constructed the object.

### 2. MetadataClient (optional)

If your system can resolve track IDs to metadata quickly (e.g. from a
database), implement `MetadataClient`:

```python
from albart.effects.playback import MetadataClient

class MyPlayerClient(MetadataClient):
    def get_track_metadata(self, track_id: str) -> tuple[str, str] | None:
        """Return (title, artist) for a track ID, or None."""
        row = my_db.lookup(track_id)
        if row:
            return (row.title, row.artist)
        return None
```

This is used by the engine to display title/artist on MapView for
tracks that haven't been fully ingested yet.  If you don't implement
it, the engine shows metadata only after ingestion completes.

### 3. Track Ingestion Pipeline (required for navigation)

The DJ navigates through embedding space.  Every track needs:

- **512D CLAP embedding** — audio fingerprint capturing timbre, energy,
  and production style
- **25D UMAP projection** — dimensionality reduction preserving genre
  structure, used for neighbor search

Without these, the track exists in the database but the DJ can't
navigate from it.

#### What CLAP needs

CLAP (`laion/clap-htsat-unfused`) takes mono float32 audio at 48kHz.
It doesn't care about the source — WAV, FLAC, MP3, or raw PCM all
work once decoded to a numpy array.

```python
import librosa
import numpy as np
from albart.pipeline.embedder import embed_audio, load_model

# Load your audio file (any format librosa supports)
audio, _ = librosa.load("track.flac", sr=48000, mono=True, dtype="float32")

# Load CLAP model (do this once, reuse for all tracks)
model, processor, device = load_model(allow_mps=True)

# Embed (3x10s chunks averaged, same as the pipeline)
sr = 48000
chunk_samples = 10 * sr
n_chunks = 3
total = n_chunks * chunk_samples

if len(audio) < total:
    audio = np.pad(audio, (total - len(audio), 0))

chunk_embs = []
for i in range(n_chunks):
    chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
    chunk_embs.append(embed_audio(chunk, model, processor, device, norm_target=0.12))

emb_512 = np.mean(chunk_embs, axis=0).astype(np.float32)
emb_512 = emb_512 / (np.linalg.norm(emb_512) + 1e-8)
```

The `norm_target=0.12` parameter applies dynamic range compression
and a 4kHz low-pass filter before CLAP inference.  This was tuned
for consistency across different recording qualities.  Use the same
value for all tracks.

**Note on audio quality:** ALBart's Spotify pipeline embeds 30-second
preview clips.  If you have full-length high-quality audio, your
embeddings may actually be *better*.  The 3x10s chunk averaging
captures a representative sample of the track's character.  For very
long tracks (>10 minutes), you might want to sample chunks from
different parts of the track rather than just the last 30 seconds.

#### Projecting to 25D

After computing the 512D embedding, project it to 25D using the
parametric UMAP model:

```python
from albart.effects.umap_projector import UmapProjector
from pathlib import Path

projector = UmapProjector.load(Path("data/umap_25d_model/model.pt"))
umap_25d = projector.project(emb_512)  # (25,) float32
```

The parametric UMAP is a small MLP trained to approximate the full
UMAP transformation.  It handles new tracks that weren't in the
original training set.

**Caveat:** If you add a very large number of tracks from a genre
not represented in the original library, the 25D projections for those
tracks will be approximate.  After a large ingestion (1000+ tracks),
consider retraining the UMAP model:

```bash
python3 tools/build_umap_25d_parametric.py --force
```

This recomputes all 25D projections from scratch and retrains the MLP.
Takes about a minute for 5000 tracks.

### 4. Database Records

Every track needs a row in the `tracks` table:

```sql
INSERT INTO tracks (
    track_id, title, artist, album,
    art_path_original, art_path_32,
    embedding_status, embedding_512, umap_25d, artist_lower
) VALUES (
    'your-track-id',
    'Track Title',
    'Artist Name',
    'Album Name',
    'art_original/your-track-id.jpg',   -- or NULL
    'art_32/your-track-id.png',         -- 32x32 for LED display, or NULL
    'ok',                                -- 'ok' = has embedding
    embedding_vector,                    -- 512D float array
    umap_vector,                         -- 25D float array
    'artist name'                        -- lowercase for filtering
);
```

Art paths are relative to the `data/` directory.  The 32x32 version
is only needed for the LED display prototype.  MapView uses
`art_path_original`.

You can use the `DatabaseClient` directly:

```python
from albart.effects.database import DatabaseClient, DatabaseConfig

db = DatabaseClient(DatabaseConfig())
db.upsert_track(
    track_id="your-id",
    title="Title",
    artist="Artist",
    album="Album",
    art_path_original="art_original/your-id.jpg",
    embedding_status="ok",
    embedding_512=emb_512,
)
db.upsert_umap_25d("your-id", umap_25d)
```

### 5. Album Art

MapView displays album art thumbnails loaded from disk.  Store art
images in `data/art_original/` as JPG or PNG, any reasonable size
(300x300 is typical).  The path stored in the database should be
relative to `data/`.

For the LED display prototype, also provide 32x32 pixel versions in
`data/art_32/`.

---

## Wiring It Together

```python
from albart.effects.broadcast import BroadcastClient
from albart.effects.database import DatabaseClient, DatabaseConfig
from albart.effects.udp_listener import UDPListener
from albart.engine import Engine

db = DatabaseClient(DatabaseConfig())
player = MyPlayerClient()  # your PlaybackClient implementation
broadcast = BroadcastClient()
udp = UDPListener()

engine = Engine(
    db=db,
    playback=player,
    broadcast=broadcast,
    udp_listener=udp,
    mode="exact",
    song_k=10,
)
engine.run()
```

MapView, the web UI, and the control center all work unchanged —
they read state from the engine and don't know or care what player
is behind it.

---

## What Doesn't Need to Change

| Component | Why it's player-agnostic |
|---|---|
| Pure logic (`core/`) | Works with track IDs and PlaybackSnapshot |
| Engine (`engine.py`) | Uses PlaybackClient protocol only |
| MapView | Receives embeddings via UDP, renders with OpenGL |
| Web UI (`ui/`) | Talks to the server API, not the player |
| Server (`server/app.py`) | Calls engine methods, not player methods |
| Database | Track IDs are opaque strings |
| CLAP embeddings | Takes numpy audio arrays from any source |
| UMAP projections | Works on 512D embeddings, source-agnostic |
| Orbit navigation | Pure math on embeddings |
| Mood filtering | CLAP text embeddings, no player dependency |

---

## What Changes Per-Backend

| Concern | What to implement |
|---|---|
| **Playback control** | `PlaybackClient` (6 methods) |
| **Metadata lookup** | `MetadataClient` (1 method, optional) |
| **Track ingestion** | Script to scan your library, extract audio, compute CLAP + UMAP, store in PostgreSQL |
| **Album art** | Copy/extract art files to `data/art_original/` |
| **Track IDs** | Decide your ID scheme (file hash, DB key, MusicBrainz ID) |

---

## Example: MPD Backend

MPD (Music Player Daemon) is a common local music server.  Here's a
sketch of what `MPDClient` might look like:

```python
from mpd import MPDClient as MPD
from albart.core.state import PlaybackSnapshot
import time

class MPDPlaybackClient:
    def __init__(self, host="localhost", port=6600):
        self._mpd = MPD()
        self._mpd.connect(host, port)

    def poll_playback(self) -> PlaybackSnapshot:
        status = self._mpd.status()
        song = self._mpd.currentsong()
        return PlaybackSnapshot(
            current_track_id=song.get("file", ""),  # file path as ID
            progress_ms=int(float(status.get("elapsed", 0)) * 1000),
            duration_ms=int(float(status.get("duration", 0)) * 1000),
            is_playing=status.get("state") == "play",
            volume=int(status.get("volume", -1)),
            snapshot_time=time.monotonic(),
        )

    def play_track(self, track_id: str) -> bool:
        self._mpd.clear()
        self._mpd.add(track_id)  # track_id is the file path
        self._mpd.play()
        return True

    def resume(self) -> bool:
        self._mpd.pause(0)
        return True

    def pause(self) -> bool:
        self._mpd.pause(1)
        return True

    def seek(self, position_ms: int) -> bool:
        self._mpd.seekcur(position_ms / 1000.0)
        return True

    def set_volume(self, volume: int) -> bool:
        self._mpd.setvol(max(0, min(100, volume)))
        return True
```

---

## Bulk Ingestion for Large Libraries

For a large local collection, you'd write a script that:

1. Scans your music database/directory for audio files
2. Reads metadata from tags (using `mutagen` or `tinytag`)
3. Extracts or downloads album art
4. Computes CLAP embeddings (the GPU bottleneck — ~5-10s per track)
5. Projects to 25D via the parametric UMAP
6. Stores everything in PostgreSQL

The existing pipeline functions are reusable:

- `albart.pipeline.embedder.embed_audio()` — CLAP inference
- `albart.pipeline.embedder.load_model()` — load CLAP model once
- `albart.pipeline.preprocess.downsample_art()` — 32x32 art
- `albart.effects.umap_projector.UmapProjector` — 512D → 25D
- `albart.effects.database.DatabaseClient` — PostgreSQL storage

For a million-track library, budget roughly 10 seconds per track
for CLAP embedding on Apple Silicon (MPS) or CUDA GPU.  That's
~115 days for 1M tracks on a single GPU.  The embedding step is
embarrassingly parallel — split across multiple machines or GPUs if
needed.  Everything else (metadata, art, UMAP projection, DB insert)
is fast.

After bulk ingestion, retrain the parametric UMAP to incorporate the
new tracks:

```bash
python3 tools/build_umap_25d_parametric.py --force
```

---

## Database Requirements

ALBart uses PostgreSQL with the pgvector extension for vector search.
The HNSW indexes support approximate nearest-neighbor queries at scale.
The current index parameters (`m=16, ef_construction=200`) give >99%
recall at 1M+ scale.

```bash
# Docker (easiest)
docker compose up -d

# Or install locally
brew install postgresql pgvector   # macOS
# then: CREATE EXTENSION vector;
```

Connection defaults: `localhost:5432`, database `albart`, user `albart`,
password `albart`.  Configure in `DatabaseConfig` or environment.
