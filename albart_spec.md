# ALBart: Ambient Audio Album Art Display
## Functional and Technical Specification

---

## 1. Project Overview

ALBart is an ambient audio visualization system that continuously listens to environmental sound, computes audio embeddings, and displays the album art of songs from a personal music library that most closely match the ambient audio. The display cycles through the closest matches, spending more time on better matches, and crossfades between covers.

The system runs in two phases:

- **Offline pipeline**: pulls a user's top Spotify tracks, downloads 30-second audio previews and album art, computes CLAP embeddings, and stores everything in a local database.
- **Runtime loop**: captures mic input, computes embeddings on a rolling basis, queries the database for nearest neighbors, and drives a 32x32 display.

The initial prototype runs on macOS with a simulated 32x32 display window. The production target is a Raspberry Pi 5 driving a HUB75 32x32 LED matrix panel.

---

## 2. Functional Requirements

### 2.1 Offline Pipeline

**Spotify data pull**

- Authenticate with Spotify via OAuth (Authorization Code flow)
- Pull the user's top tracks using the `long_term` time horizon only
- Retrieve up to 1000 tracks (paginate at 50 per request)
- For each track, collect:
  - Track ID, title, artist(s), album name
  - Preview URL (30-second MP3); flag and skip tracks with no preview URL
  - Album art URL (300x300 resolution preferred)
- Write a manifest file (CSV or SQLite) logging all tracks, their status (preview available / no preview), and file paths

**Audio embedding**

- Download each preview MP3
- Run the CLAP audio encoder to produce a fixed-length embedding vector
- Store embeddings in a FAISS flat index keyed by track ID
- Store the FAISS index as a file on disk

**Album art processing**

- Download each album art image
- Downsample to 32x32 pixels using high-quality resampling (Lanczos)
- Store as raw RGB (PNG) at 32x32
- Also retain the original downloaded image for reference

**Database**

- SQLite database with a single `tracks` table:
  - `track_id` (text, primary key)
  - `title` (text)
  - `artist` (text)
  - `album` (text)
  - `preview_url` (text, nullable)
  - `preview_path` (text, nullable)
  - `art_url` (text)
  - `art_path_original` (text)
  - `art_path_32` (text)
  - `embedding_status` (text: `ok`, `no_preview`, `error`)

**Re-run behavior**

- The pipeline should be safe to re-run incrementally: skip tracks already in the database with `embedding_status = ok`
- Support a `--force` flag to reprocess all tracks

### 2.2 Runtime Loop

**Audio capture**

- Capture mic input continuously using the system default input device
- Use a rolling buffer of configurable length (default: 10 seconds)
- Compute a new embedding every N seconds (default: 10 seconds, configurable)

**Embedding and lookup**

- Run CLAP audio encoder on the current buffer to produce an embedding
- Query FAISS index for the top 10 nearest neighbors by L2 distance
- Convert distances to dwell weights using a softmax with a configurable temperature parameter k: `weight = exp(-k * distance)`
- Normalize weights to sum to 1.0

**Display loop**

- Maintain a current playlist of 10 tracks with associated dwell times (proportional to weights, summing to a configurable cycle length, default: 60 seconds)
- Display each cover for its allocated dwell time
- Crossfade between covers over a configurable transition duration (default: 0.75 seconds)
- When a new embedding is computed, generate a new playlist and transition to it at the next natural cover boundary (do not interrupt a crossfade in progress)

**Display output**

- Prototype: render the 32x32 content scaled up to a window (each pixel rendered as a 16x16 block) using pygame
- Production: drive HUB75 panel via rpi-rgb-led-matrix library
- The display backend should be abstracted behind an interface so the prototype and production targets share all other code

---

## 3. Technical Specification

### 3.1 Repository Structure

```
albart/
  pipeline/
    __init__.py
    spotify.py          # Spotify API client and top tracks pull
    downloader.py       # Preview MP3 and album art download
    embedder.py         # CLAP inference, FAISS index build
    database.py         # SQLite schema and access
    preprocess.py       # Image downsampling
    run_pipeline.py     # CLI entry point for offline pipeline
  runtime/
    __init__.py
    audio.py            # Mic capture, rolling buffer
    embedder.py         # Runtime CLAP inference (shared model load)
    lookup.py           # FAISS query, weight computation
    display.py          # Abstract display interface
    display_sim.py      # pygame simulated display (prototype)
    display_hub75.py    # rpi-rgb-led-matrix display (production)
    loop.py             # Main runtime loop
    run.py              # CLI entry point for runtime
  data/
    db.sqlite           # SQLite database (gitignored)
    faiss.index         # FAISS index file (gitignored)
    previews/           # Downloaded MP3s (gitignored)
    art_original/       # Downloaded art at source resolution (gitignored)
    art_32/             # Processed 32x32 art (gitignored)
  config.yaml           # Runtime configuration
  requirements.txt
  README.md
```

### 3.2 Dependencies

```
spotipy                 # Spotify API client
requests                # HTTP downloads
Pillow                  # Image processing
torch                   # PyTorch (MPS on Apple Silicon, CPU fallback)
transformers            # CLAP model via HuggingFace
faiss-cpu               # Vector index (faiss-gpu if CUDA available)
sounddevice             # Mic capture
numpy                   # Audio buffer handling
pygame                  # Simulated display
pyyaml                  # Config file
tqdm                    # Pipeline progress bars
```

For production Pi deployment, `rpi-rgb-led-matrix` is added and `pygame` is optional.

### 3.3 CLAP Model

- Use `laion/clap-htsat-unfused` from HuggingFace as the default checkpoint
- Load once at startup, shared between pipeline and runtime
- On Apple Silicon, use `device = "mps"` if available, otherwise `"cpu"`
- On Pi 5, use `"cpu"` with optional INT8 quantization via ONNX Runtime (deferred to a later milestone)
- Embedding dimension: 512
- Input: audio array at 48kHz mono; the HuggingFace processor handles resampling

### 3.4 Audio Capture

- Use `sounddevice.InputStream` with a callback that writes into a circular buffer
- Buffer length: configurable, default 10 seconds at 48kHz mono = 480,000 samples
- On each embedding cycle, copy the current buffer contents and pass to CLAP processor
- Embedding cycle interval: configurable in `config.yaml`, default 10 seconds

### 3.5 FAISS Index

- Index type: `faiss.IndexFlatL2` (exact search, sufficient for 1000 vectors)
- Stored as `data/faiss.index`
- Parallel array `data/faiss_ids.npy` maps FAISS integer indices to `track_id` strings
- At runtime, load both files at startup

### 3.6 Weight and Dwell Computation

```python
import numpy as np

def compute_dwell_times(distances, cycle_length_seconds=60.0, k=5.0):
    weights = np.exp(-k * distances)
    weights /= weights.sum()
    dwell_times = weights * cycle_length_seconds
    return dwell_times
```

The parameter `k` controls how sharply the nearest match dominates. Higher k = more time on the best match. Default 5.0 is a reasonable starting point; expose in `config.yaml`.

### 3.7 Display Interface

```python
class DisplayBackend:
    def show_frame(self, rgb_array: np.ndarray) -> None:
        """Accept a (32, 32, 3) uint8 RGB array and render it."""
        raise NotImplementedError
```

`display_sim.py` implements this with pygame, scaling each pixel to a 16x16 block for a 512x512 window. `display_hub75.py` implements it using the rpi-rgb-led-matrix Python bindings.

### 3.8 Crossfade

Linear pixel interpolation between current and target frame:

```python
def crossfade(frame_a, frame_b, alpha):
    # alpha: 0.0 = frame_a, 1.0 = frame_b
    return (frame_a * (1 - alpha) + frame_b * alpha).astype(np.uint8)
```

Advance alpha from 0 to 1 over the transition duration at the display refresh rate (target 30fps).

### 3.9 Configuration

`config.yaml` exposes:

```yaml
spotify:
  client_id: ""
  client_secret: ""
  redirect_uri: "http://localhost:8888/callback"

pipeline:
  max_tracks: 1000
  art_source_size: 300   # pixel width of source art to request from Spotify

runtime:
  embedding_interval_seconds: 10
  buffer_length_seconds: 10
  top_n_neighbors: 10
  cycle_length_seconds: 60
  softmax_k: 5.0
  crossfade_seconds: 0.75
  display_fps: 30
  display_mode: "sim"    # "sim" or "hub75"
  sim_scale: 16          # pixels per LED in sim mode
```

### 3.10 CLI Entry Points

**Pipeline:**
```
python -m pipeline.run_pipeline [--force]
```
Runs all pipeline steps in order. Skips tracks already processed unless `--force` is passed. Prints a summary on completion: total tracks, successful embeddings, skipped (no preview), errors.

**Runtime:**
```
python -m runtime.run
```
Loads config, initializes display backend, starts audio capture and main loop.

---

## 4. Milestones

### Milestone 1: Offline Pipeline
- Spotify OAuth and top tracks pull
- Preview and art download with status tracking
- CLAP embedding and FAISS index build
- SQLite database with full manifest

### Milestone 2: Art Review Tool
- Simple script that cycles through all 32x32 processed covers in a pygame window
- Allows flagging covers that look unreadable for manual review or re-cropping
- Not part of the runtime; a developer/curation utility

### Milestone 3: Simulated Runtime (Mac prototype)
- Mic capture and rolling buffer
- Runtime CLAP embedding
- FAISS lookup and dwell time computation
- Simulated 32x32 display with crossfade

### Milestone 4: Pi Deployment
- rpi-rgb-led-matrix display backend
- Benchmarking of CLAP inference time on Pi 5 CPU
- Optional: ONNX quantized CLAP model if inference is too slow
- Systemd service for autostart

---

## 5. Open Questions and Deferred Decisions

- **ONNX quantization**: defer until Pi 5 inference speed is benchmarked. May not be needed.
- **Tracks with no preview**: currently skipped. Could later be handled by sourcing a clip from a local file or a different API.
- **Art curation**: the Milestone 2 review tool is manual. A future pass could automate detection of low-information covers (e.g., high-entropy noise after downsampling) and flag them.
- **Embedding drift detection**: a future enhancement could trigger a new embedding cycle early when the running audio embedding drifts significantly from the one that generated the current playlist, improving responsiveness to song changes.
- **Multi-song albums**: multiple tracks from the same album will share identical art. This is fine and expected behavior.
