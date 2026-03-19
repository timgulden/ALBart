# ALBart — Claude Code Guide

## Project Overview

ALBart is an ambient audio visualization system. It listens to environmental sound, computes CLAP audio embeddings, and displays album art from a personal Spotify library that most closely matches the ambient audio. It runs in two phases:

- **Offline pipeline**: pulls top Spotify tracks, downloads previews and album art, computes embeddings, builds a FAISS index.
- **Runtime loop**: captures mic input, queries the FAISS index, and drives a 32x32 LED display (simulated on macOS, real HUB75 panel on Raspberry Pi 5).

Full spec: `albart_spec.md`

---

## Architecture Decisions (locked in)

### Threading Model
The runtime has three concurrent concerns. Use Python `threading` — do NOT use `asyncio`.

- **Audio thread**: `sounddevice.InputStream` callback writes into a circular `numpy` buffer. Protect buffer reads/writes with a `threading.Lock`.
- **Embedding thread**: a `threading.Thread` runs CLAP inference every N seconds, posts results to a `queue.Queue` read by the display loop.
- **Display loop (main thread)**: runs at 30fps on the main thread. pygame requires the main thread. Reads from the embedding result queue at natural cover boundaries.

### Credentials
Spotify credentials come from environment variables only — **never from config.yaml**.
- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`
- `SPOTIPY_REDIRECT_URI`

`config.yaml` must not contain actual credential values. Use comments as placeholders.

### Device Selection
MPS/CPU selection happens once, in a shared helper (`albart/utils.py`), via:
```python
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```
Never hardcode device strings elsewhere. Never make this a config option.

### Display Backend
`DisplayBackend.show_frame(rgb_array: np.ndarray)` is the **only** way to update the display. Rules:
- `display_sim.py` and `display_hub75.py` may only be imported in `runtime/run.py`.
- All other runtime code receives a `DisplayBackend` instance and calls `show_frame`.
- The backend is selected at startup based on `config.yaml:runtime.display_mode`.

### FAISS Index + ID Map
`data/faiss.index` and `data/faiss_ids.npy` are always written and read as a pair. A single helper function in `pipeline/embedder.py` owns both write operations, and a single helper in `runtime/lookup.py` owns both read operations.

### Alias Sampling, Dwell, and Brightness (locked in)
- **Sampling**: Vose's Alias Method over the full track set (all tracks, not top-N). Weights = normalized softmax over all distances. O(1) per draw.
- **Dwell time**: Absolute, not relative to the best match. `dwell = clamp(exp(-dwell_k * d_i) * max_dwell, min_dwell, max_dwell)`. Strong match → 10s; background noise → everything floors at 1s.
- **Brightness**: Nearest-neighbor distance only. `brightness = clamp(exp(-brightness_k * d_min), 0, 1)`. Fades to new target over `brightness_fade_seconds` when a new embedding arrives.
- **With-replacement**: same cover may follow itself — natural extended display for strong matches.
- `softmax_k`, `dwell_k`, `brightness_k` are independent — tune separately.
- `embedding_interval_seconds: 0` means continuous (recompute immediately after finishing).

### Config
All tunable values are in `config.yaml` and accessed through a loaded config dict passed at startup. No module-level constants for values the spec calls "configurable" or "default". Use `pyyaml` to load.

### Pipeline Idempotency
- The pipeline never deletes rows or overwrites files without `--force`.
- `embedding_status` is the canonical record of what's been processed.
- File paths stored in the database are relative to the `data/` directory.
- Album art deduplication: do NOT deduplicate by album — treat each track independently. Deduplication by `art_url` is acceptable if trivially easy, but not required.

### Spotify Track Limit
Use Spotify's `/me/top/tracks` with `time_range=long_term`, paginating at 50 per request. Accept whatever the API returns (likely ~100 unique tracks, not 1000). Do not special-case the limit.

### Logging
Use the `logging` module throughout. No bare `print` statements except in CLI entry points for user-facing progress output (where `tqdm` is preferred). Log levels:
- `INFO`: pipeline step completions, runtime startup events
- `DEBUG`: per-track pipeline steps, per-cycle embedding timing
- `WARNING`/`ERROR`: skipped tracks, download failures, inference errors

---

## Repository Structure

```
ALBart/
  albart/
    pipeline/
      __init__.py
      spotify.py          # Spotify API client and top tracks pull
      downloader.py       # Preview MP3 and album art download
      embedder.py         # CLAP inference, FAISS index build/save
      database.py         # SQLite schema and CRUD
      preprocess.py       # Image downsampling to 32x32
      run_pipeline.py     # CLI entry point: python -m albart.pipeline.run_pipeline
    runtime/
      __init__.py
      audio.py            # Mic capture, circular buffer, threading.Lock
      embedder.py         # Runtime CLAP inference (loads model once at startup)
      lookup.py           # FAISS query, dwell time computation
      display.py          # Abstract DisplayBackend base class
      display_sim.py      # pygame simulated display (prototype)
      display_hub75.py    # rpi-rgb-led-matrix display (production)
      loop.py             # Main runtime loop
      run.py              # CLI entry point: python -m albart.runtime.run
    utils.py              # Shared helpers: get_device(), load_config()
    __init__.py
  data/                   # gitignored
    db.sqlite
    faiss.index
    faiss_ids.npy
    previews/
    art_original/
    art_32/
  config.yaml
  requirements.txt
  .gitignore
  README.md
  CLAUDE.md
  albart_spec.md
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `spotipy` | Spotify API client |
| `requests` | HTTP downloads |
| `Pillow` | Image processing and downsampling |
| `torch` | CLAP model inference (MPS on Apple Silicon) |
| `transformers` | CLAP model via HuggingFace (`laion/clap-htsat-unfused`) |
| `faiss-cpu` | Vector similarity index |
| `sounddevice` | Mic capture via InputStream callback |
| `numpy` | Audio buffers, embedding math |
| `pygame` | Simulated display (macOS prototype) |
| `pyyaml` | Config file loading |
| `tqdm` | Pipeline progress bars |

CLAP model: `laion/clap-htsat-unfused`, embedding dim 512, input 48kHz mono.

---

## Development Phases

1. **Milestone 1** — Offline pipeline (Spotify pull → download → embed → FAISS index)
2. **Milestone 2** — Art review tool (pygame utility to inspect 32x32 covers)
3. **Milestone 3** — Simulated runtime (macOS, pygame display)
4. **Milestone 4** — Pi deployment (HUB75 backend, systemd service)
