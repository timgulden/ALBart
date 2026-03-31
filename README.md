# ALBart

**Ambient audio visualization and DJ system.** ALBart listens to music, identifies tracks from your Spotify library using CLAP audio embeddings, and displays album art on a 32×32 LED matrix. A companion 3D visualization shows the musical neighborhood around whatever is playing, and an automated DJ navigates through your library by embedding similarity.

## Three Programs

### ALBart Listener
Captures live audio, computes CLAP embeddings, identifies the closest track in your library, and drives a 32×32 LED display (simulated on macOS, HUB75 panel on Raspberry Pi).

```bash
python -m albart.listener
python -m albart.listener --device "BlackHole"
```

### ALBart MapView
A GPU-accelerated 3D visualization of your music library. The currently playing track appears at center, surrounded by a cloud of album covers arranged by musical similarity. The cloud slowly rotates, with foreground tracks fading to keep the central cover visible.

```bash
python -m albart.mapview --mode half --view neighborhood
python -m albart.mapview --mode half --view umap
```

**Views:**
- `neighborhood` — 3D perspective cloud centered on the current track (default)
- `umap` — 2D global map with heatmap trail and Voronoi labels

**Mouse:** hover for track info tooltip, click to send track to the DJ.

### ALBart DJ
Automated music exploration through embedding space. Plays tracks from your Spotify library, hopping to nearby tracks with occasional longer jumps to fresh territory.

**CLI mode:**
```bash
python -m albart.dj                           # exact mode (default)
python -m albart.dj --mode listen             # use live audio embedding
python -m albart.dj --seed "paint it black"   # start from a specific track
python -m albart.dj --temperature 5           # tight: choose from 5 nearest
python -m albart.dj --hop-interval 20         # long hop every 20 min
python -m albart.dj --mood "chill jazz, no metal"  # mood-filtered
```

**Web UI (Control Center):**
```bash
python -m albart.dj_server                    # API server on :8765
cd ui && npm run dev                          # React UI on :5173
```
The web UI is a full control center: DJ controls, mood filtering, orbit navigation, and system management (start/stop MapView and Listener as subprocesses). Prevents macOS idle sleep automatically via `caffeinate`.

**Modes:**
- `exact` — hops based on the stored preview embedding of each track. Controlled, repeatable.
- `listen` — hops based on the Listener's live audio embedding. Reflects what the song actually sounds like but may drift at song boundaries.

**Mood filtering:**
Describe the vibe in plain text (e.g., "hipster party, indie electronic, no opera"). Claude expands this into ~20 genre descriptors, which are embedded via CLAP into the same 512D space as the audio. Tracks are pre-tagged as in-mood or out-of-mood based on cosine similarity. The DJ only hops to in-mood tracks. Mood strictness is adjustable live via the web UI.

**Orbit mode:**
Describe a musical journey (e.g., "Pacific Northwest grunge to K-pop and back"). Claude picks 5-6 anchor tracks from your library and determines whether same-artist runs are appropriate. The DJ then cycles through the anchors in two alternating phases:

- **Dwell** (~30 min per anchor): plays tracks from the anchor's neighborhood using **5D UMAP** distances, which capture genre structure. This keeps the music within the genre cluster — K-pop anchors play K-pop, grunge anchors play grunge.
- **Transit** (~10 tracks between anchors): navigates from one anchor to the next through **512D CLAP** embedding space using fractional stepping (1/10, 1/9, 1/8... of remaining distance). This bridges genres via idiosyncratic audio similarities rather than genre labels, producing smooth and surprising transitions. Transit stops early when within 15% of the target distance.

The orbit viewer is a draggable floating panel showing anchor album covers in a circle with progress bars between them (yellow = traversed, gray = upcoming). The current dwell anchor glows green.

**Overrides:**
- Change the track in Spotify → DJ detects it and continues from there
- Click a track in the MapView → DJ plays it and continues from there
- Search and play via the web UI (Next = queue, Now = interrupt)
- Skip — advance one track within the current phase
- New Set — leave dwell, start transit to the next anchor

## How They Work Together

**With BlackHole (live audio capture):**
```
Spotify → BlackHole → Listener → embeddings → MapView
                                            → LED display
```
The Listener captures system audio via BlackHole, identifies tracks, drives the LED, and sends embeddings to the MapView via UDP. The MapView shows a shifting 3D cloud that evolves as the audio changes (~every 10 seconds).

**With AirPlay speakers (no BlackHole):**
```
Spotify → AirPlay → speakers
   DJ → stored embeddings → MapView
```
The DJ controls Spotify and sends stored embeddings directly to the MapView. The visualization updates on each track change. No Listener needed.

**All three together:**
```
Spotify → BlackHole → Listener → live embeddings → MapView
   DJ → playback control                        → LED display
        ↳ stored embeddings → MapView (backup)
```
The Listener provides rich live embeddings (shifting within a song). The DJ controls what plays. Both feed the MapView — the Listener's updates dominate when running.

## Setup

### Prerequisites
- Python 3.9+
- Spotify Developer credentials ([dashboard](https://developer.spotify.com/dashboard))
- macOS with BlackHole for live audio capture (optional — DJ mode works without it)

### Install
```bash
pip install -r requirements.txt
```

### Build the Database
```bash
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"

python setup_database.py
```

This runs the full pipeline:
1. Pulls your top tracks from Spotify
2. Downloads 30-second preview clips and album art
3. Computes CLAP audio embeddings (512-dimensional)
4. Builds FAISS nearest-neighbor indices
5. Computes 2D and 5D UMAP projections for visualization
6. Optionally builds Voronoi cluster labels (requires `ANTHROPIC_API_KEY`)

The database lives in `data/` (~500MB for ~5000 tracks). Run `setup_database.py --force` to rebuild everything.

### Configure
Edit `config.yaml` for runtime tuning:

```yaml
runtime:
  embedding_interval_seconds: 10    # how often to re-embed live audio
  embedding_alpha: 1.0             # EMA smoothing (1.0 = no smoothing)
  norm_target_raw: 0.12            # RMS normalization before CLAP
  display_mode: sim                # sim (macOS) or hub75 (Raspberry Pi)

neighborhood:
  k: 100                          # neighbors for PCA basis
  base_thumb_px: 375              # album cover size at z=0
  focal_length: 0.55              # perspective depth effect
  lerp_speed: 0.3                 # position transition speed
  recompute_threshold: 0.50       # how far embedding must move to recompute PCA
```

## Architecture

### Embedding Pipeline
Audio → 3×10s chunks → CLAP (`laion/clap-htsat-unfused`) → 512-dim embeddings → L2-normalized → FAISS index.

Both raw (no normalization) and norm (RMS-normalized to 0.12) embeddings are computed. The runtime uses the norm path for better volume-independent recognition.

### Neighborhood Visualization
The 3D cloud uses a hybrid projection:
- **5D UMAP** captures global genre structure (pre-computed, stable)
- **Local PCA** (5D→3D) adapts the view to the current neighborhood
- **Actual L2 distances** (shifted so nearest = center) determine radial position
- **OpenGL** renders textured quads at float positions with mipmapping and MSAA

### DJ Navigation

**Two embedding spaces, used for different purposes:**
- **512D CLAP** (raw audio embeddings): captures what tracks *sound* like — timbre, energy, production style. Used for transit between genres and normal hop selection.
- **5D UMAP** (pre-computed projection): captures *genre structure* — which tracks belong to the same musical category. Used for dwell (staying within a genre cluster).

**Normal mode:**
- **Normal hops**: picks from the K nearest unplayed tracks (K = song temperature, 1-50) in 512D, weighted by closeness, with artist penalty to avoid same-artist runs
- **Long hops** (every ~30 min): extrapolates the trajectory of the last two tracks in 512D, landing N× further along (N = set temperature, 1-20×)

**Orbit mode (two-phase cycle):**
- **Dwell**: picks from K nearest unplayed tracks to the anchor in **5D UMAP** space (genre clustering keeps music coherent). Artist penalty 0.01× unless Claude flagged same-artist runs as appropriate.
- **Transit**: fractional stepping through **512D CLAP** space — each step covers 1/N of the remaining distance to the next anchor (N counts down from 10). Finds the nearest unplayed track to each target point. Stops early when within 15% of the initial distance. Artist penalty applied to prevent same-artist runs during genre bridging.

**Shared filters:**
- **Mood filtering**: tracks pre-tagged as in/out based on CLAP cosine similarity to mood descriptors. Stacks with orbit.
- **Near-duplicate filter**: skips tracks within L2 < 0.01 of recently played (catches remasters)

### Mood Pipeline
Free text → Claude API (expands into ~20 genre descriptors) → CLAP text embedding (same 512D space as audio) → cosine similarity against all tracks → boolean mask. The CLAP model loads on demand and auto-unloads after 5 min idle.

### Orbit Pipeline
Journey description → Claude API (picks 5-6 anchor tracks from your library + decides same-artist policy) → fuzzy match against library metadata → anchor tracks' actual 512D embeddings and 5D UMAP positions define the orbit. The orbit viewer uses 5D positions for display; navigation uses 512D for transit and 5D for dwell.

## Project Structure

```
ALBart/
  albart/
    listener.py              # Entry: python -m albart.listener
    mapview.py               # Entry: python -m albart.mapview
    dj.py                    # Entry: python -m albart.dj
    dj_server.py             # Entry: python -m albart.dj_server (control center API)
    orbit.py                 # Orbit navigation: anchors, dwell/transit state machine
    text_embedder.py         # Lazy-loading CLAP text embedding (for mood)
    utils.py                 # Shared: device selection, config loading
    pipeline/
      run_pipeline.py        # Master pipeline: pull → download → embed → index
      spotify.py             # Spotify API client
      downloader.py          # Preview + art download
      embedder.py            # CLAP inference + FAISS index build
      database.py            # SQLite schema
      preprocess.py          # Image downsampling
      deezer.py              # Deezer preview fallback
      itunes.py              # iTunes preview fallback
    runtime/
      run.py                 # Listener implementation
      run_map.py             # MapView implementation
      neighborhood_display.py # 3D OpenGL neighborhood view
      map_display.py         # 2D UMAP map view
      audio.py               # Mic capture + circular buffer
      embedder.py            # Runtime CLAP inference
      lookup.py              # FAISS query + alias sampling
      loop.py                # LED display loop
      display.py             # Display backend interface
      display_sim.py         # pygame simulated display
      display_hub75.py       # HUB75 LED panel display
      agc.py                 # Automatic gain control
      startup.py             # Startup animation
  ui/                        # React frontend (Vite + TypeScript)
    src/App.tsx              # DJ web UI
  tools/
    build_umap.py            # 2D UMAP projection
    build_umap_5d.py         # 5D UMAP projection
    build_voronoi.py         # Voronoi cluster labels (uses Claude API)
    build_text_labels.py     # CLAP text embeddings
    dj_simulate.py           # Offline DJ trajectory simulation
    test_transit.py          # Offline transit strategy testing
    review_art.py            # 32×32 art inspection
    reembed_existing.py      # Re-embed tracks (schema migration)
    archive/                 # Diagnostic scripts from development
  setup_database.py          # One-command database build
  config.yaml                # All tunable parameters
  data/                      # Generated data (gitignored)
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `spotipy` | Spotify API |
| `torch` + `transformers` | CLAP model inference |
| `faiss-cpu` | Nearest-neighbor search |
| `umap-learn` | Dimensionality reduction |
| `pygame` + `PyOpenGL` | Display + GPU rendering |
| `sounddevice` | Audio capture |
| `numpy` | Everything numerical |
| `fastapi` + `uvicorn` | DJ web server |
| `anthropic` | Claude API (mood interpretation, Voronoi labels) |

## License

Personal project. Not licensed for redistribution.
