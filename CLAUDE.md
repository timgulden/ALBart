# ALBart — Claude Code Guide

## Project Overview

ALBart started as an ambient audio visualization system — listening to the room and displaying matching album art on a 32x32 LED matrix. It has evolved into a full AI DJ that navigates through high-dimensional audio embedding space, with orbit navigation, mood filtering, and a web control center.

It runs in several modes:

- **DJ mode** (`python3 -m albart.dj`): automated music navigation through Spotify, controlled via web UI
- **Listener mode** (`python3 -m albart.listener`): ambient audio recognition driving an LED display
- **MapView** (`python3 -m albart.mapview`): real-time visualization of position in embedding space
- **Control Center** (`python3 -m albart.dj_server`): web API + React UI for DJ control
- **Offline pipeline** (`python3 -m albart.pipeline.run_pipeline`): builds the track database from Spotify

Full spec: `albart_spec.md`

---

## Design Principles

### Functional Architecture (core → effects → engine)

The DJ system follows a functional architecture inspired by the MAExpert project:

- **Pure logic** (`albart/core/`): All decision-making is in pure functions. They take immutable state + parameters and return `LogicResult(new_state, commands)`. No I/O, no side effects, no `time.monotonic()` calls — time is a parameter. This layer is fully unit-testable with zero infrastructure.

- **Effects** (`albart/effects/`): All side effects (Spotify API, database queries, UDP broadcast, CLAP inference) are isolated in this layer. Effects never modify DJ state — they execute commands and return results.

- **Engine** (`albart/engine.py`): The single-threaded state owner. Runs the poll loop: gather inputs → call pure logic → execute commands → publish new state. The ONLY writer of `DJState`.

- **Server** (`albart/server/`): Thin FastAPI adapter. Reads engine state via immutable snapshots (`engine.get_snapshot()`). Mutations go through `engine.update_param()`, `engine.enqueue_action()`, or direct engine methods like `engine.skip()`.

### Immutable State

`DJState` is a frozen Pydantic model. Logic functions return new copies via `model_copy(update={...})`. This eliminates race conditions by design — the server thread reads snapshots that can never be mutated out from under it.

### Commands as Data

Pure logic never executes side effects. Instead it returns command objects (`PlayTrackCommand`, `FindNeighborsCommand`, `ComputeMoodMaskCommand`, etc.) that the engine dispatches to the effects layer. This makes the logic testable and the control flow auditable.

### One Implementation Per Concept

Before the refactor, neighbor-finding was duplicated in 4 methods with inconsistent artist penalties. Now there is one `select_from_candidates()` function parameterized by penalty strength. Apply this principle to new features: parameterize, don't duplicate.

---

## Architecture Decisions

### Database: PostgreSQL + pgvector

All track metadata, 512D CLAP embeddings, and UMAP projections are stored in PostgreSQL with pgvector HNSW indexes. This replaced the previous SQLite + .npy + FAISS setup.

- `DatabaseClient` (`effects/database.py`) owns all database access
- Connection pooling via `ThreadedConnectionPool`
- Vector search via `<->` operator (L2 distance) with HNSW indexes
- Docker Compose available (`docker-compose.yml`) or use local PostgreSQL

The SQLite database (`data/db.sqlite`) still exists for backward compatibility with tools and the listener/mapview runtime, but the DJ system uses PostgreSQL exclusively.

### Embedding Space Usage

Three embedding representations exist, with two actively used for navigation:
- **512D CLAP**: captures audio texture (timbre, energy, production). Used for normal DJ hops.
- **25D UMAP**: parametric projection of 512D CLAP. Primary navigation space for both orbit dwell and transit. Trained via `tools/build_umap_25d_parametric.py`; new tracks are projected on ingest.
- **5D UMAP** (legacy): kept in the database for map display compatibility, but no longer used for navigation.

### Orbit Navigation

The orbit is a two-phase state machine cycling through anchor tracks:
- **Dwell phase**: play music near the current anchor using 25D neighbor hops.
- **Transit phase**: step through 25D space toward the next anchor.

Anchor tracks are chosen by Claude from the full library. The orbit state machine is pure (`core/orbit_logic.py`) — all time-dependent decisions take `current_time` as a parameter.

### Credentials

Spotify credentials come from environment variables only — never from config files.
- `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI`

### Display Backend (Listener/MapView)

`DisplayBackend.show_frame(rgb_array)` is the only way to update the display. Backend selected at startup from config. This applies to the listener and mapview — the DJ system uses Spotify for playback, not the display.

### Pipeline Idempotency

- The pipeline never deletes rows without `--force`
- `embedding_status` is the canonical record of what's been processed
- File paths stored in the database are relative to `data/`

### Logging

Use the `logging` module throughout. No bare `print` except in CLI entry points (where `tqdm` is preferred).

---

## Repository Structure

```
ALBart/
  albart/
    core/                     # Pure logic — NO I/O imports allowed
      state.py                # Frozen Pydantic: DJState, OrbitState, MoodState, etc.
      commands.py             # Command types + LogicResult
      navigation.py           # Main decision functions (on_poll_tick, etc.)
      orbit_logic.py          # Pure orbit state machine
      mood.py                 # Mood descriptor logic
      sampling.py             # Unified weighted track selection
      neighbor.py             # Neighbor query builder

    effects/                  # Side effects — all I/O lives here
      database.py             # PostgreSQL + pgvector client
      spotify.py              # Spotify playback control
      broadcast.py            # UDP broadcast to map display
      udp_listener.py         # Live audio embedding receiver
      migrate.py              # SQLite → PostgreSQL migration

    server/                   # FastAPI thin adapter
      app.py                  # API endpoints (reads snapshots, dispatches commands)
      models.py               # Request/response Pydantic models

    engine.py                 # DJ engine: poll loop, command dispatch, state owner

    dj.py                     # CLI entry: python3 -m albart.dj
    dj_server.py              # CLI entry: python3 -m albart.dj_server
    listener.py               # CLI entry: python3 -m albart.listener
    mapview.py                # CLI entry: python3 -m albart.mapview
    orbit.py                  # Legacy orbit (kept for tools compatibility)
    text_embedder.py          # Lazy-loading CLAP text embedding
    utils.py                  # Shared helpers: get_device(), load_config()

    pipeline/                 # Offline data pipeline
      spotify.py              # Spotify API client
      downloader.py           # Preview + art download
      embedder.py             # CLAP inference + PostgreSQL storage
      database.py             # Pipeline database access (delegates to effects/)
      preprocess.py           # Image downsampling
      deezer.py               # Deezer preview fallback
      itunes.py               # iTunes preview fallback
      run_pipeline.py         # CLI: python3 -m albart.pipeline.run_pipeline

    runtime/                  # Listener/MapView runtime (uses legacy paths)
      audio.py, embedder.py, lookup.py, loop.py, display.py, ...

  ui/                         # React frontend (Vite + TypeScript)
    src/App.tsx               # DJ web UI (API at http://127.0.0.1:8765)

  tests/                      # Unit tests (pure logic, zero infrastructure)
  tools/                      # Offline analysis and diagnostic scripts
  docker-compose.yml          # PostgreSQL + pgvector
  setup_database.py           # One-command database build
  config.yaml
  requirements.txt
  data/                       # gitignored — generated data + art files
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `psycopg2-binary` + `pgvector` | PostgreSQL + vector search |
| `spotipy` | Spotify API client |
| `torch` + `transformers` | CLAP model inference (`laion/clap-htsat-unfused`) |
| `fastapi` + `uvicorn` | DJ web server |
| `anthropic` | Claude API (mood, orbit, Voronoi labels) |
| `pydantic` | Immutable state models + API schemas |
| `numpy` | Audio buffers, embedding math |
| `sounddevice` | Mic capture (listener mode) |
| `pygame` + `PyOpenGL` | Display rendering (listener/mapview) |
| `umap-learn` | Dimensionality reduction |

CLAP model: `laion/clap-htsat-unfused`, embedding dim 512, input 48kHz mono.

---

## Development

### Running

```bash
# Start PostgreSQL (if using Docker)
docker compose up -d

# Start the DJ server with hot-reload
python3 -m albart.dj_server --reload

# Start the React UI (separate terminal)
cd ui && npm run dev

# Run tests
python3 -m pytest tests/ -v
```

### Testing

The pure core has 78 unit tests that run in <200ms with zero infrastructure — no database, no Spotify, no network. When adding navigation logic, write tests against the core functions first.

### Adding Features

1. If it's a decision (which track, when to pick, how to filter): add it to `core/` as a pure function.
2. If it needs I/O (API call, database query, file read): add it to `effects/`.
3. Wire them together in `engine.py` via the command pattern.
4. Expose to the UI via `server/app.py` endpoints.
