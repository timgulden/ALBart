# ALBart Music Map — Design & Reference

## Overview

The Music Map is a second visualization for ALBart, running alongside the 32×32 LED display. It projects all ~5,200 tracks in the library onto a 2D plane using UMAP dimensionality reduction of their CLAP embeddings, then animates the live mic audio against that map in real time.

The visualization runs as a **separate process** from the main runtime, receiving embeddings over a local UDP socket. If the map process isn't running, the main runtime is unaffected.

---

## What It Shows

```
┌─────────────────────────────────────────────────────┐
│  · · ·  [tiny thumbnails at UMAP positions]  · · ·  │
│                                                      │
│     ·····trail·····●  ← grey dot (live audio)       │
│                                                      │
│              [BIG COVER]  ← RRF top-1 match,        │
│              Title — Artist   at its map position    │
└─────────────────────────────────────────────────────┘
```

**Thumbnails** — All ~5,200 tracks rendered as micro-thumbnails (10px at 1080p / 20px at 4K) at their fixed UMAP positions on a black canvas. Pre-rendered once at startup to a single surface. The spatial layout reflects musical similarity: tracks with similar CLAP embeddings cluster together.

**Grey dot** — The live audio embedding, projected into the UMAP space via `model.transform()`. This moves continuously as the mic input changes. It shows *where the current sound is* in music space, independent of any matching.

**Fading trail** — A history of recent dot positions. Each trail point is colored white, with brightness = `confidence_at_capture × (1 − age/max_age)`. Strong confident moments leave bright marks; background noise leaves dim ones. Trail length is configurable (default 20 minutes).

**Large cover + label** — The RRF top-1 match from the main process (same track the LED display is showing), drawn at **that track's own UMAP position** on the map. The cover scales with confidence; a `brightness_min` floor keeps it visible even during uncertain periods. The gap (or lack thereof) between the grey dot and the cover is itself diagnostic: a small gap means the live embedding and the best match are spatially close in music space; a large gap means they're not.

---

## Architecture

### Separate Process + UDP

The map runs as a separate process (`python3 -m albart.runtime.run_map`). The main runtime's `DisplayLoop` broadcasts a UDP packet on each new embedding:

```
Main process                          Map process
──────────────                        ───────────
DisplayLoop._check_embedding_queue()
  → RRF query → AliasTable
  → broadcast({                  UDP  → map_display.update(
      raw:        emb_raw,      ────►      emb_raw, emb_norm,
      norm:       emb_norm,                top1_track_id,
      top1:       track_ids[0],            d_min_raw
      d_min_raw:  table.d_min_raw         )
    })
```

Payload is ~4KB (two 512-dim float32 arrays + metadata). If the map process isn't running, `sendto()` is a silent no-op.

The broadcast port is configurable in `config.yaml` (`map_broadcast_port: 57001`). Set to `0` to disable broadcasting entirely.

### Why the broadcast comes from DisplayLoop, not EmbeddingWorker

The `top1_track_id` and `d_min_raw` are only available after the RRF dual-index FAISS query, which happens in `DisplayLoop._check_embedding_queue()`. Broadcasting from there ensures the map display shows exactly the same best-guess track as the LED panel.

### UMAP Transform Thread

Calling `umap_model.transform()` on a single point takes ~1ms (measured on this hardware). The transform runs on a background thread (`UMAPTransformer`) so it never blocks the 30fps render loop, but in practice the latency is negligible.

The transformer always returns the most recently completed result. If `update()` is called faster than `transform()` can process, intermediate embeddings are dropped (only the latest pending call is held). This is appropriate for a visualization that updates once per second.

---

## Files

| File | Role |
|---|---|
| `tools/build_umap.py` | Offline: fits UMAP, writes `data/umap_2d.npy`, `umap_ids.npy`, `umap_model.joblib` |
| `albart/runtime/map_display.py` | `MapDisplay` class — all rendering logic |
| `albart/runtime/run_map.py` | Entry point: UDP socket + render loop |
| `data/umap_2d.npy` | (N, 2) float32 projected coordinates |
| `data/umap_ids.npy` | (N,) track ID strings, row-aligned with umap_2d |
| `data/umap_model.joblib` | Fitted UMAP model for `transform()` at runtime |

Main runtime changes (minimal):
- `albart/runtime/loop.py` — `DisplayLoop` accepts `map_broadcast` callable, calls it after each RRF query
- `albart/runtime/run.py` — creates UDP socket, passes `_broadcast_map` to `DisplayLoop`
- `albart/runtime/embedder.py` — `EmbeddingWorker` accepts optional `on_embedding` callback (wired up but not used for broadcast; broadcast is from `DisplayLoop` instead)

---

## Setup

### 1. Build the UMAP model (one-time, ~15 seconds)

```bash
python3 tools/build_umap.py
```

Options:
```bash
python3 tools/build_umap.py --n-neighbors 15 --min-dist 0.1 --metric cosine
python3 tools/build_umap.py --force   # rebuild even if outputs exist
```

The model is fitted on the raw FAISS index vectors (5,191 L2-normalized 512-dim CLAP embeddings). Coordinate ranges from the initial fit: x ∈ [−2.78, 9.12], y ∈ [−2.12, 7.89].

Re-run `build_umap.py --force` any time tracks are added to the library or after a pipeline re-run.

### 2. Run the map display

```bash
# In a second terminal while the main runtime is running:
python3 -m albart.runtime.run_map

# Options:
python3 -m albart.runtime.run_map --mode half          # 1920×1080 (default, for monitor)
python3 -m albart.runtime.run_map --mode full          # 3840×2160 (4K TV via HDMI)
python3 -m albart.runtime.run_map --trail-minutes 30   # longer trail history
python3 -m albart.runtime.run_map --port 57001         # UDP port (match config.yaml)
```

### 3. Run both together

```bash
# Terminal 1
python3 -m albart.runtime.run

# Terminal 2
python3 -m albart.runtime.run_map --mode half
```

The map window can be opened and closed independently without affecting the main runtime.

---

## Configuration (`config.yaml`)

```yaml
runtime:
  map_broadcast_port: 57001  # UDP port; 0 = disable broadcasting

  # These are also used by the map for confidence/cover sizing:
  brightness_k: 80.0
  brightness_floor: 0.003
  brightness_power: 2.0
  brightness_min: 0.15       # floor — keeps cover visible even with poor match

umap:
  n_neighbors: 15    # larger = more global structure, slower fit
  min_dist: 0.1      # smaller = tighter clusters
  metric: cosine     # cosine matches CLAP's contrastive objective
```

`brightness_min` was added in this session to keep the cover visible during room-mic recording conditions where `d_min_raw` is high and raw confidence approaches zero. Set to `0.0` to restore the original behavior where covers fully dim when there's no good match.

---

## Display Modes

| Mode | Window | Thumbnail | Use |
|---|---|---|---|
| `half` | 1920×1080 | 10×10 px | Development, side monitor |
| `full` | 3840×2160 | 20×20 px | Samsung QN55Q7FDA (4K TV via HDMI) |

All layout math is in normalized [0,1]² UMAP space and scaled to canvas size, so the two modes produce identical layouts.

The `full` mode is the eventual target for Pi deployment via direct HDMI to the TV.

---

## Design Decisions

**Why UMAP over other dimensionality reduction?** The CLAP embedding space is 512-dimensional; UMAP preserves local neighborhood structure better than PCA for high-dimensional semantic embeddings, and the resulting layout reflects musical similarity in a way that's visually meaningful (genres, energy levels, and moods cluster naturally).

**Why separate process?** The main runtime's pygame window must run on the main thread. A second pygame window in the same process would compete for the main thread. Separate process also means the map can crash or be killed without affecting the LED display, and the Pi can boot the map conditionally based on whether an HDMI display is connected.

**Why broadcast from DisplayLoop, not EmbeddingWorker?** The `top1_track_id` required by the map only exists after the RRF dual-index FAISS query in `DisplayLoop`. Broadcasting from there guarantees the map and LED show the same track.

**Why draw the cover at the track's map position, not near the dot?** The gap between dot and cover is informative. When the system is confident and correct, they should be spatially close in UMAP space. When they're far apart, it means the live embedding is landing in a different musical neighborhood than the best-match track — a visible diagnostic for the identification accuracy issues.

**Why not label the spatially nearest track?** Tried in session: the nearest track on the map changes rapidly as the dot moves through dense regions and the label flickers. The RRF top-1 from the main process is more stable and matches what the LED display shows, making the two displays coherent.

---

## Known Issues / Future Work

- **Dot stuck in one region**: Room-mic recordings of music tend to project into the same UMAP region (quiet/ambient cluster) regardless of what's playing, because the room acoustic embedding differs from the clean studio preview embedding. This is the fundamental identification accuracy problem being worked on separately. When identification accuracy improves, the dot should navigate the full map.

- **Cover always near lower-left**: Same cause — the best match for room-mic audio tends to be ambient/quiet music. As accuracy improves, the cover will jump to the correct region.

- **No Pi/TV autostart yet**: The systemd service and HDMI display detection for Pi deployment have not been implemented. The map is macOS-only for now. When ready, `run_map.py` should check `$DISPLAY` / `$WAYLAND_DISPLAY` on Linux before initializing pygame and exit cleanly if no display is available.

- **Trail only starts accumulating on first embedding**: Before the first UDP packet arrives, the canvas shows just the thumbnail background. This is correct behavior.

- **UMAP transform of out-of-distribution embeddings**: Live embeddings that are very different from any training point may project outside the [0,1]² normalized range. Currently clamped to canvas bounds. Could be visualized differently (e.g., an off-edge indicator) in future.

- **`embedding_alpha` smoothing vs. spatial movement**: With `alpha=0.5`, the EMA-smoothed embedding converges slowly to a new track (~5 embedding cycles / ~5 seconds). The dot moves gradually between positions. Setting `alpha=1.0` would give snappier spatial jumps at track changes but more jitter within a track.
