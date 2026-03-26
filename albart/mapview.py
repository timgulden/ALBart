"""ALBart MapView — 3D neighborhood visualization of the embedding space.

Renders album art thumbnails in a rotating 3D cloud, centered on the
currently playing track.  Receives embeddings via UDP from either the
Listener (live audio) or the DJ (stored embeddings).

Usage:
    python -m albart.mapview
    python -m albart.mapview --mode half --view neighborhood
    python -m albart.mapview --mode half --view umap
"""

from albart.runtime.run_map import main

if __name__ == "__main__":
    main()
