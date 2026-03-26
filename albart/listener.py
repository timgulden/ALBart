"""ALBart Listener — captures live audio, identifies tracks, drives LED display.

Captures audio via the system input device (e.g., BlackHole), computes CLAP
embeddings, queries the FAISS index, and drives the 32×32 LED display.
Broadcasts embeddings to the MapView and DJ via UDP.

Usage:
    python -m albart.listener
    python -m albart.listener --device "BlackHole"
"""

from albart.runtime.run import main

if __name__ == "__main__":
    main()
